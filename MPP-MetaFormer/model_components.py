import math

import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=0.0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len, dropout):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class WordEmbeddingAndPosEncoding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len, dropout)

    def forward(self, input_ids):
        return self.pos_encoding(self.embedding(input_ids) * math.sqrt(self.hidden_size))


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_size, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position_size, embedding_dim)

        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class AddAndLayerNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class FFNLayer(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.activation = nn.GELU(approximate='tanh')
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))


class DoubleRouteResidueFFNLayer(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.LazyLinear(ffn_num_outputs),
                                   nn.Sigmoid())
        self.mlp = nn.Sequential(nn.LazyLinear(ffn_num_hiddens),
                                 nn.GELU(approximate='tanh'),
                                 nn.LazyLinear(ffn_num_outputs))
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(ffn_num_outputs)

    def forward(self, X):
        gate = self.gate(X)
        x_mlp = self.dropout(self.mlp(X))
        x_left = self.ln(X + x_mlp)
        x_right = X + x_mlp
        return gate * x_left + (1 - gate) * x_right


class SeqEncoderBlockPre(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, token_mixer_name='Self-Attention'):
        super().__init__()
        self.token_mixer_name = token_mixer_name
        if self.token_mixer_name == 'Self-Attention':
            self.token_mixer = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        elif self.token_mixer_name == 'LSTM':
            self.token_mixer = nn.LSTM(num_hiddens, 512 // 2, 1, use_bias, True, dropout, True, num_hiddens // 2)
        elif self.token_mixer_name == 'GRU':
            self.token_mixer = nn.GRU(num_hiddens, num_hiddens // 2, 1, use_bias, True, dropout, True)
        elif self.token_mixer_name == 'CNN':
            self.token_mixer = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.ln = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFNLayer(ffn_num_hiddens, num_hiddens)

    def forward(self, X, valid_lens):
        Y = self.ln(X)
        if self.token_mixer_name == 'Self-Attention':
            Y = self.token_mixer(Y, Y, Y, valid_lens)
        elif self.token_mixer_name == 'LSTM' or self.token_mixer_name == 'GRU':
            Y = self.token_mixer(Y)[0]
        elif self.token_mixer_name == 'CNN':
            Y = Y.permute(0, 2, 1)
            Y = self.token_mixer(Y)
            Y = Y.permute(0, 2, 1)
        else:
            raise ValueError()
        Y = self.dropout(Y)
        Y = X + Y

        X = Y
        Y = self.ln(X)
        Y = self.ffn(Y)
        Y = self.dropout(Y)
        Y = X + Y

        return Y


class SeqEncoderBlockPost(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, token_mixer_name='Self-Attention'):
        super().__init__()
        self.token_mixer_name = token_mixer_name
        if self.token_mixer_name == 'Self-Attention':
            self.token_mixer = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        elif self.token_mixer_name == 'LSTM':
            self.token_mixer = nn.LSTM(num_hiddens, 512 // 2, 1, use_bias, True, dropout, True, num_hiddens // 2)
        elif self.token_mixer_name == 'GRU':
            self.token_mixer = nn.GRU(num_hiddens, num_hiddens // 2, 1, use_bias, True, dropout, True)
        elif self.token_mixer_name == 'CNN':
            self.token_mixer = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.addnorm1 = AddAndLayerNorm(num_hiddens, dropout)
        self.ffn = FFNLayer(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddAndLayerNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        if self.token_mixer_name == 'Self-Attention':
            Y = self.addnorm1(X, self.token_mixer(X, X, X, valid_lens))
        elif self.token_mixer_name == 'LSTM' or self.token_mixer_name == 'GRU':
            Y = self.addnorm1(X, self.token_mixer(X)[0])
        elif self.token_mixer_name == 'CNN':
            X_processed = X.permute(0, 2, 1)
            X_processed = self.token_mixer(X_processed)
            X_processed = X_processed.permute(0, 2, 1)
            Y = self.addnorm1(X, X_processed)
        else:
            raise ValueError()
        return self.addnorm2(Y, self.ffn(Y))


class DoubleRouteResidueSeqEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, token_mixer_name='Self-Attention'):
        super().__init__()
        self.token_mixer_name = token_mixer_name
        if self.token_mixer_name == 'Self-Attention':
            self.token_mixer = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        elif self.token_mixer_name == 'LSTM':
            self.token_mixer = nn.LSTM(num_hiddens, 512 // 2, 1, use_bias, True, dropout, True, num_hiddens // 2)
        elif self.token_mixer_name == 'GRU':
            self.token_mixer = nn.GRU(num_hiddens, num_hiddens // 2, 1, use_bias, True, dropout, True)
        elif self.token_mixer_name == 'CNN':
            self.token_mixer = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.gate = nn.Sequential(nn.LazyLinear(num_hiddens),
                                   nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(num_hiddens)
        self.double_route_residue_ffn = DoubleRouteResidueFFNLayer(ffn_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens):
        gate = self.gate(X)

        if self.token_mixer_name == 'Self-Attention':
            Y = self.token_mixer(X, X, X, valid_lens)
        elif self.token_mixer_name == 'LSTM' or self.token_mixer_name == 'GRU':
            Y = self.token_mixer(X)[0]
        elif self.token_mixer_name == 'CNN':
            Y = X.permute(0, 2, 1)
            Y = self.token_mixer(Y)
            Y = Y.permute(0, 2, 1)
        else:
            raise ValueError()

        Y = self.dropout(Y)
        Y_left = self.ln(X + Y)
        Y_right = X + Y
        Y = gate * Y_left + (1 - gate) * Y_right

        return self.double_route_residue_ffn(Y)


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, max_len, dropout,
                 use_bias=False,
                 token_mixer_name='Self-Attention',
                 use_pre_ln = False,
                 use_double_route_residue=False):
        super().__init__()
        self.token_mixer_name = token_mixer_name
        self.num_hiddens = num_hiddens
        self.pos_plus_embedding = WordEmbeddingAndPosEncoding(vocab_size, num_hiddens, max_len, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            if not use_double_route_residue:
                if not use_pre_ln:
                    self.blks.add_module("block" + str(i),
                                         SeqEncoderBlockPost(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, self.token_mixer_name))
                else:
                    self.blks.add_module("block" + str(i),
                                         SeqEncoderBlockPre(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, self.token_mixer_name))
            else:
                self.blks.add_module("block" + str(i),
                                     DoubleRouteResidueSeqEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, self.token_mixer_name))

    def forward(self, X, valid_lens):
        X = self.pos_plus_embedding(X)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X


class AttentionPoolBlock(nn.Module):
    def __init__(self, hidden_size, query_num):
        super().__init__()
        self.W_p1 = nn.LazyLinear(hidden_size, False)
        self.W_p2 = nn.LazyLinear(query_num, False)

    def forward(self, x, valid_lens):
        attention_matrix = self.W_p2(F.tanh(self.W_p1(x)))
        mask = valid_lens.unsqueeze(-1).unsqueeze(-1).expand_as(attention_matrix)
        a_tensor = (torch.arange(1, attention_matrix.size(1) + 1, dtype=torch.float32, device=x.device)
                    .repeat(attention_matrix.size(0), attention_matrix.size(-1), 1).transpose(-1, -2))
        mask = a_tensor > mask
        attention_matrix[mask] = -1e6
        attention_matrix = F.softmax(attention_matrix, dim=-2)
        output = torch.bmm(x.permute(0, 2, 1), attention_matrix)
        return output.permute(0, 2, 1)
