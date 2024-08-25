import torch
from torch import nn
import torch.nn.functional as F

from model_components import SeqEncoder, AttentionPoolBlock, EmbeddingLayer


class SingleLabelSelfAttentionModel(nn.Module):
    def __init__(self, **config):
        super().__init__()

        self.drug_vocab_size = config['drug_vocab_size']
        self.max_drug_seq = config['max_drug_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']

        self.drug_embedding_layer = EmbeddingLayer(self.drug_vocab_size, self.emb_size, self.max_drug_seq,
                                                   self.dropout_rate)

        transformer_encode_layer = nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=1024,
                                                              dropout=self.dropout_rate,
                                                              batch_first=True, activation=nn.ReLU())
        self.transformer_encoder = nn.TransformerEncoder(transformer_encode_layer,
                                                         num_layers=3,
                                                         enable_nested_tensor=False)

        self.fc_first = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.query = nn.Parameter(torch.randn(64, 256))

        self.conv1d_1 = nn.Conv1d(256, 32, kernel_size=1, stride=1)
        self.conv1d_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.conv1d_3 = nn.Conv1d(64, 32, kernel_size=5, stride=1)

        self.conv_layer_norm = nn.LayerNorm(58)

        self.attention_vector = nn.Parameter(torch.randn(1, 58))

        self.fc = nn.Sequential(
            nn.Linear(58, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(16, 1)
        )

    def forward(self, drug, drug_mask):
        drug_mask = drug_mask.to(torch.bool)
        drug_emb = self.drug_embedding_layer(drug)

        x = self.transformer_encoder(drug_emb, src_key_padding_mask=drug_mask)
        x = self.fc_first(x)

        x = F.scaled_dot_product_attention(torch.unsqueeze(self.query, 0), x, x,
                                           ~(drug_mask.unsqueeze(1).repeat_interleave(self.query.shape[0], dim=1)))


        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = F.relu(x)
        x = self.conv_layer_norm(x)

        x = F.scaled_dot_product_attention(torch.unsqueeze(self.attention_vector, 0), x, x)
        x = torch.squeeze(x, 1)

        out = self.fc(x)

        return out


class SignleLabelModel(nn.Module):
    def __init__(self, **config):
        super().__init__()

        self.drug_max_len = config['max_drug_seq']
        self.vocab_size = config['drug_vocab_size']
        self.emb_size = config['emb_size']
        self.token_mixer_name = config['token_mixer_name']
        self.dropout_rate = config['dropout_rate']
        self.use_pre_ln = config['use_pre_ln']
        self.use_double_route_residue = config['use_double_route_residue']

        self.seq_encoder = SeqEncoder(self.vocab_size, self.emb_size, 512, 4, 3,
                                      self.drug_max_len, self.dropout_rate,
                                      use_bias=True,
                                      token_mixer_name=self.token_mixer_name,
                                      use_pre_ln=self.use_pre_ln,
                                      use_double_route_residue=self.use_double_route_residue)

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv1d(512, 256, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=True)
        )
        self.conv1d_layer_norm = nn.LayerNorm(256)

        self.pool = AttentionPoolBlock(64, 1)

        self.classifier = nn.Sequential(
            nn.LazyLinear(128, bias=True),
            nn.GELU(approximate='tanh'),
            nn.LazyLinear(64, bias=True),
            nn.GELU(approximate='tanh'),
            nn.LazyLinear(1, bias=True)
        )

    def forward(self, drugs, valid_lens):
        x = self.seq_encoder(drugs, valid_lens)

        x = x.permute(0, 2, 1)
        x = self.conv1d_layer(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d_layer_norm(x)

        x = torch.squeeze(self.pool(x, valid_lens))

        out = self.classifier(x)

        return out


class Tox21MultiLabelModel(nn.Module):
    def __init__(self, **config):
        super().__init__()

        self.drug_max_len = config['max_drug_seq']
        self.vocab_size = config['drug_vocab_size']
        self.emb_size = config['emb_size']
        self.token_mixer_name = config['token_mixer_name']
        self.dropout_rate = config['dropout_rate']
        self.use_pre_ln = config['use_pre_ln']
        self.use_double_route_residue = config['use_double_route_residue']

        self.seq_encoder = SeqEncoder(self.vocab_size, self.emb_size, 512, 4, 3,
                                      self.drug_max_len, self.dropout_rate,
                                      use_bias=True,
                                      token_mixer_name=self.token_mixer_name,
                                      use_pre_ln=self.use_pre_ln,
                                      use_double_route_residue=self.use_double_route_residue)

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv1d(512, 256, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=True)
        )
        self.conv1d_layer_norm = nn.LayerNorm(256)

        self.pool = AttentionPoolBlock(64, 1)

        self.classifier = nn.Sequential(
            nn.LazyLinear(128, bias=True),
            nn.GELU(approximate='tanh'),
            nn.LazyLinear(12, bias=True)
        )

    def forward(self, drugs, valid_lens):
        x = self.seq_encoder(drugs, valid_lens)

        x = x.permute(0, 2, 1)
        x = self.conv1d_layer(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d_layer_norm(x)

        x = torch.squeeze(self.pool(x, valid_lens))

        out = self.classifier(x)

        return out


class ToxCastMultiLabelModel(nn.Module):
    def __init__(self, **config):
        super().__init__()

        self.drug_max_len = config['max_drug_seq']
        self.vocab_size = config['drug_vocab_size']
        self.emb_size = config['emb_size']
        self.token_mixer_name = config['token_mixer_name']
        self.dropout_rate = config['dropout_rate']
        self.use_pre_ln = config['use_pre_ln']
        self.use_double_route_residue = config['use_double_route_residue']

        self.seq_encoder = SeqEncoder(self.vocab_size, self.emb_size, 512, 4, 3,
                                      self.drug_max_len, self.dropout_rate,
                                      use_bias=True,
                                      token_mixer_name=self.token_mixer_name,
                                      use_pre_ln=self.use_pre_ln,
                                      use_double_route_residue=self.use_double_route_residue)

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=True)
        )
        self.conv1d_layer_norm = nn.LayerNorm(1024)

        self.pool = AttentionPoolBlock(512, 1)

        self.classifier = nn.Sequential(
            nn.LazyLinear(617, bias=True)
        )

    def forward(self, drugs, valid_lens):
        x = self.seq_encoder(drugs, valid_lens)

        x = x.permute(0, 2, 1)
        x = self.conv1d_layer(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d_layer_norm(x)

        x = torch.squeeze(self.pool(x, valid_lens))

        out = self.classifier(x)

        return out
