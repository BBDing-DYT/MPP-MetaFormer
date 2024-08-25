import torch
import numpy as np
from torch.utils import data
from rdkit import Chem

from config import hyper_parameters_config_set

smiles_chars = ['!', 'H', "Li",  "Be", 'B', 'C', 'N', 'O', 'F', "Na", "Mg", "Al", "Si", 'P', 'S', "Cl", "K", "Ca", "Ti",
                "V", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "Ge", "As", "Se", "se", "Br", "Sr", "Zr", "Mo",
                "Pd", "Ag", "Cd", "Sb", 'I', "Ba", "Nd", "Eu", "Gd", "Dy", "Yb", "Pt",
                "Au", "Hg", "Tl", "Pb", "Bi",
                'c', 'n', 'o', 's',
                '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '(', ')', '[', ']', '=', '#', '@', '+', '-', '/', '\\', '*', '%', '^', ':', '.']

char_to_index = {c: i for i, c in enumerate(smiles_chars)}


def smiles_to_onehot(original_smiles):
    unified_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(original_smiles))
    max_length = hyper_parameters_config_set()['max_drug_seq']
    onehot = [0] * max_length
    mask = [1] * max_length

    current_char = 0
    idx = 0
    while idx < max_length and current_char < len(unified_smiles):
        if current_char + 1 >= len(unified_smiles):
            single_char = unified_smiles[current_char]
            onehot[idx] = char_to_index[single_char]
        else:
            double_char = unified_smiles[current_char : current_char + 2]
            if double_char in char_to_index:
                onehot[idx] = char_to_index[double_char]
                current_char += 1
            else:
                single_char = unified_smiles[current_char]
                onehot[idx] = char_to_index[single_char]
        mask[idx] = 0
        current_char += 1
        idx += 1

    return np.asarray(onehot), np.asarray(mask), idx


class SingleLabelDataFrameToDataset(data.Dataset):

    def __init__(self, raw_df):
        processed_df = raw_df[raw_df.iloc[:, 0].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
        self.df = processed_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drug_sequence = self.df.iloc[index]['SMILES']
        drug_one_hot_array, mask, valid_len = smiles_to_onehot(drug_sequence)
        drug_one_hot_array, mask, valid_len = torch.from_numpy(drug_one_hot_array).long(), torch.from_numpy(mask).long(), torch.tensor(valid_len).long()

        label = torch.tensor(self.df.Label.values[index]).long()

        return drug_one_hot_array, mask, valid_len, label


class MultiLabelDataFrameToDataSet(data.Dataset):

    def __init__(self, raw_df):
        processed_df = raw_df[raw_df.iloc[:, 0].apply(lambda x: Chem.MolFromSmiles(x) is not None)]  # 某些SMILES序列是错的，无法解析，删除这些数据
        self.df = processed_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drug_sequence = self.df.iloc[index]['SMILES']
        drug_one_hot_array, mask, valid_len = smiles_to_onehot(drug_sequence)
        drug_one_hot_array, mask, valid_len = torch.from_numpy(drug_one_hot_array).long(), torch.from_numpy(mask).long(), torch.tensor(valid_len).long()

        label = torch.from_numpy(self.df.iloc[index, 1:].values.astype(np.int64)).long()

        return drug_one_hot_array, mask, valid_len, label
