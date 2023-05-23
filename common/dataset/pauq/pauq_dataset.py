import torch
from torch.utils.data import Dataset

from CL.common import types


class Text2SQLDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        BOS = types.SpecialTokens.BOS.value
        EOS = types.SpecialTokens.EOS.value

        text = self.src_data[idx]
        sql = self.trg_data[idx]

        text_tokens = [BOS] + text + [EOS]
        sql_tokens = [BOS] + sql + [EOS]

        return torch.tensor(text_tokens), torch.tensor(sql_tokens)
