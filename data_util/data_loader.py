import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class LMGNNDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sent = item['sent']
        answer = item['answer']
        token_to_node = item['token_to_node']
        return idx, sent, answer, token_to_node
    
