import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SMPData(Dataset):
    def __init__(self, act='train', type_='usual', it=5):
        assert act in ['train', 'dev', 'test']
        assert type_ in ['usual', 'virus']  # usual是通用数据集，virus是疫情有关的
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.label_dict = json.load(open('./data/label_dict.json'))
        
        root_dir = f'./data'
        data = pd.read_csv(f'{root_dir}/{act}.csv')
        for i in range(it):
            temp_data = pd.read_csv(f'{root_dir}/{act}{i}.csv')
            data = pd.concat((data, temp_data[temp_data['type'] == 'usual']))

        self.all_data = data
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.all_data[idx:idx+1]
        if pd.isna(data['content'].item()):
            return self[idx-1]
        input_ids, token_type_ids, attention_mask = self.encoder(data['content'].item())
        labels = self.label_dict[data['labels'].item()]
        return input_ids, token_type_ids, attention_mask, torch.tensor(labels)
    
    
    def encoder(self, text_list, max_len=150):
        tokenizer = self.tokenizer(
            text_list,
            padding = 'max_length',
            truncation = True,
            max_length = max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
            )
        input_ids = tokenizer['input_ids'].view(-1)
        token_type_ids = tokenizer['token_type_ids'].view(-1)
        attention_mask = tokenizer['attention_mask'].view(-1)
        return input_ids, token_type_ids, attention_mask

    def decoder(self, input_ids):
        return self.tokenizer.decode(input_ids[0])
