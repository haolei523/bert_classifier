import re

import torch
from torch.utils.data import DataLoader

from model import BertClassificationModel
from dataset import SMPData


class Predictor:
    def __init__(self, checkpoint_path='./checkpoints/bert.pth'):
        self.device = torch.device("cpu")

        # 加载预训练模型
        self.model = BertClassificationModel(num_labels=6).to(self.device)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['bert'])

    def interface(self, data):
        input_ids, token_type_ids, attention_mask, labels = data
        input_ids, token_type_ids, attention_mask, labels = input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        with torch.no_grad():
            output =  self.model(input_ids, token_type_ids, attention_mask)
            _, predict = torch.max(output.data, 1)
        return predict, labels


if __name__ == '__main__':
    # 加载测试数据集
    test_dataset = SMPData(act='test', it=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    data = iter(test_loader).next()

    p = Predictor()
    predict, labels = p.interface(data)
    label = [k for k, l in test_dataset.label_dict.items() if l == labels.item()][0]
    output = [k for k, l in test_dataset.label_dict.items() if l == predict.item()][0]
    senc = test_dataset.decoder(data[0])
    senc = senc.lstrip("[CLS]").rstrip(re.findall("\[SEP\].*\[PAD\]", senc)[0])
    print(f'输入：{senc}')
    print(f'预测：{output}')
    print(f'答案：{label}')