
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model import BertClassificationModel
from dataset import SMPData

class Coach:
    def __init__(self, num_labels=6):
        self.device = torch.device('cuda')
        torch.cuda.set_device(1)
        
        # 超参数设置
        self.epochs = 2
        self.batch_size = 64
        self.num_labels = num_labels

        # 模型初始化
        self.model = BertClassificationModel(num_labels=num_labels).to(self.device)
    
        # 初始化数据集
        self.train_dataset = SMPData(act='train')
        self.dev_dataset = SMPData(act='dev', it=4)
        
        # 损失及优化器初始化
        self.criterion = nn.CrossEntropyLoss().to(self.device).eval()
        self.optimizer = AdamW(self.model.parameters(), lr=0.0001)
        
        # 模型最终保存位置
        self.save_path = './checkpoints/bert.pth'
        
    def train(self):
        best_acc = 0
        correct = 0
        total = 0
        
        for epoch in range(self.epochs):

            self.model.train()
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
            
            for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):

                input_ids, token_type_ids, attention_mask, labels = input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                
                output =  self.model(input_ids, token_type_ids, attention_mask)
                loss = self.criterion(output.view(-1, self.num_labels), labels.view(-1))

                _, predict = torch.max(output.data, 1)
                correct += (predict == labels).sum().item()
                total += labels.size(0)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                if (step + 1) % 100 == 0:
                    train_acc = correct / total
                    print(f'loss: {float(loss):.3f} train_acc: {float(train_acc):.3f}')

                #每500次进行一次验证
                if (step + 1) % 500 == 0:
                    train_acc = correct / total
                    acc = self.dev()
                    if best_acc < acc:
                        best_acc = acc
                        #模型保存路径
                        torch.save({
                            'bert': self.model.state_dict(),
                        }, self.save_path)
                    print(f'best_acc:{float(best_acc):.3f}')
                    self.model.train()

    # 验证
    def dev(self):
        self.model.eval()
        dev_loader = DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            correct = 0
            total = 0
            for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dev_loader):
                input_ids, token_type_ids, attention_mask, labels = input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                output = self.model(input_ids,token_type_ids,attention_mask)
                _, predict = torch.max(output.data, 1)
                correct += (predict==labels).sum().item()
                total += labels.size(0)
            acc = correct / total
            return acc

if __name__ == '__main__':
    coach = Coach()
    coach.train()





