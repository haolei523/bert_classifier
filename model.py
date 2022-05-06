import torch.nn as nn
from transformers import BertModel

class BertClassificationModel(nn.Module):
    def __init__(self, num_labels=6):
        super(BertClassificationModel, self).__init__()   

        self.bert = BertModel.from_pretrained('bert-base-chinese')

        # 全连接，用于调整最终输出类别数   
        self.classifier = nn.Linear(768, num_labels)  #bert默认的隐藏单元数是768， 输出单元是类别数，表示二分类
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[1]
        linear_output = self.classifier(bert_cls_hidden_state)
        return linear_output
    

