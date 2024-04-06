import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel





# BERTweet-LSTM
class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select,dropout):

        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        #tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
        #model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
        #tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
        #model = AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")
        self.bert = AutoModel.from_pretrained("bertweet-base")

        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear2 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 2)
        self.lstm = nn.LSTM(self.bert.config.hidden_size*2, self.bert.config.hidden_size,bidirectional=True)
        # self.MultiAttention = MultiHeadedAttention()
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, x_input_ids2):

        last_hidden = self.bert(input_ids=x_input_ids, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        last_hidden2 = self.bert(input_ids=x_input_ids2, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        ccc = self.bert.config.hidden_size*2
        print(ccc)
        query = last_hidden[0][:,0]
        query2 = last_hidden2[0][:,0]
        query = self.dropout(query)
        query2 = self.dropout(query2)
        context_vec = torch.cat((query, query2), dim=1)
        out1, h_n = self.lstm(context_vec)  # lstm层
        aaa = self.linear(query)
        linear = self.relu(aaa)
        out = self.out(linear)
        # linear2 = self.relu(self.linear2(context_vec))
        bbb = self.linear2(out1)
        linear2 = self.relu(bbb)
        out2 = self.out2(linear2)
        feature1 = out2.unsqueeze(1)
        feature1 = F.normalize(feature1, dim=2)
        feature2 = out2.unsqueeze(1)
        feature2 = F.normalize(feature2, dim=2)
        return out, out2, feature1, feature2