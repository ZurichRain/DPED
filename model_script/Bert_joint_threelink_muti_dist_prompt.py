import sys
import os
sys.path.append('./code/')
from torch.nn.modules import loss
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torchcrf import CRF
import config_script.config as config
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor


class BertBaseJointPromptMutiDist_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(BertBaseJointPromptMutiDist_Model,self).__init__()

        
        self.num_labels1=4
        
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        
        self.tanh=nn.Tanh()
        self.outlin1=nn.Linear(4*768,self.num_labels1)
        torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.drop1=nn.Dropout(p=0.15) 
        self.loss1 = nn.CrossEntropyLoss()
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=768)



    def forward(self, datas,e1_mask,e2_mask,tr_mask,y_labels):
        
        bertout=self.bertmodel(datas)
        
        wemb=bertout[0]
        cure1_emb=self.span_extractor(wemb,e1_mask.unsqueeze(1))
        cure2_emb=self.span_extractor(wemb,e2_mask.unsqueeze(1))
        curetr_emb=self.span_extractor(wemb,tr_mask.unsqueeze(1))

        e1_e2 = torch.abs(cure1_emb - cure2_emb)
        e1_tr = torch.abs(cure1_emb - curetr_emb)
        e2_tr = torch.abs(cure2_emb - curetr_emb)
        mask_emb=wemb[:,-1,:]
        
        activate_emb=torch.cat((mask_emb.unsqueeze(1),e1_e2,e1_tr,e2_tr),dim=-1)
        
        activate_emb=self.drop1(activate_emb) 
        activate_emb=self.outlin1(activate_emb)
        
        
        wemb1=activate_emb.view(-1,self.num_labels1)
        ctrain_y=y_labels.view(-1)
        l1=self.loss1(wemb1,ctrain_y)
        return l1,wemb1