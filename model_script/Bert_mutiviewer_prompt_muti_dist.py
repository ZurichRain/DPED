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


class BertBaseJointMutiviewerPromptMutiDist_Model(nn.Module):


    def __init__(self):
        super(BertBaseJointMutiviewerPromptMutiDist_Model,self).__init__()

        
        
        self.num_labels1=4
        self.num_token=30522
        
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        
        self.tanh=nn.Tanh()
        self.outlin1=nn.Linear(4*768,self.num_labels1)
        self.trigger_outlin = nn.Linear(768,self.num_token)
        torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.drop1=nn.Dropout(p=0.15) 
        self.loss1 = nn.CrossEntropyLoss()
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=768)
        



    def forward(self, datas,e1_mask,e2_mask,tr_mask,y_labels, conf_seqs, conf_labs):
        
        bertout=self.bertmodel(datas)
        bert_conf_out = self.bertmodel(conf_seqs)[0]
        bert_conf_out = self.trigger_outlin(bert_conf_out)
        
        labels = torch.where(conf_seqs == 103, conf_labs, -100)
        conf_loss = self.loss1(bert_conf_out.view(-1,30522),labels.view(-1))
        
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
        return l1+conf_loss,wemb1