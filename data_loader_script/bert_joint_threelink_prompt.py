'''
prompt joint three link

'''

import sys
import os
sys.path.append('./code/')
from data_process_script.data_process import *
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.data import Dataset

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

class JointThreeLinkPromptDataset(Dataset):

    def __init__(self, oridocdic,docid2docname, config, mod=None):
        # bert_name = 'bert-base-uncased'
        # config.train_CP_dir config.train_ANC_dir config.train_RFC_dir
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
        # self.spatial_ele_label2id = config.spatial_ele_label2id 
        # self.spatial_ele_id2label = {_id: _label for _label, _id in config.spatial_ele_label2id.items()}
        self.oridocdic = oridocdic
        self.mod = mod
        self.dataset = self.preprocess(self.oridocdic,docid2docname)
        # self.word_pad_idx = word_pad_idx
        # self.label_pad_idx = label_pad_idx
        self.device = config.device
        
    def preprocess(self,oridocdic,docid2docname):
        # oridocdic,docid2docname=get_ori_doc_info(self.config.train_CP_dir)
        all_seq_max_len=self.get_seq_tok_all_doc(oridocdic,self.tokenizer,docid2docname)
        # self.get_seq_tok_lab_all_doc(oridocdic,self.tokenizer)
        self.get_link_candidate(oridocdic,self.tokenizer)
        # all_seq_lis=[]
        all_link_seq_lis=[]
        # all_link_lis=[]
        all_e1_mask_lis=[]
        all_e2_mask_lis=[]
        all_tr_mask_lis=[]

        all_lab_lis=[]
        
        for k,v in self.oridocdic.items():
            if(len(v.all_link_candidate)==0):
                continue
            all_link_seq_lis+=v.all_link_seqs
            all_e1_mask_lis+=v.all_e1_sted
            all_e2_mask_lis+=v.all_e2_sted
            all_tr_mask_lis+=v.all_tr_sted
            all_lab_lis+=v.all_link_label

        data=[]
        for seq,e1msk,e2msk,trmsk,lab in zip(all_link_seq_lis,all_e1_mask_lis,all_e2_mask_lis,all_tr_mask_lis,all_lab_lis):
            data.append((seq,e1msk,e2msk,trmsk,lab))
        if self.mod == 'train':
            # print(len(data))
            split_n = len(data)*3//4
            return data[:split_n]
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        seq = self.dataset[idx][0]
        # link = self.dataset[idx][1]
        e1msk=self.dataset[idx][1]
        e2msk=self.dataset[idx][2]
        trmsk=self.dataset[idx][3]
        lab = self.dataset[idx][4]
        return [seq,e1msk,e2msk,trmsk,lab]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def get_seq_tok_one_doc(self,doc,tokenizer):
        curseqtoklis=[]
        curseqtokidslis=[]
        doc_max_seq_len=0
        for seq in doc.seqlis:
            curseqtok=tokenizer.tokenize(seq)
            doc_max_seq_len=max(doc_max_seq_len,len(curseqtok))
            curseqtoklis.append(curseqtok)
            curseqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))
        
        doc.seqtoklis=curseqtoklis
        doc.seqtokidslis=curseqtokidslis
        return doc_max_seq_len

    

    def get_seq_tok_all_doc(self,oridocdic,tokenizer,docid2docname):

        all_max_seq_len=0
        for k,v in oridocdic.items():
            curmaxlen=self.get_seq_tok_one_doc(v,tokenizer)

            all_max_seq_len=max(all_max_seq_len,curmaxlen)

        return all_max_seq_len

    def legal(self,ele):
        if int(ele.start) ==-1 or not hasattr(ele,'seqstid'):
            return False
        return True

    def get_tok_lis_from_ele(self,ele,char2tok_spanlis):
        eletoklis=[] 
        seqids=ele.seqstid[0]
        st=ele.seqstid[1] 
        ed=ele.seqedid[1]
        char2tok_span=char2tok_spanlis[seqids]
        for i in range(st,ed):
            if(char2tok_span[i][0] not in eletoklis and char2tok_span[i][0]!=-1 and char2tok_span[i][1]!=-1):
                eletoklis.append(char2tok_span[i][0])
        return eletoklis
    
    def get_link_one_doc(self,doc,char2tok_spanlis):
        gold_link=[]
        bad_link=[]
        all_link_seqs_good=[]
        all_link_seqs_bad=[]
        all_link_labs_good=[]
        all_link_labs_bad=[]
        good_link_prob_type=[]
        bad_link_prob_type=[]

        
        
        
        qslink_tj=[]
        qslink_ld=[]
        qslink_tr=[]
        #### 
        for lin in doc.qslink_lis:
            #
            if(len(lin.trajector)==0 or len(lin.landmark)==0 or len(lin.trigger)==0):
                continue
            tj=doc.id2obj[lin.trajector]
            ld=doc.id2obj[lin.landmark]
            tr=doc.id2obj[lin.trigger]
            if(self.legal(tj) and self.legal(ld) and self.legal(tr)):
                #
                if(tj.seqstid[0]!=ld.seqstid[0] or tj.seqstid[0]!=tr.seqstid[0] or ld.seqstid[0]!=tr.seqstid[0]):
                    
                    continue
                seqids=tj.seqstid[0]
                tjtoklis=self.get_tok_lis_from_ele(tj,char2tok_spanlis)
                ldtoklis=self.get_tok_lis_from_ele(ld,char2tok_spanlis)
                trtoklis=self.get_tok_lis_from_ele(tr,char2tok_spanlis)
                qslink_tj.append([[seqids],tjtoklis])
                qslink_ld.append([[seqids],ldtoklis])
                qslink_tr.append([[seqids],trtoklis])
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                all_link_labs_good.append(1)
                good_link_prob_type.append(1)

        ##############################
        #### 
        for ctr in qslink_tr:
            for ctj in qslink_tj:
                for cld in qslink_ld:
                    if(ctr[0] != ctj[0] or ctr[0]!=cld[0] or cld[0]!=ctj[0]):
                        continue
                    if([ctr[0],ctr[1],ctj[1],cld[1]] in gold_link or [ctr[0],ctr[1],ctj[1],cld[1]] in bad_link):
                        continue
                    if(ctj[1]==cld[1]):
                        continue
                    
                    all_link_seqs_bad.append(doc.seqtokidslis[ctr[0][0]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    all_link_labs_bad.append(0)
                    bad_link_prob_type.append(1)
        ##############################

        olink_tj=[]
        olink_ld=[]
        olink_tr=[]
        #### 
        for lin in doc.olink_lis:
            
            if(len(lin.trajector)==0 or len(lin.landmark)==0 or len(lin.trigger)==0):
                continue
            tj=doc.id2obj[lin.trajector]
            ld=doc.id2obj[lin.landmark]
            tr=doc.id2obj[lin.trigger]
            if(self.legal(tj) and self.legal(ld) and self.legal(tr)):
                
                if(tj.seqstid[0]!=ld.seqstid[0] or tj.seqstid[0]!=tr.seqstid[0] or ld.seqstid[0]!=tr.seqstid[0]):
                    
                    continue
                seqids=tj.seqstid[0]
                tjtoklis=self.get_tok_lis_from_ele(tj,char2tok_spanlis)
                ldtoklis=self.get_tok_lis_from_ele(ld,char2tok_spanlis)
                trtoklis=self.get_tok_lis_from_ele(tr,char2tok_spanlis)
                olink_tj.append([[seqids],tjtoklis])
                olink_ld.append([[seqids],ldtoklis])
                olink_tr.append([[seqids],trtoklis])
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                all_link_labs_good.append(2)
                good_link_prob_type.append(2)

        ##############################
        ####
        for ctr in olink_tr:
            for ctj in olink_tj:
                for cld in olink_ld:
                    if(ctr[0] != ctj[0] or ctr[0]!=cld[0] or cld[0]!=ctj[0]):
                        continue
                    if([ctr[0],ctr[1],ctj[1],cld[1]] in gold_link or [ctr[0],ctr[1],ctj[1],cld[1]] in bad_link):
                        continue
                    if(ctj[1]==cld[1]):
                        continue
                    
                    all_link_seqs_bad.append(doc.seqtokidslis[ctr[0][0]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    all_link_labs_bad.append(0)
                    bad_link_prob_type.append(2)
        ##############################

        movelink_tj=[]
        movelink_ld=[]
        movelink_tr=[]
        #### 
        for lin in doc.movelink_lis:
            
            if(len(lin.mover)==0 or len(lin.goal)==0 or len(lin.trigger)==0):
                continue
            tj=doc.id2obj[lin.mover]
            ld=doc.id2obj[lin.goal]
            tr=doc.id2obj[lin.trigger]
            if(self.legal(tj) and self.legal(ld) and self.legal(tr)):
                
                if(tj.seqstid[0]!=ld.seqstid[0] or tj.seqstid[0]!=tr.seqstid[0] or ld.seqstid[0]!=tr.seqstid[0]):
                    #
                    continue
                seqids=tj.seqstid[0]
                tjtoklis=self.get_tok_lis_from_ele(tj,char2tok_spanlis)
                ldtoklis=self.get_tok_lis_from_ele(ld,char2tok_spanlis)
                trtoklis=self.get_tok_lis_from_ele(tr,char2tok_spanlis)
                movelink_tj.append([[seqids],tjtoklis])
                movelink_ld.append([[seqids],ldtoklis])
                movelink_tr.append([[seqids],trtoklis])
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                all_link_labs_good.append(3)
                good_link_prob_type.append(3)

        ##############################
        ####
        for ctr in movelink_tr:
            for ctj in movelink_tj:
                for cld in movelink_ld:
                    if(ctr[0] != ctj[0] or ctr[0]!=cld[0] or cld[0]!=ctj[0]):
                        continue
                    if([ctr[0],ctr[1],ctj[1],cld[1]] in gold_link or [ctr[0],ctr[1],ctj[1],cld[1]] in bad_link):
                        continue
                    if(ctj[1]==cld[1]):
                        continue
                    
                    all_link_seqs_bad.append(doc.seqtokidslis[ctr[0][0]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    all_link_labs_bad.append(0)
                    bad_link_prob_type.append(3)
        ##############################


        
        doc.all_link_seqs=all_link_seqs_good+all_link_seqs_bad
        doc.all_link_candidate=gold_link+bad_link
        all_e1_sted=[]
        all_e2_sted=[]
        all_tr_sted=[]
        # ans=0
        for idx,lin in enumerate(doc.all_link_candidate):
            if(lin[1][0]==-1):
                all_tr_sted.append([0,0])
            else:
                all_tr_sted.append([lin[1][0],lin[1][-1]])
            
            all_e1_sted.append([lin[2][0],lin[2][-1]])

            all_e2_sted.append([lin[3][0],lin[3][-1]])
        # print(ans)
        doc.all_e1_sted=all_e1_sted
        doc.all_e2_sted=all_e2_sted
        doc.all_tr_sted=all_tr_sted

        for seqidx in range(len(doc.all_link_seqs)):
            prompt_token=self.tokenizer.convert_tokens_to_ids(\
                self.tokenizer.tokenize('in this sequence')
                )+ doc.all_link_seqs[seqidx][doc.all_e1_sted[seqidx][0]:doc.all_e1_sted[seqidx][1]+1]\
                +doc.all_link_seqs[seqidx][doc.all_tr_sted[seqidx][0]:doc.all_tr_sted[seqidx][1]+1]\
                +doc.all_link_seqs[seqidx][doc.all_e2_sted[seqidx][0]:doc.all_e2_sted[seqidx][1]+1]\
                + self.tokenizer.convert_tokens_to_ids(\
                    self.tokenizer.tokenize('form a [MASK]')
                )
            # assert len(prompt_token) <= 17 , print('error',prompt_token)
            # print(len(doc.all_link_seqs[seqidx]))
            if(len(doc.all_link_seqs[seqidx])+len(prompt_token)>=512):
                dis=len(doc.all_link_seqs[seqidx])+len(prompt_token) - 512
                doc.all_link_seqs[seqidx] = doc.all_link_seqs[seqidx][:dis] + prompt_token
            else:
                doc.all_link_seqs[seqidx] = doc.all_link_seqs[seqidx] + prompt_token

        # doc.all_link_candidate=gold_link
        doc.all_link_label=all_link_labs_good + all_link_labs_bad
        doc.link_prob_type = good_link_prob_type + bad_link_prob_type

        



    def get_link_candidate(self,oridocdic,tokenizer):

        for k,v in oridocdic.items():
            char2tok_spanlis=self.get_char2tok_spanlis_one_doc(v,tokenizer)
            self.get_link_one_doc(v,char2tok_spanlis)
        

    def get_char2tok_spanlis_one_doc(self,doc,tokenizer):
        res=[]
        for seq in doc.seqlis:

            token_span = tokenizer.encode_plus(seq, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
            
            char_num = None
            for tok_ind in range(len(token_span) - 1, -1, -1):
                if token_span[tok_ind][1] != 0:
                    char_num = token_span[tok_ind][1]
                    break
            
            char2tok_span = [[-1, -1] for _ in range(char_num)] 
            for tok_ind, char_sp in enumerate(token_span):
                for char_ind in range(char_sp[0], char_sp[1]):
                    tok_sp = char2tok_span[char_ind]
                    
                    if tok_sp[0] == -1:
                        tok_sp[0] = tok_ind
                    tok_sp[1] = tok_ind + 1 
            
            res.append(char2tok_span)
        
        return res
            
    
    
    def collate_fn(self, batch):
        
        seqs = [x[0] for x in batch]
        e1msk = [x[1] for x in batch]
        e2msk = [x[2] for x in batch]
        trmsk = [x[3] for x in batch]
        labs = [x[4] for x in batch]

        
        batch_len = len(seqs)
        max_len = max([len(s) for s in seqs])
        batch_data=[[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_data[j][:cur_len] = seqs[j]
        
        
        batch_data = torch.tensor(batch_data, dtype=torch.long).to(self.device)
        batch_e1_mask = torch.tensor(e1msk,dtype=torch.long).to(self.device)
        batch_e2_mask = torch.tensor(e2msk,dtype=torch.long).to(self.device)
        batch_tr_mask = torch.tensor(trmsk,dtype=torch.long).to(self.device)
        batch_labs=torch.tensor(labs,dtype=torch.long).to(self.device)

        return {
            'datas':batch_data,
            'y_labels':batch_labs,
            'e1_mask':batch_e1_mask,
            'e2_mask':batch_e2_mask,
            'tr_mask':batch_tr_mask
        }
    
if __name__ == '__main__':
    processor = Processor(config)
    processor.process()

    dev_dataset = JointThreeLinkPromptDataset(processor.train_oridocdic,processor.train_docid2docname, config)





