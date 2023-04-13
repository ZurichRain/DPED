


import os
import sys
# print(os.getcwd())
sys.path.append('./')
sys.path.append('./code/')

import numpy as np
import pandas as pd
import xml.dom.minidom as xmd
import logging
import config_script.config as config
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,T5Tokenizer
import spacy
import neuralcoref


from util_script.mata_data_calss import *

logging.basicConfig(level=logging.DEBUG)


class Processor(object):
    def __init__(self, config):
        # config.train_CP_dir config.train_ANC_dir config.train_RFC_dir
        self.train_data_dir = config.train_CP_dir
        self.vail_data_dir = config.vail_data_dir
        self.test_data_dir= config.test_data_dir
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config.T5base_pathname)
    def process(self):
        """
        process train and test data 
        """
        self.train_oridocdic,self.train_docid2docname=self.preprocess(self.train_data_dir) 
        self.vail_oridocdic,self.vail_docid2docname=self.preprocess(self.vail_data_dir)
        self.test_oridocdic,self.test_docid2docname=self.preprocess(self.test_data_dir)

    def get_attr_from_spatial_element(self,ele,ele_obj_type):
        ele_obj=ele_obj_type()
        for k,v in ele.attributes.items():
            curk=k.lower()
            ele_obj.update_attr(curk,v)
        return ele_obj
    def parse_data_from_xml(self,xml_file):
        id2obj=dict()
        res_doc=ori_data() 
        dom=xmd.parse(xml_file)
        root = dom.documentElement
        texts = root.getElementsByTagName('TEXT')
        for text in texts:  
            for child in text.childNodes:
                
                res_doc.text+=child.data
                res_doc.text+=' '


        places = dom.getElementsByTagName('PLACE')
        for p in places:
            aplace=place()
            for k,v in p.attributes.items():
                curk=k.lower()
                aplace.update_attr(curk,v)
            idv=p.getAttribute('id') 
            id2obj[idv]=aplace


            res_doc.place_lis.append(aplace)


        paths = dom.getElementsByTagName('PATH')
        for p in paths:
            apath=path()
            for k,v in p.attributes.items():
                curk=k.lower()
                apath.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=apath

            res_doc.path_lis.append(apath)


        spatial_entitys = dom.getElementsByTagName('SPATIAL_ENTITY')
        for p in spatial_entitys:
            aspatial_entity=spatial_entity()
            for k,v in p.attributes.items():
                curk=k.lower()
                aspatial_entity.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aspatial_entity
            
            res_doc.spatial_entity_lis.append(aspatial_entity)


        nonmotion_events = dom.getElementsByTagName('NONMOTION_EVENT')
        for p in nonmotion_events:
            anonmotion_event=nonmotion_event()
            for k,v in p.attributes.items():
                curk=k.lower()
                anonmotion_event.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=anonmotion_event

            res_doc.nonmotion_event_lis.append(anonmotion_event)

        motions = dom.getElementsByTagName('MOTION')
        for p in motions:
            amotion=motion()
            for k,v in p.attributes.items():
                curk=k.lower()
                amotion.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amotion

            res_doc.motion_lis.append(amotion)

        spatial_signals = dom.getElementsByTagName('SPATIAL_SIGNAL')
        for p in spatial_signals:
            aspatial_signal=spatial_signal()
            for k,v in p.attributes.items():
                curk=k.lower()
                aspatial_signal.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aspatial_signal
            res_doc.spatial_signal_lis.append(aspatial_signal)

        motion_signals = dom.getElementsByTagName('MOTION_SIGNAL')
        for p in motion_signals:
            amotion_signal=motion_signal()
            for k,v in p.attributes.items():
                curk=k.lower()
                amotion_signal.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amotion_signal
            res_doc.motion_signal_lis.append(amotion_signal)

        measures = dom.getElementsByTagName('MEASURE')
        for p in measures:
            ameasure=measure()
            for k,v in p.attributes.items():
                curk=k.lower()
                ameasure.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ameasure
            res_doc.measure_lis.append(ameasure)

        qslinks = dom.getElementsByTagName('QSLINK')
        for p in qslinks:
            aqslink=qslink()
            for k,v in p.attributes.items():
                curk=k.lower()
                aqslink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aqslink
            res_doc.qslink_lis.append(aqslink)

        olinks = dom.getElementsByTagName('OLINK')
        for p in olinks:
            aolink=olink()
            for k,v in p.attributes.items():
                curk=k.lower()
                aolink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aolink

            res_doc.olink_lis.append(aolink)
        
        movelinks = dom.getElementsByTagName('MOVELINK')
        for p in movelinks:
            amovelink=movelink()
            for k,v in p.attributes.items():
                curk=k.lower()
                amovelink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amovelink
            res_doc.movelink_lis.append(amovelink)

        measurelinks = dom.getElementsByTagName('MEASURELINK')
        for p in measurelinks:
            ameasurelink=measurelink()
            for k,v in p.attributes.items():
                curk=k.lower()
                ameasurelink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ameasurelink
            res_doc.measurelink_lis.append(ameasurelink)

        metalinks = dom.getElementsByTagName('METALINK')
        for p in metalinks:
            ametalink=metalink()
            for k,v in p.attributes.items():
                curk=k.lower()
                ametalink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ametalink
            res_doc.metalink_lis.append(ametalink)

        res_doc.id2obj=id2obj
        
        
        return res_doc
    def get_stedids_element_relation(self,doc):
        
        stedids2element=dict()
        element2stedids=dict()
        stedele_lis=[]
        for lin in doc.spatial_entity_lis:

            if(int(lin.start)==-1):
                
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.nonmotion_event_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.motion_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.spatial_signal_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.motion_signal_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.measure_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))
        
        for lin in doc.place_lis:
            
            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.path_lis:

            if(int(lin.start)==-1):
                continue
            
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        stedele_lis=sorted(stedele_lis)
        
        return stedids2element,element2stedids,stedele_lis
    
    def get_spatial_seqstedids(self,seqidslis,stedids2element,stedele_lis,seqid):
        one_seq_eles=[]
        for i in stedele_lis:
            if(i[1]>seqidslis[0] and i[0]>=seqidslis[0] and i[1]<=seqidslis[-1]+1):
                ele=stedids2element[i]
                ele.add_attr('seqstid',(seqid,int(ele.start)-seqidslis[0]))
                ele.add_attr('seqedid',(seqid,int(ele.end)-seqidslis[0]))
                one_seq_eles.append(ele)
        return one_seq_eles
    def check(self,alltext,cid,stchar,endsig):
        c=cid
        if(alltext[c]=='S' or alltext[c]=='A'):
            
            return False
        while(c<len(alltext)):
            if(alltext[c]==' ' or alltext[c] =='\n' or alltext[c] in endsig):
                c+=1
            elif(alltext[c] in stchar):
                return True
            elif(alltext[c] not in stchar):
                return False

    def get_doc_seq_lis(self,oridocdic,docid2docname):
        #
        endsig=['.','!','?']
        stchar=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        
        for k,v in oridocdic.items():
            
            stedids2element,element2stedids,stedele_lis=self.get_stedids_element_relation(v)
            curseqlis=[]
            curseqoriids=[]
            curseqtxt=''
            seq_ori_charids_lis=[]
            seq_ele_lis=[]
            alltext=v.text
            seqid=0
            for cid in range(len(alltext)):
                if((alltext[cid-1] in endsig and alltext[cid]==' ') and self.check(alltext,cid,stchar,endsig)):
                    continue
                if(alltext[cid] in endsig and self.check(alltext,cid+1,stchar,endsig)):
                    
                    curseqtxt+=alltext[cid]
                    seq_ori_charids_lis.append(cid)
                    curseqlis.append(curseqtxt)
                    curseqoriids.append(seq_ori_charids_lis)
                    one_seq_ele=self.get_spatial_seqstedids(seq_ori_charids_lis,stedids2element,stedele_lis,seqid)
                    seq_ele_lis.append(one_seq_ele)
                    seq_ori_charids_lis=[]
                    curseqtxt=''
                    seqid+=1
                    continue
                
                seq_ori_charids_lis.append(cid)
                curseqtxt+=alltext[cid]
            curseqlis.append(curseqtxt)
            curseqoriids.append(seq_ori_charids_lis)
            one_seq_ele=self.get_spatial_seqstedids(seq_ori_charids_lis,stedids2element,stedele_lis,seqid)
            seq_ele_lis.append(one_seq_ele)
            
            v.seqlis=curseqlis
            
            v.seqoriids=curseqoriids

    def preprocess(self, filedir):
        xml_dirs=os.listdir(filedir)
        docid=0
        oridocdic=dict()
        docid2docname=dict()
        for xml_file in xml_dirs:
            if(xml_file[-3:]!='xml'):
                
                continue
            cur_xml_file=os.path.join(filedir,xml_file)
            curdoc=self.parse_data_from_xml(cur_xml_file)
            docid2docname[docid]=cur_xml_file
            oridocdic[docid]=curdoc
            docid+=1
        self.get_doc_seq_lis(oridocdic,docid2docname)
        
        
        
        
        logging.info("--------{} data process DONE!--------".format(filedir))
        return oridocdic,docid2docname
    
    def get_one_doc_gidx_2_seq_bert_tokenidx(self,doc,char2tok_spanlis):
        '''
            gidx:[seqidx,[seqtokidx1,seqtokidx2,...]]
        '''
        gidx_2_bert_tok=dict()
        for idx,wspan in doc.nodeid_2_worispan.items():
            seqids=wspan[0]
            st=wspan[1][0]
            ed=wspan[1][1]
            
            wtoklis=[]
            char2tok_span=char2tok_spanlis[seqids]
            if(st>=len(char2tok_span) or ed >=len(char2tok_span)):
                
                continue
            for i in range(st,ed):
                if(char2tok_span[i][0] not in wtoklis and char2tok_span[i][0]!=-1 and char2tok_span[i][1]!=-1):
                    wtoklis.append(char2tok_span[i][0]) 
            
            gidx_2_bert_tok[idx]=[seqids,wtoklis]
        for idx,elespan in doc.nodeid_2_elespan.items():
            seqids=elespan[0]
            st=elespan[1][0]
            ed=elespan[1][1]
            wtoklis=[]
            char2tok_span=char2tok_spanlis[seqids]
            for i in range(st,ed):
                if(char2tok_span[i][0] not in wtoklis and char2tok_span[i][0]!=-1 and char2tok_span[i][1]!=-1):
                    wtoklis.append(char2tok_span[i][0]) 
            
            gidx_2_bert_tok[idx]=[seqids,wtoklis]
        doc.gidx_2_bert_tok=gidx_2_bert_tok
        nodeid_2_seqidx=[[-1] for _ in range(doc.node_nums)]
        for k ,v in gidx_2_bert_tok.items():
            nodeid_2_seqidx[k]=[v[0]]+v[1]
        
        doc.nodeid_2_seqidx=nodeid_2_seqidx
        
            
    def get_all_doc_gidx_2_seq_bert_tokenidx(self,oridocdic,tokenizer):
        for k,v in oridocdic.items():
            char2tok_spanlis=self.get_char2tok_spanlis_one_doc(v,tokenizer)
            self.get_one_doc_gidx_2_seq_bert_tokenidx(v,char2tok_spanlis)

    def get_sent_2_ele_one_doc(self,doc):
        res=dict()
        n_seq=len(doc.seqlis)
        for i in range(n_seq):
            res[i]=[]
        for elename in config.ele_lis_name:
            for ele in getattr(doc,elename):
                if(self.legal(ele)):
                    res[ele.seqstid[0]].append(ele)
        return res

    def get_graph_one_doc(self,doc):
        print('------------------------get onedoc graph---------------------------')
       

        w_ele_sent_matrix=[[0 for _ in range(doc.node_nums)] for _ in range(doc.node_nums)]
        for seqw in doc.words_span_lis:
            for w1span in seqw:
                for w2span in seqw:
                    idx1=doc.worispan_2_nodeid[w1span]
                    idx2=doc.worispan_2_nodeid[w2span]
                    w_ele_sent_matrix[idx1][idx2]=1
                    w_ele_sent_matrix[idx2][idx1]=1
        
        
        sent_2_ele=self.get_sent_2_ele_one_doc(doc)
        for k,v in sent_2_ele.items():
            for ele1 in v:
                for ele2 in v:
                    idx1=doc.ele_2_nodeid[ele1]
                    idx2=doc.ele_2_nodeid[ele2]
                    w_ele_sent_matrix[idx1][idx2]=1
                    w_ele_sent_matrix[idx2][idx1]=1
        
        doc.w_ele_sent_matrix=w_ele_sent_matrix
        
        w_ele_dep_matrix=[[0 for _ in range(doc.node_nums)] for _ in range(doc.node_nums)]
        nlp = spacy.load('en_core_web_lg')
        neuralcoref.add_to_pipe(nlp)
        for seqidx in range(len(doc.seqlis)):
            curseq=doc.seqlis[seqidx]
            seqpass = nlp(curseq)
            curtokens=list(seqpass)
            curtokenspanlis=[]
            curtokstidx=dict()
            for tok in curtokens:
                curtokstidx[tok]=0
            for tok in curtokens:
                st=curtokstidx[tok]
                wlen=len(tok)
                for i in range(st,len(curseq)):
                    if(curseq[i:i+wlen]==str(tok)):
                        curtokenspanlis.append((i,i+wlen))
                        curtokstidx[tok]=i+wlen
                        break
            for idx,token in enumerate(seqpass):
                
                if(len(token.text.strip())==0):
                    continue
                s,e = token.i, token.head.i
                curitokspan=curtokenspanlis[s]
                curjtokspan=curtokenspanlis[e]
                gidxi=doc.worispan_2_nodeid[(seqidx,curitokspan)]
                gidxj=doc.worispan_2_nodeid[(seqidx,curjtokspan)]
                w_ele_dep_matrix[gidxi][gidxj]=1
                w_ele_dep_matrix[gidxj][gidxi]=1
                
        doc.w_ele_dep_matrix=w_ele_dep_matrix

        


        
        w_ele_conf_matrix=[[0 for _ in range(doc.node_nums)] for _ in range(doc.node_nums)]
        for seqidx in range(len(doc.seqlis)):
            curseq=doc.seqlis[seqidx]
            seqpass = nlp(curseq)
            curtokens=list(seqpass)
            curtokenspanlis=[]
            curtokstidx=dict()
            for tok in curtokens:
                curtokstidx[tok]=0
            for tok in curtokens:
                st=curtokstidx[tok]
                wlen=len(tok)
                for i in range(st,len(curseq)):
                    if(curseq[i:i+wlen]==str(tok)):
                        curtokenspanlis.append((i,i+wlen))
                        curtokstidx[tok]=i+wlen
                        break
            for elem in seqpass._.coref_clusters:
                temp = [[mention.start, mention.end] for mention in elem.mentions]
                for i in range(len(temp)):
                    for j in range(len(temp)):
                        if(i==j):
                            continue
                        for k in range(temp[i][0],temp[i][1]):
                            
                            curinodeid=doc.worispan_2_nodeid[(seqidx,curtokenspanlis[k])]
                            for l in range(temp[j][0],temp[j][1]):
                                curjnodeid= doc.worispan_2_nodeid[(seqidx,curtokenspanlis[l])]
                                w_ele_conf_matrix[curinodeid][curjnodeid]=1
                                w_ele_conf_matrix[curjnodeid][curinodeid]=1
        doc.w_ele_conf_matrix=w_ele_conf_matrix
                    
        

        

    def get_graph_all_doc(self,oridocdic):
        for k,v in oridocdic.items():
            self.get_graph_one_doc(v)
    def get_one_doc_nodeid_dict(self,doc):
        print('------------------------get onedoc nodeid---------------------------')
        worispan_2_nodeid=dict()
        nodeid_2_worispan=dict() 
        ele_2_nodeid=dict()
        nodeid_2_ele=dict()
        elespan_2_nodeid=dict()
        nodeid_2_elespan=dict()
        words_set=set()
        

        seq_words=[]
        words_span_lis=[]
        seqidx=0
        nlp = spacy.load('en_core_web_lg')
        for seq in doc.seqlis:
           
            seqpass = nlp(seq)
            words=[str(tok) for tok in list(seqpass)]
            
            curseq_words=[]
            curseq_words_set=set()
            curseq_model_span=[]
            curseq_words_stidx=dict()
            for w in words:
                if(len(w)>0):
                    words_set.add(w)
                    curseq_words_set.add(w)
                    curseq_words.append(w)
            for tok in curseq_words:
                curseq_words_stidx[tok]=0
            for w in curseq_words_set:
                s=-1
                e=-1
                wlen=len(w)
                st=curseq_words_stidx[w]
                for i in range(st,len(seq)):
                    if(i+wlen>len(seq)):
                        break
                    if(seq[i:i+wlen]==w):
                        s=i
                        e=i+wlen
                        curseq_model_span.append((s,e))
                        curseq_words_stidx[w]=e
                        break
                if(s==-1):
                    print(seq,'*'*10,w)
                    
            curseq_model_span=sorted(curseq_model_span,key=lambda x:x[0])
            curseq_model_span=[ (seqidx,wspan) for wspan in curseq_model_span]

            words_span_lis.append(curseq_model_span)
            seq_words.append(curseq_words)
            seqidx+=1
        
        doc.seq_words=seq_words
        doc.words_span_lis=words_span_lis
        wspan_nodeidx=0
        for seq in words_span_lis:
            
            for wspan in seq:
                worispan_2_nodeid[wspan]=wspan_nodeidx
                nodeid_2_worispan[wspan_nodeidx]=wspan

                wspan_nodeidx+=1
        doc.worispan_2_nodeid=worispan_2_nodeid
        doc.nodeid_2_worispan=nodeid_2_worispan
        
        ele_set=set()
        for elename in config.ele_lis_name:
            for ele in getattr(doc,elename):
                if(self.legal(ele)):
                    ele_set.add(ele)
        w_tot_idx=len(worispan_2_nodeid.keys())
        for idx,ele in enumerate(ele_set):
            ele_2_nodeid[ele]=idx+w_tot_idx
            nodeid_2_ele[idx+w_tot_idx]=ele
            elespan_2_nodeid[(ele.seqstid[0],(ele.seqstid[1],ele.seqedid[1]))]=idx+w_tot_idx
            nodeid_2_elespan[idx+w_tot_idx]=(ele.seqstid[0],(ele.seqstid[1],ele.seqedid[1]))
        doc.node_nums=w_tot_idx+len(ele_set)
        doc.ele_2_nodeid=ele_2_nodeid
        doc.nodeid_2_ele=nodeid_2_ele
        doc.elespan_2_nodeid=elespan_2_nodeid
        doc.nodeid_2_elespan=nodeid_2_elespan
        

    def get_all_nodeid_in_all_doc(self,oridocdic):
        for k,v in oridocdic.items():
            self.get_one_doc_nodeid_dict(v)

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

    def get_one_doc_next_seqs(self,doc,tokenizer):
        t5_seqtokidslis=[]
        all_next_seqs=[]
        doclen=len(doc.seqlis)
        for seq_idx in range(doclen-1):
            t5_seqtokidslis.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc.seqlis[seq_idx])))
            all_next_seqs.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc.seqlis[seq_idx+1])))
        t5_seqtokidslis.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc.seqlis[-1])))
        all_next_seqs.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc.seqlis[0])))
        doc.t5_seqtokidslis=t5_seqtokidslis
        doc.all_next_seqs=all_next_seqs

    def get_all_next_seqs(self,oridocdic,tokenizer):
        for k,v in oridocdic.items():
            self.get_one_doc_next_seqs(v,tokenizer)
    

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
        link_t5_seqtokidslis_good=[]
        link_t5_seqtokidslis_bad=[]
        link_all_next_seqs_good=[]
        link_all_next_seqs_bad=[]
        
        link_ele_node_id_good=[]
        link_ele_node_id_bad=[]


        
        
        
        qslink_tj=[]
        qslink_ld=[]
        qslink_tr=[]
    
        qstok_2_ele=dict()
        for lin in doc.qslink_lis:
            
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
                qslink_tj.append([[seqids],tjtoklis])
                qstok_2_ele[((seqids,)+tuple(tjtoklis))]=tj
                qslink_ld.append([[seqids],ldtoklis])
                qstok_2_ele[((seqids,)+tuple(ldtoklis))]=ld
                qslink_tr.append([[seqids],trtoklis])
                qstok_2_ele[((seqids,)+tuple(trtoklis))]=tr
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                link_t5_seqtokidslis_good.append(doc.t5_seqtokidslis[seqids])
                link_all_next_seqs_good.append(doc.all_next_seqs[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                link_ele_node_id_good.append([doc.ele_2_nodeid[tj],doc.ele_2_nodeid[ld],doc.ele_2_nodeid[tr]])
                all_link_labs_good.append(1)

        ##############################
        
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
                    link_t5_seqtokidslis_bad.append(doc.t5_seqtokidslis[ctr[0][0]])
                    link_all_next_seqs_bad.append(doc.all_next_seqs[ctr[0][0]])
                    link_ele_node_id_bad.append([doc.ele_2_nodeid[qstok_2_ele[(tuple(ctr[0])+tuple(ctj[1]))]],
                                            doc.ele_2_nodeid[qstok_2_ele[(tuple(ctr[0])+tuple(cld[1]))]],
                                            doc.ele_2_nodeid[qstok_2_ele[(tuple(ctr[0])+tuple(ctr[1]))]]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    all_link_labs_bad.append(0)
        ##############################

        olink_tj=[]
        olink_ld=[]
        olink_tr=[]
        #### 
        otok_2_ele=dict()
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
                otok_2_ele[((seqids,)+tuple(tjtoklis))]=tj
                olink_ld.append([[seqids],ldtoklis])
                otok_2_ele[((seqids,)+tuple(ldtoklis))]=ld
                olink_tr.append([[seqids],trtoklis])
                otok_2_ele[((seqids,)+tuple(trtoklis))]=tr
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                link_t5_seqtokidslis_good.append(doc.t5_seqtokidslis[seqids])
                link_all_next_seqs_good.append(doc.all_next_seqs[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                link_ele_node_id_good.append([doc.ele_2_nodeid[tj],doc.ele_2_nodeid[ld],doc.ele_2_nodeid[tr]])
                all_link_labs_good.append(2)

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
                    link_t5_seqtokidslis_bad.append(doc.t5_seqtokidslis[ctr[0][0]])
                    link_all_next_seqs_bad.append(doc.all_next_seqs[ctr[0][0]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    link_ele_node_id_bad.append([doc.ele_2_nodeid[otok_2_ele[(tuple(ctr[0])+tuple(ctj[1]))]],
                                            doc.ele_2_nodeid[otok_2_ele[(tuple(ctr[0])+tuple(cld[1]))]],
                                            doc.ele_2_nodeid[otok_2_ele[(tuple(ctr[0])+tuple(ctr[1]))]]])
                    all_link_labs_bad.append(0)
        ##############################

        movelink_tj=[]
        movelink_ld=[]
        movelink_tr=[]
        move_2_ele=dict()
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
                move_2_ele[((seqids,)+tuple(tjtoklis))]=tj
                movelink_ld.append([[seqids],ldtoklis])
                move_2_ele[((seqids,)+tuple(ldtoklis))]=ld
                movelink_tr.append([[seqids],trtoklis])
                move_2_ele[((seqids,)+tuple(trtoklis))]=tr
                all_link_seqs_good.append(doc.seqtokidslis[seqids])
                link_t5_seqtokidslis_good.append(doc.t5_seqtokidslis[seqids])
                link_all_next_seqs_good.append(doc.all_next_seqs[seqids])
                gold_link.append([[seqids],trtoklis,tjtoklis,ldtoklis])
                link_ele_node_id_good.append([doc.ele_2_nodeid[tj],doc.ele_2_nodeid[ld],doc.ele_2_nodeid[tr]])
                all_link_labs_good.append(3)

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
                    link_t5_seqtokidslis_bad.append(doc.t5_seqtokidslis[ctr[0][0]])
                    link_all_next_seqs_bad.append(doc.all_next_seqs[ctr[0][0]])
                    bad_link.append([ctr[0],ctr[1],ctj[1],cld[1]])
                    link_ele_node_id_bad.append([doc.ele_2_nodeid[move_2_ele[(tuple(ctr[0])+tuple(ctj[1]))]],
                                            doc.ele_2_nodeid[move_2_ele[(tuple(ctr[0])+tuple(cld[1]))]],
                                            doc.ele_2_nodeid[move_2_ele[(tuple(ctr[0])+tuple(ctr[1]))]]])
                    all_link_labs_bad.append(0)
        ##############################


        
        doc.all_link_seqs=all_link_seqs_good+all_link_seqs_bad
        doc.link_t5_seqtokidslis=link_t5_seqtokidslis_good+link_t5_seqtokidslis_bad
        doc.link_all_next_seqs=link_all_next_seqs_good+link_all_next_seqs_bad
        doc.link_ele_node_id=link_ele_node_id_good+link_ele_node_id_bad
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

        # doc.all_link_candidate=gold_link
        doc.all_link_label=all_link_labs_good+all_link_labs_bad

        



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
           
            char2tok_span = [[-1, -1] for _ in range(char_num)] # [-1, -1] is whitespace
            for tok_ind, char_sp in enumerate(token_span):
                for char_ind in range(char_sp[0], char_sp[1]):
                    tok_sp = char2tok_span[char_ind]
                    
                    if tok_sp[0] == -1:
                        tok_sp[0] = tok_ind
                    tok_sp[1] = tok_ind + 1 
            
            res.append(char2tok_span)
        
        return res
            
    

if __name__=='__main__':
    import pickle
    fn1 = './data/data_process_dgl_pkl/ori_data_summary_55_14/train_data_summary.pkl'
    fn2 = './data/data_process_dgl_pkl/ori_data_summary_55_14/train_data_id_summary.pkl'
    fn3 = './data_process_dgl_pkl/ori_data_summary_55_14/vail_data_summary.pkl'
    fn4 = './data_process_dgl_pkl/ori_data_summary_55_14/vail_data_id_summary.pkl'
    fn5 = './data_process_dgl_pkl/ori_data_summary_55_14/test_data_summary.pkl'
    fn6 = './data_process_dgl_pkl/ori_data_summary_55_14/test_data_id_summary.pkl'

    # fn1 = './data_process_dgl_pkl/ori_data_summary_59_16/train_data_summary.pkl'
    # fn2 = './data_process_dgl_pkl/ori_data_summary_59_16/train_data_id_summary.pkl'
    # fn3 = './data_process_dgl_pkl/ori_data_summary_59_16/vail_data_summary.pkl'
    # fn4 = './data_process_dgl_pkl/ori_data_summary_59_16/vail_data_id_summary.pkl'
    # fn5 = './data_process_dgl_pkl/ori_data_summary_59_16/test_data_summary.pkl'
    # fn6 = './data_process_dgl_pkl/ori_data_summary_59_16/test_data_id_summary.pkl'
    # with open(fn1, 'wb') as f:  
    #     picklestring = pickle.dump(processor.train_oridocdic, f)
    # with open(fn2, 'wb') as f:  
    #     picklestring = pickle.dump(processor.train_docid2docname, f)

    # with open(fn3, 'wb') as f:  
    #     picklestring = pickle.dump(processor.vail_oridocdic, f)
    # with open(fn4, 'wb') as f:  
    #     picklestring = pickle.dump(processor.vail_docid2docname, f)

    # with open(fn5, 'wb') as f:  
    #     picklestring = pickle.dump(processor.test_oridocdic, f)
    # with open(fn6, 'wb') as f:  
    #     picklestring = pickle.dump(processor.test_docid2docname, f)
    with open(fn1, 'rb') as f:  
        train_oridocdic = pickle.load(f)
    with open(fn2, 'rb') as f:  
        train_docid2docname = pickle.load(f)

    with open(fn3, 'rb') as f:  
        vail_oridocdic = pickle.load(f)
    with open(fn4, 'rb') as f:  
        vail_docid2docname = pickle.load(f)

    with open(fn5, 'rb') as f:  
        test_oridocdic = pickle.load(f)
    with open(fn6, 'rb') as f:  
        test_docid2docname = pickle.load(f)



