import sys
import os
sys.path.append('./code/')
import torch
import config_script.config as config
# from util_script.metrics import f1_score_3
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer,BertTokenizerFast
import json
def eval_link(test_loader,eval_fun):
    model = torch.load(config.save_train_model_file)
    model.eval()
    dev_losses = 0

    tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
    idx2label = ['nolink','qslink','olink','movelink']
    save_res = dict() 
    def init_res(save_res):
        save_res['seq'] = []
        save_res['e1'] = []
        save_res['tr'] = []
        save_res['e2'] = []
        save_res['gold_label'] = []
        save_res['pre_label'] = []
    init_res(save_res) 
    with torch.no_grad():
        prey1_lis=[]
        truy1_lis=[]
        for idx, batch_samples in enumerate(test_loader):
            cbatch=batch_samples
            
            prey1_lab=torch.argmax(prey1,dim=-1)
            
            prey1_lis += prey1_lab.to('cpu').numpy().tolist()
            
            truy1_lis += cbatch['y_labels'].view(-1).to('cpu').numpy().tolist()

            for idx,seq in enumerate(cbatch['datas']):
                save_res['seq'].append(tokenizer.decode(seq,skip_special_tokens=True))
                save_res['e1'].append(tokenizer.decode(seq[cbatch['e1_mask'][idx][0]:cbatch['e1_mask'][idx][1]+1]))
                save_res['tr'].append(tokenizer.decode(seq[cbatch['tr_mask'][idx][0]:cbatch['tr_mask'][idx][1]+1]))
                save_res['e2'].append(tokenizer.decode(seq[cbatch['e2_mask'][idx][0]:cbatch['e2_mask'][idx][1]+1]))
                save_res['gold_label'].append(idx2label[cbatch['y_labels'][idx].item()])
            save_res['pre_label'] += [idx2label[labidx] for labidx in prey1_lab.to('cpu').numpy().tolist()]
            dev_losses += loss.item()


    metrics = {}
    metrics['f1']=eval_fun(truy1_lis,prey1_lis)
    class_report= classification_report(truy1_lis, prey1_lis,
                                                target_names=['no-link','qslink','olink','movelink'])
    id2label = ['no-link','qslink','olink','movelink']
    truy1_lis = [id2label[i] for i in truy1_lis]
    prey1_lis = [id2label[i] for i in prey1_lis]
    confusion_mat = confusion_matrix(truy1_lis, prey1_lis,labels = ['no-link','qslink','olink','movelink'])
    print(class_report)
    print(confusion_mat)
    metrics['loss'] = float(dev_losses) / len(test_loader)

    print('testf1: ',metrics['f1'])
    print('loss: ',metrics['loss'])

    if(os.path.exists(config.save_train_result_dir)):
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n'+str(class_report))
    else:
        os.makedirs(config.save_train_result_dir)
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n'+str(class_report))

    if(os.path.exists(config.save_test_all_res_dir)):
        with open (config.save_res_file,'w')as f:
            json.dump(save_res,f)
    else:
        os.makedirs(config.save_test_all_res_dir)
        with open (config.save_res_file,'w')as f:
            json.dump(save_res,f)

    with open (config.save_res_file,'r')as f:
        a = json.load(f)
    assert len(a['seq']) == len(a['gold_label']) and len(a['seq']) == len(a['pre_label'])
    
    
    