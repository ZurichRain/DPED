
# import utils
import sys
import os

sys.path.append('./code/')
import config_script.config as config
import logging
import pickle
import numpy as np
from data_process_script.data_process import Processor

from data_loader_script.data_loader_qs_o_movelink import JointThreeLinkDataset
from data_loader_script.data_loader_qs_o_move_6_labels import JointThreeLink6LabelsDataset
from data_loader_script.data_loader_for_longformer_joint import JointThreeLinkLongformerDataset
from data_loader_script.data_loader_for_longformer_joint_correct import JointThreeLinkLongformerCorrectDataset
from data_loader_script.bert_joint_threelink_prompt import JointThreeLinkPromptDataset
from data_loader_script.bert_joint_threelink_prompt_has_relation import JointThreeLinkPromptHasRelationDataset
from data_loader_script.bert_joint_threelink_after_process_prompt import JointThreeLinkAftProcessPromptDataset
from data_loader_script.bert_joint_three_link_aft_process import JointThreeLinkAftProcessDataset
from data_loader_script.bert_confidence_prompt import JointThreeLinkConfidencePromptDataset
from data_loader_script.bert_confidence_badsample_prompt import JointThreeLinkConfidenceBadSamplePromptDataset
from data_loader_script.bert_mutiviewer_prompt_muti_dist import JointThreeLinkMutiviewerPromptMutiDistDataset


from model_script.Bert_base_joint import MyBertBaseJoint_Model
from model_script.Bert_base_joint_muti_dist import BertBaseMutiDistJoint_Model
from model_script.Bert_base_joint_muti_dist_6_labels import BertBaseMutiDist6LabelsJoint_Model
from model_script.Longformer_base_joint_muti_dist import LongformerBaseMutiDistJoint_Model
from model_script.Longformer_base_joint_muti_dist_correct import LongformerBaseMutiDistJointCorrect_Model
from model_script.Bert_joint_threelink_prompt import BertBaseJointPrompt_Model
from model_script.Bert_joint_threelink_muti_dist_prompt import BertBaseJointPromptMutiDist_Model
from model_script.Bert_joint_threelink_muti_dist_prompt_has_link import BertBaseJointPromptMutiDistHasLink_Model
from model_script.Bert_joint_threelink_muti_dist_prompt_aft_process import BertBaseJointPromptMutiDistAftProcess_Model
from model_script.Bert_base_joint_muti_dist_aft_process import BertBaseMutiDistJointAftProcess_Model
from model_script.Bert_prompt_confidence import BertBaseJointPromptConf_Model
from model_script.Bert_mutiviewer_prompt_muti_dist import BertBaseJointMutiviewerPromptMutiDist_Model


from train_script import train, evaluate
from train_script_for_MLM import train_MLM

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW 
import torch.optim as optim
from util_script.optimizer_wf import RAdam
import sys
import warnings
from torch.utils.tensorboard import SummaryWriter
from vail_test_data_script.bert_base_joint import eval_link
from vail_test_data_script.bert_base_joint_aft_process import eval_link_aft_process
from vail_test_data_script.bert_promt_confidence import prompt_confidence_eval_link
import shutil
import random
import torch
from util_script.metrics import f1_score_1,f1_score_sr,f1_score_3,f1_score_6,f1_score_377

warnings.filterwarnings('ignore')

'''
MODEL_CLASSES 用来确认模型类别
MODEL_DATASET 用来确定使用的dataset
MODEL_EVAL 用来确定模型对应的评估函数
'''
MODEL_CLASSES={
    'bert_base_joint_three_link_model' :  MyBertBaseJoint_Model,
    'bert_base_joint_muti_dist_joint_three_link_model':BertBaseMutiDistJoint_Model,
    'bert_base_joint_muti_dist_6_labels_joint_three_link_model': BertBaseMutiDist6LabelsJoint_Model,
    'longformer_base_joint_muti_dist_joint_three_link_model': LongformerBaseMutiDistJoint_Model,
    'longformer_base_joint_muti_dist_correct_joint_three_link_model': LongformerBaseMutiDistJointCorrect_Model,
    'bert_prompt_joint_three_link_model': BertBaseJointPrompt_Model,
    'bert_prompt_muti_dist_joint_three_link_model': BertBaseJointPromptMutiDist_Model,
    'bert_prompt_muti_dist_has_link_joint_three_link_model': BertBaseJointPromptMutiDistHasLink_Model,
    'bert_prompt_muti_dist_aft_process_joint_three_link_model': BertBaseJointPromptMutiDistAftProcess_Model,
    'bert_base_muti_dist_aft_process_joint_three_link_model': BertBaseMutiDistJointAftProcess_Model,
    'bert_prompt_confidece_aft_process': BertBaseJointPromptConf_Model,
    'bert_prompt_confidece_badsample_aft_process': BertBaseJointPromptConf_Model,
    'bert_mutiviewer_prompt_confidece_badsample_aft_process': BertBaseJointMutiviewerPromptMutiDist_Model,
}
MODEL_DATASET={
    'bert_base_joint_three_link_model' :  JointThreeLinkDataset,
    'bert_base_joint_muti_dist_joint_three_link_model':JointThreeLinkDataset,
    'bert_base_joint_muti_dist_6_labels_joint_three_link_model': JointThreeLink6LabelsDataset,
    'longformer_base_joint_muti_dist_joint_three_link_model': JointThreeLinkLongformerDataset,
    'longformer_base_joint_muti_dist_correct_joint_three_link_model': JointThreeLinkLongformerCorrectDataset,
    'bert_prompt_joint_three_link_model': JointThreeLinkPromptDataset,
    'bert_prompt_muti_dist_joint_three_link_model': JointThreeLinkPromptDataset,
    'bert_prompt_muti_dist_has_link_joint_three_link_model': JointThreeLinkPromptHasRelationDataset,
    'bert_prompt_muti_dist_aft_process_joint_three_link_model': JointThreeLinkAftProcessPromptDataset,
    'bert_base_muti_dist_aft_process_joint_three_link_model': JointThreeLinkAftProcessDataset,
    'bert_prompt_confidece_aft_process': JointThreeLinkConfidencePromptDataset,
    'bert_prompt_confidece_badsample_aft_process': JointThreeLinkConfidenceBadSamplePromptDataset,
    'bert_mutiviewer_prompt_confidece_badsample_aft_process': JointThreeLinkMutiviewerPromptMutiDistDataset,
    
}
MODEL_EVAL={
    'bert_base_joint_three_link_model' :  eval_link,
    'bert_base_joint_muti_dist_joint_three_link_model':eval_link,
    'bert_base_joint_muti_dist_6_labels_joint_three_link_model': eval_link,
    'longformer_base_joint_muti_dist_joint_three_link_model': eval_link,
    'longformer_base_joint_muti_dist_correct_joint_three_link_model': eval_link,
    'bert_prompt_joint_three_link_model': eval_link,
    'bert_prompt_muti_dist_joint_three_link_model': eval_link,
    'bert_prompt_muti_dist_has_link_joint_three_link_model': eval_link,
    'bert_prompt_muti_dist_aft_process_joint_three_link_model': eval_link_aft_process,
    'bert_base_muti_dist_aft_process_joint_three_link_model': eval_link_aft_process,
    'bert_prompt_confidece_aft_process': prompt_confidence_eval_link,
    'bert_prompt_confidece_badsample_aft_process': prompt_confidence_eval_link,
    'bert_mutiviewer_prompt_confidece_badsample_aft_process': eval_link,
}
def seed_everything(seed=1226):
    '''
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True 

def get_three_data(train_doc_num,test_doc_num):
    if(train_doc_num==55 and test_doc_num==14):
        
        fn1 = './data_process_pkl/ori_data_summary_55_14/train_data_summary.pkl'
        fn2 = './data_process_pkl/ori_data_summary_55_14/train_data_id_summary.pkl'
        fn3 = './data_process_pkl/ori_data_summary_55_14/vail_data_summary.pkl'
        fn4 = './data_process_pkl/ori_data_summary_55_14/vail_data_id_summary.pkl'
        fn5 = './data_process_pkl/ori_data_summary_55_14/test_data_summary.pkl'
        fn6 = './data_process_pkl/ori_data_summary_55_14/test_data_id_summary.pkl'
    elif(train_doc_num==59 and test_doc_num==16):
        fn1 = './data_process_pkl/ori_data_summary_59_16/train_data_summary.pkl'
        fn2 = './data_process_pkl/ori_data_summary_59_16/train_data_id_summary.pkl'
        fn3 = './data_process_pkl/ori_data_summary_59_16/vail_data_summary.pkl'
        fn4 = './data_process_pkl/ori_data_summary_59_16/vail_data_id_summary.pkl'
        fn5 = './data_process_pkl/ori_data_summary_59_16/test_data_summary.pkl'
        fn6 = './data_process_pkl/ori_data_summary_59_16/test_data_id_summary.pkl'
    else:
        raise Exception("没有该指定的数据集！\n 请检查是否是如下之一：\n1、训练集55测试集14\n2、训练集59测试集16")
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
    return train_oridocdic,train_docid2docname,vail_oridocdic,vail_docid2docname,test_oridocdic,test_docid2docname

def run():
    """train the model"""
    seed_everything()
    logging.info("device: {}".format(config.device))
    CurDataset = MODEL_DATASET[config.model_type]
    model:torch.nn.Module = MODEL_CLASSES[config.model_type]()
    CurEvallink = MODEL_EVAL[config.model_type]
    # processor = Processor(config)
    
    # processor.process()
    train_oridocdic,train_docid2docname,vail_oridocdic,vail_docid2docname,test_oridocdic,test_docid2docname=get_three_data(55,14)
    logging.info("--------Process Done!--------")


    train_dataset = CurDataset(train_oridocdic,train_docid2docname,config, 'train')
    dev_dataset = CurDataset(vail_oridocdic,vail_docid2docname, config)
    test_dataset = CurDataset(test_oridocdic,test_docid2docname, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    #  * (config.epoch_num // 10)
    # Train the model
    logging.info("--------Start Training!--------")
    if(not os.path.exists(config.train_log_dir)):
        os.makedirs(config.train_log_dir)
    else :
        shutil.rmtree(config.train_log_dir)
    if(not os.path.exists(config.test_log_dir)):
        os.makedirs(config.test_log_dir)
    else:
        shutil.rmtree(config.test_log_dir)
    train_writer = SummaryWriter(log_dir=config.train_log_dir)
    test_writer = SummaryWriter(log_dir=config.test_log_dir)

    if config.do_train:

        train(model, optimizer, train_loader,eval_fun=f1_score_3,dev_loader=dev_loader,\
            scheduler=scheduler,train_writer=train_writer,test_writer=test_writer)
        

    if config.do_test:
       
        CurEvallink(test_loader,eval_fun=f1_score_3)

if __name__ == '__main__':
    run()
