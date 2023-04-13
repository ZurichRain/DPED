import sys
import os
sys.path.append('./code/')
import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config_script.config as config
import numpy as np
from model_script.Bert_base_nxt_seq import MyBertBaseJoint_Model
from transformers import BertTokenizer,BertTokenizerFast
import torch


def train_epoch(train_loader, model, optimizer, epoch , eval_fun ,scheduler=None,train_writer=None):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    prey1_lis=[]
    truy1_lis=[]
    
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        cbatch=batch_samples
        loss,prey1 = model(**cbatch)
        prey1_lis += torch.argmax(prey1,dim=-1).to('cpu').numpy().tolist()

        truy1_lis += cbatch['y_labels'].view(-1).to('cpu').numpy().tolist()
        train_losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        torch.cuda.empty_cache()

    train_epoch_f1=eval_fun(truy1_lis,prey1_lis)
    logging.info("Epoch: {}, train_epoch_f1: {}".format(epoch,train_epoch_f1))
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    if train_writer is not None:
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('F1', train_epoch_f1, epoch)

def train( model:torch.nn.Module, optimizer,train_loader, eval_fun,dev_loader=None, scheduler=None, model_dir=None,train_writer=None,test_writer=None):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model=MyBertBaseJoint_Model.from_pretrain(config.save_train_model_file)
       
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for epoch in range(1, config.epoch_num + 1):
        # with torch.autograd.profiler(use_cuda=True) as prof:
        train_epoch(train_loader, model, optimizer, epoch ,eval_fun , scheduler,train_writer)
        if dev_loader is None:
            pass
        else:
            val_metrics = evaluate(dev_loader, model,epoch,eval_fun,test_writer=test_writer)
            val_f1 = val_metrics['f1']
            logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
            improve_f1 = val_f1 - best_val_f1
            
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                if(os.path.exists(config.save_train_model_dir)):
                    torch.save(model,config.save_train_model_file)
                else:
                    os.makedirs(config.save_train_model_dir)
                    torch.save(model,config.save_train_model_file)
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience: 
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_val_f1))
                break
    logging.info("Training Finished!")

def evaluate(dev_loader, model, epoch,eval_fun,mode='dev',test_writer=None):
    # set model to evaluation mode
    model.eval()

   
    dev_losses = 0

    with torch.no_grad():
        prey1_lis=[]
        truy1_lis=[]
        
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            
            cbatch=batch_samples
            
            loss,prey1=model(**cbatch)
            
            prey1_lab=torch.argmax(prey1,dim=-1)
            
            prey1_lis += prey1_lab.to('cpu').numpy().tolist()
            
            truy1_lis += cbatch['y_labels'].view(-1).to('cpu').numpy().tolist()
            dev_losses += loss.item()
            

    
    metrics = {}
    metrics['f1']=eval_fun(truy1_lis,prey1_lis)
    
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    if(test_writer is not None):
        test_writer.add_scalar('F1', metrics['f1'], epoch)
        test_writer.add_scalar('loss', metrics['loss'], epoch)
    return metrics
