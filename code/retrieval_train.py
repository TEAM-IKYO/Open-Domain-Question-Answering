#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import time
import json
import random
import pickle
import argparse
import numpy as np

from konlpy.tag import Mecab
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer, 
                          AdamW,
                          TrainingArguments,
                          get_linear_schedule_with_warmup,
                          set_seed)

from retrieval_model import Encoder
from retrieval_dataset import TrainRetrievalDataset, ValidRetrievalDataset


# In[2]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def one_step_train(args, batch_list, p_encoder, q_encoder, criterion, scaler):
    p_input_ids = batch_list[0]
    p_attention_mask = batch_list[1]
    p_token_type_ids = batch_list[2]
    q_input_ids = batch_list[3]
    q_attention_mask = batch_list[4]
    q_token_type_ids = batch_list[5]
    targets_batch = batch_list[6]

    batch_loss, batch_acc = 0, 0
    for i in range(args.per_device_train_batch_size):
        batch = (p_input_ids[i],
                 p_attention_mask[i],
                 p_token_type_ids[i],
                 q_input_ids[i],
                 q_attention_mask[i],
                 q_token_type_ids[i])

        targets = torch.tensor([targets_batch[i]]).long()
        batch = tuple(t.to('cuda') for t in batch)
        p_inputs = {'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'token_type_ids': batch[2]}

        q_inputs = {'input_ids' : batch[3],
                    'attention_mask' : batch[4],
                    'token_type_ids': batch[5]}

        p_outputs = p_encoder(**p_inputs)     # (20, E)
        q_outputs = q_encoder(**q_inputs)     # (1, E)

        # Calculate similarity score & loss
        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (1, E) x (E, N) = (1, 20)
        # target : position of positive samples = diagonal element
        if torch.cuda.is_available():
            targets = targets.to('cuda')
        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, 1)

        loss = criterion(sim_scores, targets)
        scaler.scale(loss).backward()

        batch_loss += loss.cpu().item()
        batch_acc += torch.sum(preds.cpu() == targets.cpu())
    return p_encoder, q_encoder, batch_loss, batch_acc

def training(args, epoch, train_dataloader, p_encoder, q_encoder, criterion, scaler, optimizer, scheduler, logger):
    ## train
    epoch_iterator = tqdm(train_dataloader, desc="train Iteration")
    p_encoder.to('cuda').train()
    q_encoder.to('cuda').train()

    running_loss, running_acc, num_cnt = 0, 0, 0
    with torch.set_grad_enabled(True):
        for step, batch_list in enumerate(epoch_iterator):
            p_encoder, q_encoder, batch_loss, batch_acc = one_step_train(args,
                                                                         batch_list,
                                                                         p_encoder,
                                                                         q_encoder,
                                                                         criterion,
                                                                         scaler)
            running_loss += batch_loss/args.per_device_train_batch_size
            running_acc += batch_acc/args.per_device_train_batch_size
            num_cnt += 1
            
            if (step+1) % args.gradient_accumulation_steps == 0:
                log_step = epoch*len(epoch_iterator) + step
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                p_encoder.zero_grad()
                q_encoder.zero_grad()
                
                logger.add_scalar(f"Train/loss", batch_loss/args.per_device_train_batch_size, log_step)
                logger.add_scalar(f"Train/accuracy", batch_acc/args.per_device_train_batch_size*100, log_step)
                
    epoch_loss = float(running_loss / num_cnt)
    epoch_acc  = float((running_acc.double() / num_cnt).cpu()*100)
    print(f'global step-{log_step} | Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}')
    return p_encoder, q_encoder, scaler, optimizer, scheduler

def validation(args, epoch, valid_dataloader, p_encoder, q_encoder, logger, best_acc, run_name):
    ## valid
    epoch_iterator = tqdm(valid_dataloader, desc="valid Iteration")
    p_encoder.to('cuda').eval()
    q_encoder.to('cuda').eval()

    running_loss, running_acc, num_cnt = 0, 0, 0
    for step, batch in enumerate(epoch_iterator):
        with torch.set_grad_enabled(False):
            batch = tuple(t.squeeze(0) if i < 6 else t for i, t in enumerate(batch))

            targets, top_k_id = batch[-2], batch[-1]
            if torch.cuda.is_available():
                batch = tuple(t.to('cuda') for t in batch[:-2])

            p_inputs = {'input_ids' : batch[0],
                        'attention_mask' : batch[1],
                        'token_type_ids': batch[2]}

            q_inputs = {'input_ids' : batch[3],
                        'attention_mask' : batch[4],
                        'token_type_ids': batch[5]}
                
            p_outputs = p_encoder(**p_inputs)     # (N, E)
            q_outputs = q_encoder(**q_inputs)     # (1, E)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (1, E) x (E, N) = (1, N)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            
            class_0 = torch.Tensor([1 if i.item() == 0 else 0 for idx, i in enumerate(top_k_id)])
            w = (torch.sum(sim_scores, dim=1)*1/sim_scores.size()[1]).item()
            sim_scores -= w*class_0.unsqueeze(0).cuda()
            
            _, preds = torch.max(sim_scores, 1)
            if preds.item() in targets:
                running_acc += 1
            num_cnt += 1
            
    epoch_acc  = float((running_acc / num_cnt)*100)
    logger.add_scalar(f"Val/accuracy", epoch_acc, epoch)
    print(f'Epoch-{epoch} | Accuracy: {epoch_acc:.2f}')

    if epoch_acc > best_acc:
        best_idx = epoch
        best_acc = epoch_acc
        
        save_path = os.path.join(args.output_dir, 'model')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(p_encoder.cpu().state_dict(), os.path.join(save_path, f'p_{run_name}.pt'))
        torch.save(q_encoder.cpu().state_dict(), os.path.join(save_path, f'q_{run_name}.pt'))
        print(f'\t==> best model saved - {best_idx} / Accuracy: {best_acc:.2f}')
    return best_acc

def train(args, p_encoder, q_encoder, train_dataloader, valid_dataloader, criterion, scaler, optimizer, scheduler, logger, run_name):
    # Start training!
    best_acc = 0.0

    train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
    for epoch in train_iterator:
        optimizer.zero_grad()
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()
        
        p_encoder, q_encoder, scaler, optimizer, scheduler = training(args, epoch, train_dataloader, p_encoder, q_encoder, criterion, scaler, optimizer, scheduler, logger)
        best_acc = validation(args, epoch, valid_dataloader, p_encoder, q_encoder, logger, best_acc, run_name)
    return p_encoder, q_encoder

def main(args):
    seed_everything(seed=args.seed)

    p_tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    p_tokenizer.model_max_length = 1536
    q_tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    training_dataset = get_pickle(f"../data/retrieval_dataset/Top{args.top_k}_preprocess_train.pkl")
    validation_dataset = get_pickle(f"../data/retrieval_dataset/Top{args.top_k}_preprocess_valid.pkl")

    train_dataset = TrainRetrievalDataset(training_dataset, p_tokenizer, q_tokenizer)
    valid_dataset = ValidRetrievalDataset(validation_dataset, p_tokenizer, q_tokenizer)

    p_encoder = Encoder(args.model_checkpoint)
    q_encoder = Encoder(args.model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')
        print('GPU enabled')

    training_args = TrainingArguments(output_dir=args.output_dir,
                                      evaluation_strategy='epoch',
                                      learning_rate=args.learning_rate,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=1,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      num_train_epochs=args.epoch,
                                      weight_decay=0.01)

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=training_args.per_device_train_batch_size)

    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset,
                                  sampler=valid_sampler,
                                  batch_size=training_args.per_device_eval_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
                                    {'params': [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                                    {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
                                    {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                                    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.NLLLoss()

    # -- logging
    log_dir = os.path.join(training_args.output_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        raise NameError(f'Already Exists Directory >>> [ Path : {log_dir} ]')
    logger = SummaryWriter(log_dir=log_dir)

    p_encoder, q_encoder = train(training_args, p_encoder, q_encoder, train_dataloader, valid_dataloader, criterion, scaler, optimizer, scheduler, logger, args.run_name)
    print('complete !!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_dir', type=str, default='../retrieval_output/')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--run_name', type=str, default='best_dense_retrieval')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    
    print(f'Output Dir ::: {args.output_dir}')
    print(f'Model Checkpoint ::: {args.model_checkpoint}')
    print(f'Seed ::: {args.seed}')
    print(f'Epoch ::: {args.epoch}')
    print(f'Learning rate ::: {args.learning_rate}')
    print(f'Gradient Accumulation Steps ::: {args.gradient_accumulation_steps}')
    print(f'Dataset K Number ::: {args.top_k}')
    print(f'Run Name ::: {args.run_name}')
    
    main(args)