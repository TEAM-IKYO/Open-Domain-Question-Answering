import argparse
import time
import warnings

import pickle
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, ElectraForSequenceClassification, AdamW
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data_set import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_pickle(pickle_path):
    '''Custom Dataset을 Load하기 위한 함수'''
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()

    return dataset

def get_data():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    ai_hub = get_pickle("../../data/ai_hub_dataset.pkl")
    train_token, train_label = tokenized_dataset(ai_hub["train"], tokenizer)
    val_token, val_label = tokenized_dataset(ai_hub["validation"], tokenizer)
    
    train_set = RE_Dataset(train_token, train_label)
    val_set = RE_Dataset(val_token, val_label)
    
    train_iter = DataLoader(train_set, batch_size=16, shuffle=True)
    val_iter = DataLoader(val_set, batch_size=16, shuffle=True)

    return train_iter, val_iter

def get_model():
    network = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=6, hidden_dropout_prob=0.0).to("cuda:0")
    optimizer = AdamW(network.parameters(), lr=5e-6)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss().to("cuda:0")

    return network, optimizer, scaler, scheduler, criterion

def training_per_step(model, loss_fn, optimizer, scaler, input_ids, attention_mask, labels, device):
    '''매 step마다 학습을 하는 함수'''
    model.train()
    with autocast():
        labels = labels.to(device)
 
        preds = model(input_ids.to(device), attention_mask = attention_mask.to(device))[0]
        loss = loss_fn(preds, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss

def validating_per_steps(epoch, model, loss_fn, test_loader, device):
    '''특정 step마다 검증을 하는 함수'''
    model.eval()

    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []

    pbar = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    for input_ids, attention_mask, labels in pbar :
        labels = labels.to(device)

        preds = model(input_ids.to(device), attention_mask = attention_mask.to(device))[0]
        
        preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
        targets_all += [labels.detach().cpu().numpy()]

        loss = loss_fn(preds, labels)

        loss_sum += loss.item()*labels.shape[0]
        sample_num += labels.shape[0]

        description = f"epoch {epoch + 1} loss: {loss_sum/sample_num:.4f}"
        pbar.set_description(description)
    
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()

    print("     test accuracy = {:.4f}".format(accuracy))

    return accuracy

def train(model, loss_fn, optimizer, scaler, train_loader, test_loader, scheduler, device):
    '''training과 validating을 진행하는 함수'''
    prev_acc = 0
    global_steps = 0
    for epoch in range(1):
        running_loss = 0
        sample_num = 0
        preds_all = []
        targets_all = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for step, (input_ids, attention_mask, labels) in pbar:
            # training phase
            loss = training_per_step(model, loss_fn, optimizer, scaler, input_ids, attention_mask, labels, device)
            running_loss += loss.item()*labels.shape[0]
            sample_num += labels.shape[0]
            
            global_steps += 1
            description = f"{epoch+1}epoch {global_steps: >4d}step | loss: {running_loss/sample_num: .4f} "
            pbar.set_description(description)

            # validating phase
            if global_steps % 500 == 0 :
                with torch.no_grad():
                    acc = validating_per_steps(epoch, model, loss_fn, test_loader, device)
                if acc > prev_acc:
                    torch.save(model, "../../output/question_model.pt")
                    prev_acc = acc

                if scheduler is not None :
                    scheduler.step() 

def main():
    seed_everything(2021)
    train_iter, val_iter = get_data()
    network, optimizer, scaler, scheduler, criterion = get_model()
    train(network, criterion, optimizer, scaler, train_iter, val_iter, scheduler, "cuda:0")

if __name__ == "__main__":
    main()
