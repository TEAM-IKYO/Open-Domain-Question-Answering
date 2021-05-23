#!/usr/bin/env python
# coding: utf-8

import kss
import torch
import random
from tqdm.notebook import tqdm

class TrainRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer
        
    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top_context = self.dataset['top_k'][idx]
        target = self.dataset['answer_idx'][idx]
        
        p_seqs = self.p_tokenizer(top_context,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        
        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']
        
        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']
        
        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])
        for i in range(len(p_attention_mask)):
            str_idx, end_idx = self._select_range(p_attention_mask[i])

            p_input_ids_tmp = torch.cat([torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
            p_attention_mask_tmp = p_attention_mask[i][str_idx-1:end_idx+1].int().long()
            p_token_type_ids_tmp = p_token_type_ids[i][str_idx-1:end_idx+1].int().long()
            
            p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
            p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)]).int().long()
            p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)]).int().long()
            
        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target
            
    def __len__(self):
        return len(self.dataset['question'])

    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return 1, 511
        else:
            start_idx = random.randint(1, sent_len-511)
            end_idx = start_idx + 510
            return start_idx, end_idx
    
class ValidRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer
        
    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top_context = self.dataset['top_k'][idx]
        target = self.dataset['answer_idx'][idx]
        
        p_seqs = self.p_tokenizer(top_context,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        
        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']
        
        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']
        
        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])

        top_k_id = []
        for i in range(len(p_attention_mask)):
            ids_list = self._select_range(p_attention_mask[i])
            if i == target:
                target = list(range(len(p_input_ids_list), len(p_input_ids_list)+len(ids_list)))
            for str_idx, end_idx in ids_list:
                p_input_ids_tmp = torch.cat([torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
                p_attention_mask_tmp = p_attention_mask[i][str_idx-1:end_idx+1].int().long()
                p_token_type_ids_tmp = p_token_type_ids[i][str_idx-1:end_idx+1].int().long()

                p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
                p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)]).int().long()
                p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)]).int().long()
                top_k_id.append(i)

        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target, top_k_id

    def __len__(self):
        return len(self.dataset['question'])
    
    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return [(1,511)]
        else:
            num = sent_len // 255
            res = sent_len % 255
            if res == 0:
                num -= 1
            ids_list = []
            for n in range(num):
                if res > 0 and n == num-1:
                    end_idx = sent_len-1
                    start_idx = end_idx - 510
                else:
                    start_idx = n*255+1
                    end_idx = start_idx + 510
                ids_list.append((start_idx, end_idx))
            return ids_list