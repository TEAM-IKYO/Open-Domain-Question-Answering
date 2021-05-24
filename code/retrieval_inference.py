#!/usr/bin/env python
# coding: utf-8

import logging
import os
import sys
import time
import json

import torch
import random
import numpy as np
import pandas as pd
import os
import pickle

from scipy.special import log_softmax

from tqdm import tqdm
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from konlpy.tag import Mecab
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from elasticsearch import Elasticsearch
from subprocess import Popen, PIPE, STDOUT

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from retrieval_model import Encoder

# 엘라스틱 서치 노트북 파일 (es_retrieval.ipynb 를 먼저 실행하여 index 등록후 사용해야합니다. )
def elastic_setting(index_name):
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])
    return es


def search_es(es_obj, index_name, question_text, n_results):
    # search query
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    # n_result => 상위 몇개를 선택?
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res

def elastic_retrieval(es, index_name, question_text, n_results):
    res = search_es(es, index_name, question_text, n_results)
    # 매칭된 context만 list형태로 만든다.
    context_list = list((hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits'])
    return context_list


def get_pickle(pickle_path):
    '''Custom Dataset을 Load하기 위한 함수'''
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset


def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()
    return None


def select_range(attention_mask):
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

def inference(args, p_encoder, q_encoder, question_texts, p_tokenizer, q_tokenizer):
    es = elastic_setting(args.index_name)
    
    p_encoder.eval()
    q_encoder.eval()

    dense_retrieval_result = {}
    for question_text in tqdm(question_texts):
        es_context_list = elastic_retrieval(es, args.index_name, question_text, args.es_top_k = 70)
        es_context_list = [context for context, score in es_context_list]

        p_seqs = p_tokenizer(es_context_list,
                              padding='max_length',
                              truncation=True,
                              return_tensors='pt')

        q_seqs = q_tokenizer(question_text,
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
            ids_list = select_range(p_attention_mask[i])
            for str_idx, end_idx in ids_list:
                p_input_ids_tmp = torch.cat([torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
                p_attention_mask_tmp = p_attention_mask[i][str_idx-1:end_idx+1].int().long()
                p_token_type_ids_tmp = p_token_type_ids[i][str_idx-1:end_idx+1].int().long()

                p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
                p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)]).int().long()
                p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)]).int().long()
                top_k_id.append(i)

        batch_num = 20
        if len(p_input_ids_list) % batch_num == 0:
            num = len(p_input_ids_list) // batch_num
        else:
            num = len(p_input_ids_list) // batch_num + 1

        p_output_list = []
        for i in range(num):
            p_input_ids = p_input_ids_list[i*batch_num:(i+1)*batch_num]
            p_attention_mask = p_attention_mask_list[i*batch_num:(i+1)*batch_num]
            p_token_type_ids =p_token_type_ids_list[i*batch_num:(i+1)*batch_num]

            batch = (p_input_ids, p_attention_mask, p_token_type_ids)        
            p_inputs = {'input_ids' : batch[0].to('cuda'),
                        'attention_mask' : batch[1].to('cuda'),
                        'token_type_ids': batch[2].to('cuda')}
            p_outputs = p_encoder(**p_inputs).cpu()
            p_output_list.extend(p_outputs.cpu().tolist())
        p_output_list = np.array(p_output_list)

        batch = (q_input_ids, q_attention_mask, q_token_type_ids)
        q_inputs = {'input_ids' : batch[0].to('cuda'),
                    'attention_mask' : batch[1].to('cuda'),
                    'token_type_ids': batch[2].to('cuda')}
        q_outputs = q_encoder(**q_inputs).cpu() # (N, E)
        q_outputs = np.array(q_outputs.cpu().tolist())

        sim_scores = np.matmul(q_outputs, np.transpose(p_output_list, [1, 0])) # (1, E) x (E, N) = (1, N)
        sim_scores = log_softmax(sim_scores, axis=1)

        class_0 = np.array([1 if i == 0 else 0 for idx, i in enumerate(top_k_id)])
        w = np.sum(sim_scores, axis=1) * 1/np.shape(sim_scores)[1]
        sim_scores = sim_scores[0] - w[0]*class_0

        preds_idx = np.argsort(-1*sim_scores, axis=0)

        top_idx_list = []
        top_k_list = []
        for idx in preds_idx:
            top_idx = top_k_id[idx]
            if top_idx in top_idx_list:
                continue
            top_idx_list.append(top_idx)
            top_k_list.append((es_context_list[top_idx], sim_scores[idx]))
        dense_retrieval_result[question_text] = top_k_list[:args.dr_top_k]
    return dense_retrieval_result


def main(args):
    
    text_data = load_from_disk('../../data/test_dataset')
    question_texts = text_data["validation"]["question"]

    p_tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    p_tokenizer.model_max_length = 1536
    q_tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    p_encoder = Encoder(args.model_checkpoint)
    q_encoder = Encoder(args.model_checkpoint)

    p_encoder.load_state_dict(torch.load(f'../retrieval_output/{args.run_name}/model/p_{args.run_name}.pt'))
    q_encoder.load_state_dict(torch.load(f'../retrieval_output/{args.run_name}/model/q_{args.run_name}.pt'))

    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')
        print('GPU enabled')
        
    dense_retrieval_result = inference(args, p_encoder, q_encoder, question_texts, p_tokenizer, q_tokenizer)

    save_path = f'../data/test_ex{args.es_top_k}_dr{args.dr_top_k}_dense.pkl'
    save_pickle(save_path, dense_retrieval_result)
    print('complete !!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--run_name', type=str, default='best_dense_retrieval')
    parser.add_argument('--es_top_k', type=int, default=70)
    parser.add_argument('--dr_top_k', type=int, default=70)
    parser.add_argument('--index_name', type=str, default="nori-index")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    
    print(f'Model Checkpoint ::: {args.model_checkpoint}')
    print(f'Run Name ::: {args.run_name}')
    print(f'Top k Number of Elastic Retrieval ::: {args.es_top_k}')
    print(f'Top k Number of Dense Retrieval ::: {args.dr_top_k}')
    print(f'Index Name ::: {args.index_name}')
    
    if args.es_top_k < args.dr_top_k:
        raise ValueError(f' Top k number of elastic retrieval must be greater than Top k number of dense retrieval >>> [ Top k number of elastic retrieval : {args.es_top_k} / Top k number of dense retrieval : {args.dr_top_k} ]')
    
    main(args)



