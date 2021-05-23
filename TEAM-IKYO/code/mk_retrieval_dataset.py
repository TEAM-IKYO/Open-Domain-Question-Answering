#!/usr/bin/env python
# coding: utf-8

import os
import re
import time
import json
import torch
import pickle
import argparse

from konlpy.tag import Mecab
from transformers import AutoTokenizer
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
from elasticsearch import Elasticsearch
from subprocess import Popen, PIPE, STDOUT
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_metric, load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset

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

def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text

def mk_new_file(mode, files, top_k, es, index_name):
    if mode == 'test':
        new_files = {'id':[], 'question':[], 'top_k':[]}
        for file in files:
            question_text = file['question']
            
            top_list = elastic_retrieval(es, index_name, question_text, top_k)
            top_list = [text for text, score in top_list]

            new_files['id'].append(file['id'])
            new_files['question'].append(question_text)
            new_files['top_k'].append(top_list)
        return new_files
    
    else:
        new_files = {'context':[], 'id':[], 'question':[], 'top_k':[], 'answer_idx':[], 'answer':[], 'start_idx':[]}
        for file in files:
            start_ids = file["answers"]["answer_start"][0]
            
            before = file["context"][:start_ids]
            after = file["context"][start_ids:]
            
            process_before = preprocess(before)
            process_after = preprocess(after)
            new_context = process_before + process_after
            
            start_idx = start_ids - len(before) + len(process_before)

            question_text = file['question']
            top_list = elastic_retrieval(es, index_name, question_text, top_k)
            top_list = [text for text, score in top_list]
            
            if not new_context in top_list:
                top_list = top_list[:-1] + [new_context]
                answer_idx = top_k-1
            else:
                answer_idx = top_list.index(new_context)

            answer = file['answers']['text'][0]

            new_files['context'].append(new_context)
            new_files['id'].append(file['id'])
            new_files['question'].append(question_text)
            new_files['top_k'].append(top_list)
            new_files['answer_idx'].append(answer_idx)
            new_files['answer'].append(answer)
            new_files['start_idx'].append(start_idx)
        return new_files

def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def main(args):
    train_file = load_from_disk("../data/train_dataset")["train"]
    validation_file = load_from_disk("../data/train_dataset")["validation"]
    test_file = load_from_disk("../data/test_dataset")["validation"]
    
    es = elastic_setting(args.index_name)

    print('wait...', end='\r')
    new_train_file =  mk_new_file('train', train_file, args.top_k, es, args.index_name)
    print('make train dataset!!')
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_train.pkl'), new_train_file)
    
    print('wait...', end='\r')
    new_valid_file =  mk_new_file('valid', validation_file, args.top_k, es, args.index_name)
    print('make validation dataset!!')
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_valid.pkl'), new_valid_file)
    
    print('wait...', end='\r')
    new_test_file =  mk_new_file('test', test_file, args.top_k, es, args.index_name)
    print('make test dataset!!')    
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_test.pkl'), new_test_file)
    
    print('complete!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='../data/retrieval_dataset')
    parser.add_argument('--index_name', type=str, default="nori-index")
    
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    print(f'TOP K ::: {args.top_k}')
    print(f'SAVE PATH ::: {args.save_path}')
    print(f'INDEX NAME ::: {args.index_name}')
    
    main(args)
