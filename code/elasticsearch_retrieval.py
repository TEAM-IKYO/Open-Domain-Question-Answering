import json
import os
import time

from elasticsearch import Elasticsearch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm

def elastic_setting(index_name='wiki-index'):
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])
    
    return es, index_name


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
