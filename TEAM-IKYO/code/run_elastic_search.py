import argparse
import json
import os
import time
from subprocess import Popen, PIPE, STDOUT

from datasets import load_from_disk
from elasticsearch import Elasticsearch
from prepare_dataset import make_custom_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def populate_index(es_obj, index_name, evidence_corpus):

    for i, rec in enumerate(tqdm(evidence_corpus)):
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')
    return


def set_datas(args) :
    if not os.path.isfile("../data/preprocess_train.pkl") :
        make_custom_dataset("../data/preprocess_train.pkl")
#     train_file = load_from_disk("../data/train_dataset")["train"]
#     validation_file = load_from_disk("../data/train_dataset")["validation"]
    train_file = load_from_disk("../data/train_dataset")["train"]
    validation_file = load_from_disk("../data/train_dataset")["validation"]

    #[wiki-index, wiki-index-split-400, wiki-index-split-800, wiki-index-split-1000]
    if args.index_name == 'wiki-index':
        dataset_path = "../data/preprocess_wiki.json"
    elif args.index_name == 'wiki-index-split-400':
        dataset_path = "../data/split_wiki_400.json"
    elif args.index_name == 'wiki-index-split-800':
        dataset_path = "../data/split_wiki_800.json"
    elif args.index_name == 'wiki-index-split-1000':
        dataset_path = "../data/split_wiki_1000.json"
        
    if not os.path.isfile(dataset_path) :
        print(dataset_path)
        make_custom_dataset(dataset_path)

    with open(dataset_path, "r") as f:
        wiki = json.load(f)
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    qa_records = [{"example_id" : train_file[i]["id"], "document_title" : train_file[i]["title"], "question_text" : train_file[i]["question"], "answer" : train_file[i]["answers"]} for i in range(len(train_file))]
    wiki_articles = [{"document_text" : wiki_contexts[i]} for i in range(len(wiki_contexts))]
    return qa_records, wiki_articles


def set_index_and_server(args) :
    es_server = Popen([args.path_to_elastic],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    time.sleep(30)

    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    index_config = {
        "settings": {
            "analysis": {
                "filter":{
                    "my_stop_filter": {
                        "type" : "stop",
                        "stopwords_path" : "user_dic/my_stop_dic.txt"
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter" : ["my_stop_filter"]
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict", 
            "properties": {
                "document_text": {"type": "text", "analyzer": "nori_analyzer"}
                }
            }
        }

    print('elastic serach ping :', es.ping())
    print(es.indices.create(index=args.index_name, body=index_config, ignore=400))

    return es


def main(args) :
    print('Start to Set Elastic Search')
    _, wiki_articles = set_datas(args)
    es = set_index_and_server(args)
    populate_index(es_obj=es, index_name=args.index_name, evidence_corpus=wiki_articles)
    print('Finish')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_elastic', type=str, default='elasticsearch-7.6.2/bin/elasticsearch', help='Path to Elastic search')
    parser.add_argument('--index_name', type=str, default='wiki-index', help='Elastic search index name[wiki-index, wiki-index-split-400, wiki-index-split-800, wiki-index-split-1000]')

    args = parser.parse_args()
    main(args)