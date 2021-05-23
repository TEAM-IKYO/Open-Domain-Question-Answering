import os
import re
import json
import pickle
import kss
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_metric, load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, util
from data_processing import *
from mask import mask_to_tokens


def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()
    return None


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset


def save_data(data_path, new_wiki):
    with open(data_path, 'w', encoding='utf-8') as make_file:
        json.dump(new_wiki, make_file, indent="\t", ensure_ascii=False)

        
def passage_split_400(text):
    num = len(text) // 400
    count = 1
    split_datas = kss.split_sentences(text)
    data_list = []
    data = ""
    for split_data in split_datas:
        if abs(len(data) - 400) > abs(len(data) + len(split_data) - 400) and count < num:
            if len(data) == 0:
                data += split_data
            else:
                data += (" " + split_data)
        elif count < num:
            data_list.append(data)
            count += 1
            data = ""
            data += split_data
        else:
            data += split_data
        
    data_list.append(data)
    return data_list, len(data_list)


def passage_split(text):
    length = len(text) // 2
    split_datas = kss.split_sentences(text)
    data_1 = ""
    data_2 = ""
    for split_data in split_datas:
        if abs(len(data_1) - length) > abs(len(data_1) + len(split_data) - length):
            if len(data_1) == 0:
                data_1 += split_data
            else:
                data_1 += (" " + split_data)
        else:
            if len(data_2) == 0:
                data_2 += split_data
            else:
                data_2 += (" " + split_data)
    
    return data_1, data_2


def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text


def run_preprocess(data_dict):
    context = data_dict["context"]
    start_ids = data_dict["answers"]["answer_start"][0]
    before = data_dict["context"][:start_ids]
    after = data_dict["context"][start_ids:]
    process_before = preprocess(before)
    process_after = preprocess(after)
    process_data = process_before + process_after
    ids_move = len(before) - len(process_before)
    data_dict["context"] = process_data
    data_dict["answers"]["answer_start"][0] = start_ids - ids_move
    return data_dict


def run_preprocess_to_wiki(data_dict):
    context = data_dict["text"]
    process_data = preprocess(context)
    data_dict["text"] = process_data
    return data_dict


def search_es(es_obj, index_name, question_text, n_results):
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    res = es_obj.search(index=index_name, body=query, size=n_results)
    return res


def make_custom_dataset(dataset_path) :
    if not (os.path.isdir("../data/train_dataset") or
            os.path.isdir("../data/wikipedia_documents.json")) :
        raise Exception ("Set the original data path to '../data'")
    
    train_f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
                        'context': Value(dtype='string', id=None),
                        'id': Value(dtype='string', id=None),
                        'question': Value(dtype='string', id=None)})

    if not os.path.isfile("../data/preprocess_wiki.json") :
        with open("../data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)
        new_wiki = dict()
        for ids in range(len(wiki)):
            new_wiki[str(ids)] = run_preprocess_to(wiki[str(ids)])
        with open('../data/preprocess_wiki.json', 'w', encoding='utf-8') as make_file:
            json.dump(new_wiki, make_file, indent="\t", ensure_ascii=False)

    if not os.path.isfile("/opt/ml/input/data/preprocess_train.pkl"):
        train_dataset = load_from_disk("../data/train_dataset")['train']
        val_dataset = load_from_disk("../data/train_dataset")['validation']
        
        new_train_data, new_val_data = [], []
        for data in train_dataset:
            new_data = run_preprocess(data)
            new_train_data.append(new_data)
        for data in val_dataset:
            new_data = run_preprocess(data)
            new_val_data.append(new_data)
        
        train_df = pd.DataFrame(new_train_data)
        val_df = pd.DataFrame(new_val_data)
        dataset = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f),
                               'validation': Dataset.from_pandas(val_df, features=train_f)})
        save_pickle(dataset_path, dataset)
        
        if 'preprocess' in dataset_path:
            return dataset
    
    if 'squad' in dataset_path :
        train_data = get_pickle("../data/preprocess_train.pkl")["train"]
        val_data = get_pickle("../data/preprocess_train.pkl")["validation"]
        korquad_data = load_dataset("squad_kor_v1")["train"]

        df_train_data = pd.DataFrame(train_data)
        df_val_data = pd.DataFrame(val_data)
        df_korquad_data = pd.DataFrame(korquad_data, columns=['answers', 'context', 'id', 'question'])
        df_total_train = pd.concat([df_train_data, df_korquad_data])

        dataset = DatasetDict({'train': Dataset.from_pandas(df_total_train, features=train_f), 
                                     'validation': Dataset.from_pandas(df_val_data, features=train_f)})
        save_pickle("../data/korquad_train.pkl", dataset)        
        return train_dataset

    if 'concat' in dataset_path :
        base_dataset = get_pickle("../data/preprocess_train.pkl")
        train_dataset, val_dataset = base_dataset["train"], base_dataset["validation"]

        train_data = [{"id" : train_dataset[i]["id"], "question" : train_dataset[i]["question"], 
                     "answers" : train_dataset[i]["answers"], "context" : train_dataset[i]["context"]}
                    for i in range(len(train_dataset))]
        val_data = [{"id" : val_dataset[i]["id"], "question" : val_dataset[i]["question"], 
                    "answers" : val_dataset[i]["answers"], "context" : val_dataset[i]["context"]}
                  for i in range(len(val_dataset))]

        config = {'host':'localhost', 'port':9200}
        es = Elasticsearch([config])

        k = 5 # k : how many contexts to concatenate
        for idx, train in enumerate(train_data):
            res = search_es(es, "wiki-index", question["question"], k)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits']]
            contexts = train["context"]
            count = 0
            for context in context_list:
                # if same context already exists, don't concatenate
                if train["context"] == context[0]:
                    continue
                contexts += " " + context[0]
                count += 1
                if count == (k-1):
                    break
            train_data[idx]["context"] = contexts

        for idx, val in enumerate(val_data):
            res = search_es(es, "wiki-index", question["question"], k)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits']]
            contexts = val["context"]
            count = 0
            for context in context_list:
                if val["context"] == context[0]:
                    continue
                contexts += " " + context[0]
                count += 1
                if count == (k-1):
                    break
            val_data[idx]["context"] = contexts
        
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        dataset = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f), 
                                      'validation': Dataset.from_pandas(val_df, features=train_f)})
        save_pickle(dataset_path, dataset)
        return dataset

    if "split_wiki_400" in dataset_path:
        with open("/opt/ml/input/data/preprocess_wiki.json", "r") as f:
            wiki = json.load(f)
        new_wiki = dict()
        for i in tqdm(range(len(wiki))):
            if len(wiki[str(i)]["text"]) < 800:
                new_wiki[str(i)] = wiki[str(i)]
                continue
            data_list, count = passage_split_400(wiki[str(i)]["text"])
            for j in range(count):
                new_wiki[str(i) + f"_{j}"] = {"text" : data_list[j], "corpus_source" : wiki[str(i)]["corpus_source"], 
                                              "url" :  wiki[str(i)]["url"], "domain" : wiki[str(i)]["domain"], 
                                              "title" : wiki[str(i)]["title"], "author" : wiki[str(i)]["author"], 
                                              "html" : wiki[str(i)]["html"],"document_id" : wiki[str(i)]["document_id"]}
                
        save_data("../data/wiki-index-split-400.json", new_wiki)
    
    if "split_wiki" in dataset_path and dataset_path != "split_wiki_400":
        with open("/opt/ml/input/data/preprocess_wiki.json", "r") as f:
            wiki = json.load(f)
            
        limit = 0
        if "800" in dataset_path:
            limit = 800
        if "1000" in dataset_path:
            limit = 1000
            
        new_wiki = dict()
        for i in tqdm(range(len(wiki))):
            if len(wiki[str(i)]["text"]) < limit:
                new_wiki[str(i)] = wiki[str(i)]
                continue
            data_1, data_2 = passage_split(wiki[str(i)]["text"])
            new_wiki[str(i) + f"_1"] = {"text" : data_1, "corpus_source" : wiki[str(i)]["corpus_source"], "url" :  wiki[str(i)]["url"], 
                                        "domain" : wiki[str(i)]["domain"], "title" : wiki[str(i)]["title"], "author" : wiki[str(i)]["author"], 
                                        "html" : wiki[str(i)]["html"], "document_id" : wiki[str(i)]["document_id"]}
            new_wiki[str(i) + f"_2"] = {"text" : data_2, "corpus_source" : wiki[str(i)]["corpus_source"], "url" :  wiki[str(i)]["url"], 
                                        "domain" : wiki[str(i)]["domain"], "title" : wiki[str(i)]["title"], 
                                        "author" : wiki[str(i)]["author"], "html" : wiki[str(i)]["html"], "document_id" : wiki[str(i)]["document_id"]}

        save_data(f"../data/split_wiki_{limit}.json")

        
def make_mask_dataset(dataset_path, k, tokenizer):
    base_dataset, opt = None, None
    if 'default' in dataset_path:
        base_dataset = get_pickle("../data/preprocess_train.pkl")
    if 'concat' in dataset_path:
        base_dataset = get_pickle("../data/concat_train.pkl")
    k = int(re.findall("\d", dataset_path)[0])
    
    data_processor = DataProcessor(tokenizer)
    train_dataset, val_dataset = base_dataset['train'], base_dataset['val']
    column_names = train_dataset.column_names
    train_dataset = data_processor.train_tokenizer(train_dataset, column_names)
    val_dataset = data_processor.val_tokenizer(val_dataset, column_names)
    
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    mask_dataset = mask_to_tokens(train_dataset, tokenizer, k, model)
    
    dataset = DatasetDict({'train': mask_dataset,
                           'validation': val_dataset})
    
    save_pickle(dataset_path, dataset)        
    return dataset

    
    
    
