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

from tqdm import tqdm
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from konlpy.tag import Mecab
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from sentence_transformers import SentenceTransformer
import kss

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from elasticsearch_retrieval import *
from data_processing import DataProcessor
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize, cos_sim
from trainer_qa import QuestionAnsweringTrainer
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

def get_pickle(pickle_path):
    '''Custom Dataset을 Load하기 위한 함수'''
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def get_config():
    """
    get config

    Returns:
        model_args: model arguments
        data_args: data arguments
        training_args: training arguments
    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

def fix_seed(seed):
    """
    fix_seed

    Args:
        seed (int): seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)


def get_model(model_args, training_args):
    """
    get model

    Args:
        model_args : model arguments
        training_args : training arguments

    Returns:
        tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=True
    )
    model = torch.load(model_args.model_name_or_path)

    return tokenizer, model


def run_elasticsearch(text_data, concat_num, model_args, is_sentence_trainformer):
    """
    run elasticsearch and filter sentences

    Args:
        text_data
        concat_num: number of texts to import from elasticsearch
        is_sentence_trainformer: whether sentence trainformer is used or not

    Returns:
        datasets: test data
        scores: elasticsearch scores
    """
    # elastic setting & load index
    es, index_name = elastic_setting(model_args.retrieval_elastic_index)
    # load sentence transformer model
    if is_sentence_trainformer:
        model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    question_texts = text_data["validation"]["question"]
    total = []
    scores = []
    
    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        # concat_num만큼 context 검색
        context_list = elastic_retrieval(es, index_name, question_text, concat_num)
        score = []
        concat_context = []
        
        if is_sentence_trainformer:
            # question embedding
            question_embedding = model.encode(question_text)
            # use sentence transformer
            for i in range(len(context_list)):
                temp_context = []
                # separate context by sentence 
                for sent in kss.split_sentences(context_list[i][0]):
                    # question embedding과 sentence embedding의 cosine similarity 계산
                    # -0.2 보다 높은 sentence만 append
                    if cos_sim(question_embedding, model.encode(sent)) > -0.2:
                        temp_context.append(sent)

                concat_context.append(" ".join(temp_context))
        else:
            # not use sentence transformer
            for i in range(len(context_list)):
                concat_context.append(context_list[i][0])

        tmp = {
            "question" : question_text,
            "id" : text_data["validation"]["id"][step],
            "context" : " <SEP> ".join(concat_context) if is_sentence_trainformer else " ".join(concat_context)
        }

        score.append(context_list[0][1])
        total.append(tmp)
        scores.append(score)

    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                'id': Value(dtype='string', id=None),
                'question': Value(dtype='string', id=None)})
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets, scores

def run_concat_dense_retrival(text_data, concat_num):
    test_data = get_pickle("../data/test_ex70_dr70_dense.pkl")
    question_texts = text_data["validation"]["question"]
    total = []
    scores = []

    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        context_list = test_data[question_text][:concat_num]
        score = []
        concat_context = ""
        # 유일하게 다른 부분 : context list를 concat 시켜주는 부분
        for i in range(len(context_list)):
            if i == 0 :
                concat_context += context_list[i][0]
            else:
                concat_context += " " + context_list[i][0]
        
        tmp = {
            "question" : question_text,
            "id" : text_data["validation"]["id"][step],
            "context" : concat_context
        }
        
        score.append(context_list[0][1])
        total.append(tmp)
        scores.append(score)
        
    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                'id': Value(dtype='string', id=None),
                'question': Value(dtype='string', id=None)})
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    
    return datasets, scores

def get_data(model_args, training_args, tokenizer, text_data_path = "../data/test_dataset"): # 경로 변경 ../data/test_dataset
    """
    get data

    Args:
        model_args: model arguments
        training_args: training arguments
        tokenizer: tokenizer
        text_data_path: Defaults to "../data/test_dataset"

    Returns:
        text_data, val_iter, val_dataset, scores
    """
    text_data = load_from_disk(text_data_path)

    # run_ lasticsearch
    if "elastic" in model_args.retrieval_type:
        is_sentence_trainformer = False
        if "sentence_trainformer" in model_args.retrieval_type:
            is_sentence_trainformer = True
        # number of text to concat
        concat_num = model_args.retrieval_elastic_num
        text_data, scores = run_elasticsearch(text_data, concat_num, model_args, is_sentence_trainformer)
    elif model_args.retrieval_type == "dense":
        concat_num = model_args.retrieval_elastic_num
        text_data, scores = run_concat_dense_retrival(text_data, concat_num)
    
    column_names = text_data["validation"].column_names

    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )
    # 데이터 tokenize(mrc 모델안에 들어 갈 수 있도록)
    data_processor = DataProcessor(tokenizer)
    val_text = text_data["validation"]
    val_dataset = data_processor.val_tokenzier(val_text, column_names)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=1)

    return text_data, val_iter, val_dataset, scores


def post_processing_function(features, predictions, text_data, data_args, training_args):
    """
    post processing

    Args:
        features, predictions, text_data, data_args, training_args

    Returns:
        inference or evaluation results
    """
    predictions = postprocess_qa_predictions(
        examples=text_data["validation"],
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex["answers"].strip()}
            for ex in text_data["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    step = 0

    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)

    for i, output_logit in enumerate(start_or_end_logits):
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    """
    Create prediction json using MRC model

    Args:
        model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device
    """
    
    metric = load_metric("squad")
    # xlm의 input 예외처리
    if "xlm" in model_args.tokenizer_name:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()

    all_start_logits = []
    all_end_logits = []

    t = time.time()
    # start predic
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    for step, batch in pbar:
        batch = batch.to(device)
        outputs = model(**batch)

        if model_args.use_custom_model:
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
        else:
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits


        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())
    
    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits
    
    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(test_dataset, output_numpy, text_data, data_args, training_args)


def remove_particle(training_args):
    """
    remove particle

    Args:
        training_args
    """
    # load tokenizer
    mecab = Mecab()
    kkma = Kkma()
    hannanum = Hannanum()
    # load prediction file
    with open(os.path.join(training_args.output_dir, "predictions.json"), "r") as f:
        prediction_json = json.load(f)

    prediction_dict = dict()
    for mrc_id in prediction_json.keys():
        final_predictions = prediction_json[mrc_id]
        pos_tag = mecab.pos(final_predictions)
        
        # 조사가 있는 경우 삭제
        if final_predictions[-1] == "의":
            min_len = min(len(kkma.pos(final_predictions)[-1][0]), len(mecab.pos(final_predictions)[-1][0]), len(hannanum.pos(final_predictions)[-1][0]))
            if min_len == 1:
                final_predictions = final_predictions[:-1]
        elif pos_tag[-1][-1] in {"JX", "JKB", "JKO", "JKS", "ETM", "VCP", "JC"}:
            final_predictions = final_predictions[:-len(pos_tag[-1][0])]

        prediction_dict[str(mrc_id)] = final_predictions
    
    # save final results
    with open(os.path.join(training_args.output_dir, "final_predictions.json"), 'w', encoding='utf-8') as make_file:
        json.dump(prediction_dict, make_file, indent="\t", ensure_ascii=False)
    print(prediction_dict)


def main():
    # get arguments
    model_args, data_args, training_args = get_config()
    # fix seed
    fix_seed(training_args.seed)
    # set device
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get tokenizer, model
    tokenizer, model = get_model(model_args, training_args)
    model.cuda()

    if not os.path.isdir(training_args.output_dir) :
        os.mkdir(training_args.output_dir)

    # load data
    text_data, test_loader, test_dataset, scores = get_data(model_args, training_args, tokenizer)
    # prediction
    predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)
    # remove particle
    remove_particle(training_args)

if __name__ == "__main__":
    main()