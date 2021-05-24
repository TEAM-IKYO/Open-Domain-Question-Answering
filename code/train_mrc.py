import os
import sys
import time
import pickle
import random
import logging

import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_metric, load_from_disk, load_dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from model.ConvModel import ConvModel
from model.QueryAttentionModel import QueryAttentionModel
from model.QAConvModelV1 import QAConvModelV1
from model.QAConvModelV2 import QAConvModelV2
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize, AverageMeter, last_processing
from trainer_qa import QuestionAnsweringTrainer
from arguments import ModelArguments, DataTrainingArguments
from data_processing import DataProcessor
# from prepare_dataset import make_custom_dataset


def get_args() :
    '''훈련 시 입력한 각종 Argument를 반환하는 함수'''
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


def set_seed_everything(seed):
    '''Random Seed를 고정하는 함수'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

    return None


def get_model(model_args, training_args) :
    '''tokenizer, model_config, model, optimizer, scaler, shceduler를 반환하는 함수'''
    # Load pretrained model and tokenizer
    model_config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
        
    if model_args.use_custom_model == 'ConvModel' :
        model = ConvModel(model_args.config_name, model_config, model_args.tokenizer_name)
    elif model_args.use_custom_model == 'QueryAttentionModel' :
        model = QueryAttentionModel(model_args.config_name, model_config, model_args.tokenizer_name)
    elif model_args.use_custom_model == 'QAConvModelV1' :
        model = QAConvModelV1(model_args.config_name, model_config, model_args.tokenizer_name)
    elif model_args.use_custom_model == 'QAConvModelV2' :
        model = QAConvModelV2(model_args.config_name, model_config, model_args.tokenizer_name)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=model_config,
        )
        
    if model_args.use_pretrained_model:
        pretrained_model = torch.load(f'/opt/ml/output/{model_args.model_name_or_path}/{model_args.model_name_or_path}.pt')
        pretrained_model_state = deepcopy(pretrained_model.state_dict())
        model.load_state_dict(pretrained_model_state)
        del pretrained_model
        
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scaler = GradScaler()
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=12820, num_cycles=2)

    return tokenizer, model_config, model, optimizer, scaler, scheduler 


def get_pickle(pickle_path):
    '''Custom Dataset을 Load하기 위한 함수'''
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()

    return dataset

def get_data(data_args, training_args, tokenizer) :
    '''train과 validation의 dataloader와 dataset를 반환하는 함수'''
    if data_args.dataset_name == 'basic' :
        if os.path.isdir("../data/train_dataset") :
            dataset = load_from_disk("../data/train_dataset")
        else :
            raise Exception ("Set the data path to 'p3-mrc-team-ikyo/data/.'")
    elif data_args.dataset_name == 'preprocessed' :
        if os.path.isfile("../data/preprocess_train.pkl") :
            dataset = get_pickle("../data/preprocess_train.pkl")
        else :
            dataset = make_custom_dataset("../data/preprocess_train.pkl")
    elif data_args.dataset_name == 'concat' :
        if os.path.isfile("../data/concat_train.pkl") :
            dataset = get_pickle("../data/concat_train.pkl")
        else :
            dataset = make_custom_dataset("../data/concat_train.pkl")
    elif data_args.dataset_name == 'korquad' :
        if os.path.isfile("../data/korquad_train.pkl") :
            dataset = get_pickle("../data/korquad_train.pkl")
        else :
            dataset = make_custom_dataset("../data/korquad_train.pkl")
    elif data_args.dataset_name == "question_type":
        if os.path.isfile("../data/question_type.pkl") :
            dataset = get_pickle("../data/question_type.pkl")
        else :
            dataset = make_custom_dataset("../data/question_type.pkl")
    elif data_args.dataset_name == "ai_hub":
        if os.path.isfile("../data/ai_hub_dataset.pkl") :
            dataset = get_pickle("../data/ai_hub_dataset.pkl")
        else :
            dataset = make_custom_dataset("../data/ai_hub_dataset.pkl")
    elif data_args.dataset_name == "only_korquad":
        dataset = load_dataset("squad_kor_v1")
    elif data_args.dataset_name == "random_masking":
        if os.path.isfile("../data/random_mask_train.pkl") :
            dataset = get_pickle("../data/random_mask_train.pkl")
        else :
            dataset = make_custom_dataset("../data/random_mask_train.pkl")
    elif data_args.dataset_name == "token_masking":
        if os.path.isfile("../data/concat_token_mask_top_3.pkl") :
            dataset = get_pickle("../data/concat_token_mask_top_3.pkl")
        else :
            dataset = make_mask_dataset("../data/concat_token_mask_top_3.pkl", tokenizer)
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
    else :
        raise Exception ("dataset_name have to be one of ['basic', 'preprocessed', 'concat', 'korquad', 'only_korquad', 'question_type', 'ai_hub', 'random_masking', 'token_masking']")

    if data_args.dataset_name != "token_masking":
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        train_column_names = train_dataset.column_names
        val_column_names = val_dataset.column_names

        data_processor = DataProcessor(tokenizer, data_args.max_seq_length, data_args.doc_stride)
        train_dataset = data_processor.train_tokenizer(train_dataset, train_column_names)
        val_dataset = data_processor.val_tokenzier(val_dataset, val_column_names)

    data_collator = (DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None))
    train_iter = DataLoader(train_dataset, collate_fn = data_collator, batch_size=training_args.per_device_train_batch_size)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=training_args.per_device_eval_batch_size)

    return dataset, train_iter, val_iter, train_dataset, val_dataset


def post_processing_function(examples, features, predictions, text_data, data_args, training_args):
    '''Model의 Prediction을 Text 형태로 변환하는 함수'''
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": last_processing(v)} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in text_data["validation"]
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    '''Model의 Logit을 Context 단위로 연결하기 위한 함수'''
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


def custom_to_mask(batch, tokenizer):
    '''Question 부분에 Random Masking을 적용하는 함수'''
    mask_token = tokenizer.mask_token_id
    
    for i in range(len(batch["input_ids"])):
        # sep 토큰으로 question과 context가 나뉘어져 있다.
        sep_idx = np.where(batch["input_ids"][i].numpy() == tokenizer.sep_token_id)
        # q_ids = > 첫번째 sep 토큰위치
        q_ids = sep_idx[0][0]
        mask_idxs = set()
        while len(mask_idxs) < 1:
            # 1 ~ q_ids까지가 Question 위치
            ids = random.randrange(1, q_ids)
            mask_idxs.add(ids)

        for mask_idx in list(mask_idxs):
            batch["input_ids"][i][mask_idx] = mask_token
    
    return batch


def cal_loss(start_positions, end_positions, start_logits, end_logits):
    '''MRC Task에서 Loss를 계산하는 함수'''
    total_loss =None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
    return total_loss

def cal_query_loss(question_type, query_logits) :
    '''Sub Task에서 Loss를 계산하는 함수'''
    return nn.CrossEntropyLoss()(query_logits, question_type)/5

def training_per_step(model, optimizer, scaler, batch, model_args, data_args, training_args, tokenizer, device):
    '''매 step마다 학습을 하는 함수'''
    model.train()
    with autocast():
        mask_props = 0.8
        mask_p = random.random()
        if mask_p < mask_props:
            # 확률 안에 들면 mask 적용
            batch = custom_to_mask(batch, tokenizer)

        batch = batch.to(device)
        outputs = model(**batch)

        # output안에 loss가 들어있는 형태
        if model_args.use_custom_model:
            loss = cal_loss(batch["start_positions"], batch["end_positions"], outputs["start_logits"], outputs["end_logits"])
            if 'query_logits' in outputs.keys() and 'question_type' in batch.keys() :
                loss += cal_query_loss(batch['question_type'], outputs['query_logits'])
        else:
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item()


def validating_per_steps(epoch, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    '''특정 step마다 검증을 하는 함수'''
    metric = load_metric("squad")
    if "xlm" in model_args.tokenizer_name:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in test_loader :
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
    prediction = post_processing_function(text_data["validation"], test_dataset, output_numpy, text_data, data_args, training_args)
    val_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

    return val_metric


def train_mrc(model, optimizer, scaler, text_data, train_loader, test_loader, train_dataset, test_dataset, scheduler, model_args, data_args, training_args, tokenizer, device):
    '''training과 validating을 진행하는 함수'''
    prev_f1 = 0
    prev_em = 0
    global_steps = 0
    train_loss = AverageMeter()
    for epoch in range(int(training_args.num_train_epochs)):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for step, batch in pbar:
            # training phase
            loss = training_per_step(model, optimizer, scaler, batch, model_args, data_args, training_args, tokenizer, device)
            train_loss.update(loss, len(batch['input_ids']))
            global_steps += 1
            description = f"{epoch+1}epoch {global_steps: >5d}step | loss: {train_loss.avg: .4f} | best_f1: {prev_f1: .4f} | em : {prev_em: .4f}"
            pbar.set_description(description)
            if scheduler is not None :
                scheduler.step()

            # validating phase
            if global_steps % training_args.logging_steps == 0 :
                with torch.no_grad():
                    val_metric = validating_per_steps(epoch, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)
                if val_metric["f1"] > prev_f1:
                    torch.save(model, training_args.output_dir + f"/{training_args.run_name}.pt")
                    prev_f1 = val_metric["f1"]
                    prev_em = val_metric["exact_match"]
                wandb.log({
                'train/loss' : train_loss.avg,
                'train/learning_rate' : scheduler.get_last_lr()[0] if scheduler is not None else training_args.learning_rate,
                'eval/exact_match' : val_metric['exact_match'],
                'eval/f1_score' : val_metric['f1'],
                'global_steps': global_steps
                })
                train_loss.reset()
            else : 
                wandb.log({'global_steps':global_steps})


def main():
    '''각종 설정 이후 train_mrc를 실행하는 함수'''
    model_args, data_args, training_args = get_args()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    set_seed_everything(training_args.seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer, model_config, model, optimizer, scaler, scheduler  = get_model(model_args, training_args)
    text_data, train_loader, val_loader, train_dataset, val_dataset = get_data(data_args, training_args, tokenizer)
    model.cuda()

    if not os.path.isdir(training_args.output_dir) :
        os.mkdir(training_args.output_dir)

    # set wandb
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    wandb.login()
    wandb.init(project='P3-MRC', entity='team-ikyo', name=training_args.run_name)

    train_mrc(model, optimizer, scaler, text_data, train_loader, val_loader, train_dataset, val_dataset, scheduler, model_args, data_args, training_args, tokenizer, device)


if __name__ == "__main__":
    main()
