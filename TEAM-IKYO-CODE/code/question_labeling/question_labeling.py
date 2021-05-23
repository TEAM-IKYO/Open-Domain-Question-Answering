import torch
import pickle
import pandas as pd
from tqdm import tqdm
from data_set import *
from datasets import load_metric, load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset

def get_pickle(pickle_path):
    '''Custom Dataset을 Load하기 위한 함수'''
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()

    return dataset


def get_data():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    data = get_pickle("../../data/concat_train.pkl")
    
    train_token, train_labels = tokenized_testset(data["train"], tokenizer)
    val_token, val_labels = tokenized_testset(data["validation"], tokenizer)
    
    train_set = RE_Dataset(train_token, train_labels)
    val_set = RE_Dataset(val_token, val_labels)

    train_iter = DataLoader(train_set, batch_size=1)
    val_iter = DataLoader(val_set, batch_size=1)

    return train_iter, val_iter


def question_labeling(model, train_iter, val_iter):
    train_file = get_pickle("../../data/concat_train.pkl")["train"]
    validation_file = get_pickle("../../data/concat_train.pkl")["validation"]

    train_qa = [{"id" : train_file[i]["id"], "question" : train_file[i]["question"], "answers" : train_file[i]["answers"], "context" : train_file[i]["context"]} for i in range(len(train_file))]
    validation_qa = [{"id" : validation_file[i]["id"], "question" : validation_file[i]["question"], "answers" : validation_file[i]["answers"], "context" : validation_file[i]["context"]} for i in range(len(validation_file))]
    
    device = "cuda:0"
    for step, (input_ids, attention_mask, labels) in tqdm(enumerate(train_iter), total=len(train_iter), position=0, leave=True):
        score = model(input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        pred = torch.argmax(score, 1).detach().cpu().numpy()
        train_qa[step]["question_type"] = pred
    
    for step, (input_ids, attention_mask, labels) in tqdm(enumerate(val_iter), total=len(val_iter), position=0, leave=True):
        score = model(input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        pred = torch.argmax(score, 1).detach().cpu().numpy()
        validation_qa[step]["question_type"] = pred
    
    train_df = pd.DataFrame(train_qa)
    val_df = pd.DataFrame(validation_qa)
        
    return train_df, val_df


def save_data(train_df, val_df):
    train_f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
                    'context': Value(dtype='string', id=None),
                    'id': Value(dtype='string', id=None),
                    'question': Value(dtype='string', id=None),
                    'question_type' : Value(dtype='int32', id=None)})
    
    train_datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f), 'validation': Dataset.from_pandas(val_df, features=train_f)})
    file = open("../../data/question_type.pkl", "wb")
    pickle.dump(train_datasets, file)
    file.close()
        
        
def main():
    model = torch.load("../../output/question_model.pt")
    train_iter, val_iter = get_data()
    train_df, val_df = question_labeling(model, train_iter, val_iter)
    save_data(train_df, val_df)

if __name__ == "__main__":
    main()
    data_set = get_pickle("../../data/question_type.pkl")
    print(data_set)
    