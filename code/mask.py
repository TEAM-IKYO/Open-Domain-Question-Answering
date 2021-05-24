import numpy as np
import random
import re
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def make_word_index_dict(tokens, cls_token, sep_token):
    word_start = False
    word_index = {}
    word = ''
    index = []
    
    for i, t in enumerate(tokens):
        if t == cls_token:
            continue
        elif t == sep_token:
            break
        if t.startswith('▁') and not word_start:
            word_start = True
            word += t
            index.append(i)
            if tokens[i+1].startswith('▁'):
                word_start = False
                word_index[word.replace('▁', '')] = index
                word = ''
                index = []
        if not t.startswith('▁') and word_start:
            word += t
            index.append(i)
            if i < 383 and (tokens[i+1].startswith('▁') or tokens[i+1] == sep_token):
                word_start = False
                word_index[word.replace('▁', '')] = index
                word = ''
                index = []

    return word_index


def mask_to_tokens(batch, tokenizer, top_k, model):
    '''
    Span 단위로 Random Masking을 적용하는 함수
    '''
    mask_token = tokenizer.mask_token_id

    for i, input_id in enumerate(batch["input_ids"]):
        sep_idx = np.where(input_id.numpy() == tokenizer.sep_token_id)[0][0]
        pad_idx = 0
        if tokenizer.pad_token_id in input_id.numpy():
            pad_idx = np.where(input_id.numpy() == tokenizer.pad_token_id)[0][0]
        tokenizer.pad_token_id
        question = tokenizer.decode(input_id[1:sep_idx]) # sep_idx[0][0]: 첫 번째 sep 토큰 위치
        answer  = tokenizer.decode(input_id[batch['start_positions'][i]:batch['end_positions'][i]+1])
        context = None
        if pad_idx == 0:
            context = tokenizer.decode(input_id[sep_idx+2:-1])
        else:
            context = tokenizer.decode(input_id[sep_idx+2:pad_idx-1])
        q_emb = model.encode(question)
        tokens = tokenizer.convert_ids_to_tokens(input_id)
        
        word_dict = make_word_index_dict(tokens, tokenizer, answer)
        
        sim_dict = {}
        for word in word_dict.keys():
            sim = cos_sim(q_emb, model.encode(word))
            if sim > 0.35:
                sim_dict[sim] = word_dict[word]

        ordered_sim_dict = sorted(sim_dict.items(), reverse=True)
        tokens_to_mask = []
        if len(ordered_sim_dict) < top_k:
            for val in ordered_sim_dict:
                tokens_to_mask.extend(val[1])
        else:
            for val in ordered_sim_dict[:top_k]:
                tokens_to_mask.extend(val[1])

        for token_idx in list(tokens_to_mask):
            input_id[token_idx] = mask_token
        
        batch["input_ids"][i] = input_id
    
    return batch


def mask_to_random(dataset):
    context_list = []
    question_list = []
    id_list = []
    answer_list = []
    train_dataset = dataset["train"]

    for i in tqdm(range(train_dataset.num_rows)):
        question = train_dataset["question"]
        
        for word, pos in mecab.pos(text):
            # first_word = True
            # 첫번째 단어는 무조건 Masking(질문 중 가장 중요한 의미를 가지고 있다고 생각)
            # 두번째 단어부터는 20% 확률로 Masking
            # 하나의 단어만 Masking
            if pos in {"NNG", "NNP"} and (random.random() > 0.8):
                context_list.append(train_dataset["context"])
                question_list.append(re.sub(word, "MASK", question)) # tokenizer.mask_token
                id_list.append(train_dataset[i]["id"])
                answer_list.append(train_dataset[i]["answers"])

    random.Random(2021).shuffle(context_list)
    random.Random(2021).shuffle(question_list)
    random.Random(2021).shuffle(id_list)
    random.Random(2021).shuffle(answer_list)

    dataset["train"] = Dataset.from_dict({"id" : id_list,
                                          "context": context_list, 
                                          "question": question_list,
                                          "answers": answer_list})

    return dataset["train"]