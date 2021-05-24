from datasets import load_metric, load_from_disk
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

class DataProcessor():
    def __init__(self, tokenizer, max_length = 384, doc_stride = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
    
    def prepare_train_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        if 'question_type' in examples.keys() :
            tokenized_examples['question_type'] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            sequence_ids = tokenized_examples.sequence_ids(i)
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            if 'question_type' in examples.keys() :
                tokenized_examples['question_type'].append(examples['question_type'][sample_index])
                
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1):
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    def prepare_validation_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        if 'question_type' in examples.keys() :
            tokenized_examples['question_type'] = []

        for i in range(len(tokenized_examples['input_ids'])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            if 'question_type' in examples.keys() :
                tokenized_examples['question_type'].append(examples['question_type'][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            
        return tokenized_examples
    
    def train_tokenizer(self, train_dataset, column_names):
        train_dataset = train_dataset.map(
            self.prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )

        return train_dataset

    def val_tokenzier(self, val_dataset, column_names):
        val_dataset = val_dataset.map(
            self.prepare_validation_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )

        return val_dataset