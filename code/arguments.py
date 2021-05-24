from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="xlm-roberta-large",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_custom_model: str = field(
        default='QAConvModelV2',
        metadata={"help": "Choose one of ['ConvModel', 'QueryAttentionModel', 'QAConvModelV1', 'QAConvModelV2']"}
    )
    use_pretrained_model: bool = field(
        default=False,
        metadata={"help": "use_pretrained_koquard_model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    retrieval_type: Optional[str] = field(
        default="elastic", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    retrieval_elastic_index: Optional[str] = field(
        default="wiki-index-split-800", metadata={"help": "Elastic search index name[wiki-index, wiki-index-split-400, wiki-index-split-800(best), wiki-index-split-1000]"}
    )
    retrieval_elastic_num: Optional[int] = field(
        default=35,
        metadata={"help": "The number of context or passage from Elastic search"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="question_type", metadata={"help": "Choose one of ['basic', 'preprocessed', 'concat', 'korquad', 'only_korquad', 'quetion_type', 'ai_hub', 'random_masking', 'token_masking']"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    train_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to train sparse/dense embedding (prepare for retrieval)."},
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help":"Whether to run passage retrieval using sparse/dense embedding )."},
    )

