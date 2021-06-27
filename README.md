# 📖 Stage 3 - MRC (Machine Reading Comprehension) ✏️
> More explanations of our codes are in 🔎[TEAM-IKYO's WIKI](https://github.com/TEAM-IKYO/Open-Domain-Question-Answering/wiki)

<!-- Hosted by [2021 Boostcamp AI Tech](https://boostcamp.connect.or.kr/)-->
![TEAM-IKYO’s ODQA Final Score_v3](https://user-images.githubusercontent.com/45359366/121469986-6ea03e80-c9f8-11eb-9fd1-355e50537450.png)

## 💻 Task Description
> More Description can be found at [AI Stages](http://boostcamp.stages.ai/competitions/31/overview/description)  
  
When you have a question like, "What is the oldest tree in Korea?", you may have asked it at a search engine. And these days it gives you an especially surprisingly accurate answer. How is it possible?

Question Answering is a field of research that creates artificial intelligence model that answers various kinds of questions. Among them, Open-Domain Question Answering is a more challenging issue because it has to find documents that can answer questions only using pre-built knowledge resources.

![image](https://user-images.githubusercontent.com/59340911/119260267-118d4600-bc0d-11eb-95bc-6ea68f7b0df4.png)

The model we'll create in this competition is made up of 2 stages.The first stage is called "retriever", which is the step to find question-related documents, and the next stage is called "reader", which is the step to read the document we've found at 1st stage and find the answer in the document. If we concatenate these two stages properly, we can make a question answering system that can answer no matter how tough questions are. The team which create a model that makes a more accurate answer will win this stage.
![image](https://user-images.githubusercontent.com/59340911/119260915-f1ab5180-bc0f-11eb-9ddc-cad4585bc8ce.png)

---

## 🗂 Directory
```
p3-mrc-team-ikyo
├── code
│   ├── arguments.py
│   ├── data_processing.py
│   ├── elasticsearch_retrieval.py
│   ├── inference.py
│   ├── mask.py
│   ├── mk_retrieval_dataset.py
│   ├── model
│   │   ├── ConvModel.py
│   │   ├── QAConvModelV1.py
│   │   ├── QAConvModelV2.py
│   │   └── QueryAttentionModel.py
│   ├── prepare_dataset.py
│   ├── question_labeling
│   │   ├── data_set.py
│   │   ├── question_labeling.py
│   │   └── train.py
│   ├── requirements.txt
│   ├── retrieval_dataset.py
│   ├── retrieval_inference.py
│   ├── retrieval_model.py
│   ├── retrieval_train.py
│   ├── run_elastic_search.py
│   ├── script
│   │   ├── inference.sh
│   │   ├── pretrain.sh
│   │   ├── retrieval_inference.sh
│   │   ├── retrieval_prepare_dataset.sh
│   │   ├── retrieval_train.sh
│   │   ├── run_elastic_search.sh
│   │   └── train.sh
│   ├── train_mrc.py
│   ├── trainer_qa.py
│   └── utils_qa.py
└── etc
    └── my_stop_dic.txt
```
