# 📖 Stage 3 - MRC (Machine Reading Comprehension) ✏️
> More explanations of our codes are in 🔎[TEAM-IKYO's WIKI](https://github.com/TEAM-IKYO/Open-Domain-Question-Answering/wiki)

<!-- Hosted by [2021 Boostcamp AI Tech](https://boostcamp.connect.or.kr/)-->

<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/59340911/119260977-29b29480-bc10-11eb-8543-cf7ef73ddcd4.png">
</p>


## 🥇 Final Scores
||Public Leaderboard|Private Leaderboard|
| :----------: |  :--------:  | :--------:  |
||![image](https://user-images.githubusercontent.com/59340911/119259895-4dbfa700-bc0b-11eb-9633-6eb4d4c633d7.png)|![image](https://user-images.githubusercontent.com/59340911/119259913-5d3ef000-bc0b-11eb-8c10-2f667e960d31.png)|
|**Score**|1st|1st|
|**EM**|72.92|70.00|
|**F1**|81.52|79.78|

---

## 💻 Task Description
> More Description can be found at [AI Stages](http://boostcamp.stages.ai/competitions/31/overview/description)  
  
When you have a question like, "What is the oldest tree in Korea?", you may have asked it at a search engine. And these days it gives you an especially surprisingly accurate answer. How is it possible?

Question Answering is a field of research that creates artificial intelligence model that answers various kinds of questions. Among them, Open-Domain Question Answering is a more challenging issue because it has to find documents that can answer questions only using pre-built knowledge resources.

![image](https://user-images.githubusercontent.com/59340911/119260267-118d4600-bc0d-11eb-95bc-6ea68f7b0df4.png)

The model we'll create in this competition is made up of 2 stages.The first stage is called "retriever", which is the step to find question-related documents, and the next stage is called "reader", which is the step to read the document we've found at 1st stage and find the answer in the document. If we concatenate these two stages properly, we can make a question answering system that can answer no matter how tough questions are. The team which create a model that makes a more accurate answer will win this stage.
![image](https://user-images.githubusercontent.com/59340911/119260915-f1ab5180-bc0f-11eb-9ddc-cad4585bc8ce.png)

---
