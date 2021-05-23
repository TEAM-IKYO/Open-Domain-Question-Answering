### 0. Install requirements

```python
# download elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.6.2
elasticsearch-7.6.2/bin/elasticsearch-plugin install analysis-nori
# collapse-hide
pip install elasticsearch
```

### +) stop word 파일 추가

- `elasticsearch-7.6.2/config/user_dic/my_stop_dic.txt` 경로에 아래 파일 추가

    [my_stop_dic.txt](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97406f13-640f-485f-a13c-a016df1d9d21/my_stop_dic.txt)

### 1. preprocess.ipynb 실행

- new_baseline에 맞도록 파일 경로 수정하여 반영
- `/opt/ml/input/data/preprocess_wiki.json` 파일을 얻기 위한 과정임😎

[preprocess.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8cfc158-8ce0-43d5-9136-9f18f3210858/preprocess.ipynb)

### 2. 기존 elastic search 실행 여부 확인하여 삭제(optional)

```bash
ps -ef | grep elastic
kill -9 <PID>
```
![image](https://user-images.githubusercontent.com/59340911/119265351-5b802700-bc21-11eb-901b-c66b7fb2e53d.png)

실행 중인 elastic search 죽이기 예시

### 3. es_retrieval.ipynb 실행

만약 **설치 에러가 이유 없이 발생하는 경우 위 2번의 실행 중인 elastic search 삭제 후 노트북 재실행을 권장함** 

⇒ index 바꾸어 elastic search를 재실행 할 때도 2번처럼 삭제 후 재실행

[es_retrieval.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afb793b3-549b-42c2-8e79-0d9dabf28c20/es_retrieval.ipynb)

### 참고) ikyo's index(위 `es_retrieval.ipynb` 파일에 그대로 반영해둠)

- 아래 인덱스 코드를 변경하여 elastic search에 여러 옵션을 줄 수 있음

```python
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
```

![image](https://user-images.githubusercontent.com/59340911/119265362-663abc00-bc21-11eb-9827-0fdba23c8ac5.png)
elastic search 띄우기에 성공한 경우 위와 같은 메세지 출력됨

![image](https://user-images.githubusercontent.com/59340911/119265378-705cba80-bc21-11eb-882b-3c2039732ee3.png)
wiki 등록 시 확인되는 출력(8분 정도 소요)

![image](https://user-images.githubusercontent.com/59340911/119265385-78b4f580-bc21-11eb-8079-e65e649d215e.png)

최종 출력되는 metrics
