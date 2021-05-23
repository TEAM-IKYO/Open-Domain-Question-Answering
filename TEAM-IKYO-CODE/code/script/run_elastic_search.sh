# elasticsearch-7.6.2 설치
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q -P ../etc/
tar -xzf ../etc/elasticsearch-7.6.2-linux-x86_64.tar.gz -C ../etc/
chown -R daemon:daemon ../etc/elasticsearch-7.6.2

# Python Library 설치
pip install elasticsearch
pip install tqdm

# nori Tokenizer 설치
../etc/elasticsearch-7.6.2/bin/elasticsearch-plugin install analysis-nori

# elastic search stop word 설정
mkdir ../etc/elasticsearch-7.6.2/config/user_dic
cp ../etc/my_stop_dic.txt ../etc/elasticsearch-7.6.2/config/user_dic/.

# python script file 실행
python3 run_elastic_search.py --path_to_elastic ../etc/elasticsearch-7.6.2/bin/elasticsearch --index_name wiki-index
python3 run_elastic_search.py --path_to_elastic ../etc/elasticsearch-7.6.2/bin/elasticsearch --index_name wiki-index-split-400
python3 run_elastic_search.py --path_to_elastic ../etc/elasticsearch-7.6.2/bin/elasticsearch --index_name wiki-index-split-800
python3 run_elastic_search.py --path_to_elastic ../etc/elasticsearch-7.6.2/bin/elasticsearch --index_name wiki-index-split-1000

# elastic search 실행 여부 확인
ps -ef | grep elastic