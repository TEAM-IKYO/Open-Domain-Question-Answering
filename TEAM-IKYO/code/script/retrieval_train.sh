python3 retrieval_train.py --output_dir ../retrieval_output/ \
                           --model_checkpoint bert-base-multilingual-cased \
                           --seed 2021 \
                           --epoch 1 \
                           --learning_rate 1e-5 \
                           --gradient_accumulation_steps 1 \
                           --top_k 20 \
                           --run_name best_dense_retrieval