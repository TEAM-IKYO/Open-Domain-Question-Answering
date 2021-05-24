python3 inference.py --output_dir ../output/baseline_train/submission \
                    --model_name_or_path ../output/baseline_train/baseline_train.pt \
                    --tokenizer_name deepset/xlm-roberta-large-squad2 \
                    --config_name deepset/xlm-roberta-large-squad2 \
                    --retrieval_type elastic_sentence_transformer \
                    --retrieval_elastic_index wiki-index-split-800 \
                    --retrieval_elastic_num 35 \
                    --use_custom_model QAConvModelV2 \
                    --do_predict