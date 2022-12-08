CUDA_VISIBLE_DEVICES=2,3 \
python train.py --model_name 'bert-base-uncased' \
                --train_query_path ./data/queries.train.sampled.tsv \
                --train_passage_path ./data/collection.train.sampled.tsv \
                --train_triple_path ./data/qidpidtriples.train.sampled.tsv \
                --val_qp_path ./data/msmarco-passagetest2019-43-top1000.tsv \
                --val_qrels_path ./data/2019qrels-pass.txt \
                --batch_size 16 \
                --num_samples 50000 \
                --learning_rate 5e-5 \
                --ngpus 1