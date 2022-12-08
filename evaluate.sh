CUDA_VISIBLE_DEVICES=1,2,3 \
python evaluate.py --model_name 'bert-base-uncased' \
                --test_qp_path ./data/msmarco-passagetest2020-54-top1000.tsv \
                --test_qrels_path ./data/2020qrels-pass.txt \
                --eval_weights ./logs/bert-base-uncased/2022-12-07-23:04:46-0.6531/weights/model_bert-base-uncased_epoch_0000_ndcg_0.656.pth \
                --candidate_path ./logs/bert-base-uncased/2022-12-07-23:04:46-0.6531/test_ndcg.tsv \
                --batch_size 6 \
                --ngpus 1