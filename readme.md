# PassageReranking Baseline
implentment by pytorch

# dataset format

## train dataset
- collection.train.sampled.tsv

    pid passage
    
- queries.train.sampled.tsv

    qid query
- qidpidtriples.train.sampled.tsv

    qid positive_pid negative_pid


## validation and test dataset

- msmarco-passagetest2019-43-top1000.tsv

    qid pid query passage

- msmarco-passagetest2020-54-top1000.tsv

    qid pid query passage

# train
> train.sh

# test
evaluate.sh

# eval by ndcg
eval.py

or you can eval usr offical offered C++ script

download from https://trec.nist.gov/trec_eval/

```bash
tar xzvf trec_eval-9.0.7.tar.gz
cd trec_eval-9.0.7
make
./trec_eval-9.0.7/trec_eval -m qrels model_result
```
trec_eval.sh
