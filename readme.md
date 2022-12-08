# PassageReranking Baseline
implentment by pytorch

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
