
from metrics.msmarco_eval import compute_NDCG_from_files

if __name__ == '__main__':
    val_qrels_path = '/disk16t/yzk/IR_homework/IR_2021_Project/2019qrels-pass.txt'
    candidate_path = '/disk16t/yzk/IR_homework/PassageReranking/data/evaluation/model/run.tsv'
    res = compute_NDCG_from_files(val_qrels_path, candidate_path, 10)
    print(
        'Queries ranked: {}, NDCG@10: {}'.format(
            res['querys_num'], res['NDCG'])
    )
