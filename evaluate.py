from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
import argparse
from model.scorer import Scorer
from metrics.run import EvaluationQueries
from metrics.msmarco_eval import compute_NDCG_from_files
import torch


def main(args):
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = BertConfig.from_pretrained(args.model_name)
    bert = BertForSequenceClassification.from_pretrained(
        args.model_name, config=config)
    model = Scorer(tokenizer, bert, args.max_length, args.device)
    model.to(args.device)
    model.load_state_dict(torch.load(
        args.eval_weights, map_location=args.device), strict=False)
    if args.device == 'cuda' and args.ngpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.ngpus)))
    evaluator = EvaluationQueries(
        args.model_name, args.test_qp_path, args.batch_size)
    evaluator.score(model, args.candidate_path)
    for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        res = compute_NDCG_from_files(
            args.test_qrels_path, args.candidate_path, k)
        print(
            'Queries ranked: {}, NDCG@{}: {}'.format(
                res['querys_num'], k, res['NDCG'])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for the model
    '''
    parser.add_argument("--model_name", type=str,
                        help="Name of the HugginFace Model", default="bert-base-uncased")
    parser.add_argument("--device", type=str,
                        help="Type of device", default="cuda")
    parser.add_argument("--ngpus", type=int,
                        help="Number of GPUs", default=1)
    parser.add_argument("--eval_weights", type=str,
                        help="weights path", default='/disk16t/yzk/IR_homework/PassageReranking/logs/bert-base-uncased/2022-12-06 15:50:04/weights/model_bert-base-uncased_epoch_0002_ndcg_0.763.pth')
    '''
    Variables for dataset
    '''
    parser.add_argument("--test_qp_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/msmarco-passagetest2020-54-top1000.tsv")
    parser.add_argument("--test_qrels_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/2020qrels-pass.txt")
    parser.add_argument("--max_length", type=int,
                        help="max length of the tokenized input", default=256)
    parser.add_argument("--batch_size", type=int,
                        help="batch size", default=12)
    parser.add_argument("--num_classes", type=int,
                        help="number of output score class", default=2)
    '''
    Variables for result
    '''
    parser.add_argument("--candidate_path", type=str,
                        help="path to the candidate run .tsv file", default="test_result.tsv")

    '''
    Run main
    '''
    args = parser.parse_args()
    main(args)
