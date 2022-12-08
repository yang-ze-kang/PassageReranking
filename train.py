import torchmetrics
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from tqdm import tqdm
import argparse
from model.scorer import Scorer
from utils.dataset import XYDataset
from utils.log import Log
from torch.utils.data import DataLoader
from metrics.run import EvaluationQueries
from metrics.msmarco_eval import compute_NDCG_from_files
import torch.nn.functional as F
import torch
import time
import os


def get_train_dataset(querypath, passpath, triplepath, tokenizer, max_length, batch_size, num_samples, shuffle=10000):
    qid2query = {}
    with open(querypath, 'r') as f:
        for line in tqdm(f, desc="Reading train query file"):
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            qid2query[line[0]] = line[1]
    pid2pass = {}
    with open(passpath, 'r') as f:
        for line in tqdm(f, desc="Reading train passage file"):
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            pid2pass[line[0]] = line[1]
    X = []
    y = []
    count = 0
    with open(triplepath, 'r') as f:
        for line in tqdm(f, desc="Reading train triple file"):
            count += 1
            if count >= num_samples:
                break
            line = line.strip('\n').split('\t')
            assert len(line) == 3, '\\t in querie or passage. \nQUERIE: {}\nPASSAGE1: {}\nPASSAGE2: {}'.format(
                line[0], line[1], line[2])
            relevant_inputs = tokenizer.encode_plus(text=str(qid2query[line[0]]),
                                                    text_pair=str(
                                                        pid2pass[line[1]]),
                                                    max_length=max_length,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True)
            X.append([relevant_inputs['input_ids'],
                      relevant_inputs['attention_mask'],
                      relevant_inputs['token_type_ids']
                      ])
            y.append([0, 1])
            # Add no relevant passage
            no_relevant_inputs = tokenizer.encode_plus(text=str(qid2query[line[0]]),
                                                       text_pair=str(
                                                           pid2pass[line[2]]),
                                                       max_length=max_length,
                                                       pad_to_max_length=True,
                                                       return_token_type_ids=True,
                                                       return_attention_mask=True)
            X.append([no_relevant_inputs['input_ids'],
                      no_relevant_inputs['attention_mask'],
                      no_relevant_inputs['token_type_ids']
                      ])
            y.append([1, 0])
    train_dataset = DataLoader(XYDataset(
        X, y), batch_size=batch_size, shuffle=shuffle, collate_fn=XYDataset.collate_fn)
    return train_dataset, len(X)+1


def get_val_dataset(val_dataset_path, val_qrels_path, tokenizer, max_length, batch_size, num_samples, shuffle=10000):
    X = []
    y = []
    qidpid2rel = {}
    with open(val_qrels_path, 'r') as f:
        for line in tqdm(f, desc='Reading qrels file'):
            line = line.strip('\n').split('\t')
            assert len(line) == 4
            if line[0] not in qidpid2rel:
                qidpid2rel[line[0]] = {}
            qidpid2rel[line[0]][line[2]] = line[3]
    with open(val_dataset_path, 'r') as f:
        count = 0
        for line in tqdm(f, desc="Reading val file"):
            count += 1
            if count >= num_samples:
                break
            line = line.strip('\n').split('\t')
            assert len(line) == 4
            relevant_inputs = tokenizer.encode_plus(text=str(line[2]),
                                                    text_pair=str(line[3]),
                                                    max_length=max_length,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True)
            X.append([relevant_inputs['input_ids'],
                      relevant_inputs['attention_mask'],
                      relevant_inputs['token_type_ids']
                      ])
            if line[1] not in qidpid2rel[line[0]]:
                y.append([1, 0])
            else:
                if qidpid2rel[line[0]][line[1]] != '0':
                    y.append([0, 1])
                else:
                    y.append([1, 0])
    dataset = DataLoader(XYDataset(
        X, y), batch_size=batch_size, shuffle=shuffle, collate_fn=XYDataset.collate_fn)
    return dataset, len(X)+1


def one_step(model, inputs, gold, loss, acc, confusion_matrix):
    gold = gold.to(args.device)
    inputs = inputs.to(args.device)
    outputs = model(inputs, gold[:, -1])
    t_loss = outputs[0]
    predictions = F.softmax(outputs[1].cpu(), dim=1)[:, -1]
    gold = gold.cpu()[:, -1]
    loss(t_loss.item())
    acc(predictions, gold)
    confusion_matrix(predictions, gold)
    return t_loss


def main(args, model_name, max_length, batch_size, num_samples, epochs, learning_rate, epsilon, save_path):
    '''
    Load Hugging Face tokenizer and model
    '''
    log = Log(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    bert = BertForSequenceClassification.from_pretrained(
        model_name, config=config)
    model = Scorer(tokenizer, bert, max_length,  args.device)
    model = model.to(args.device)
    if args.freeze == True:
        for name, params in model.named_parameters():
            if 'encoder' in name or 'embeddings' in name:
                params.requires_grad = False
    print('Parameters to train:')
    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name)
    if args.device == 'cuda' and args.ngpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.ngpus)))

    '''
    Create train and validation dataset
    '''
    train_dataset, train_length = get_train_dataset(args.train_query_path, args.train_passage_path, args.train_triple_path,
                                                    tokenizer, max_length, batch_size, num_samples)
    val_dataset, val_length = get_val_dataset(args.val_qp_path, args.val_qrels_path,
                                              tokenizer, max_length, batch_size, num_samples)

    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, eps=epsilon)
    '''
    Define metrics
    '''
    train_loss = torchmetrics.MeanMetric()
    val_loss = torchmetrics.MeanMetric()
    train_acc = torchmetrics.Accuracy("binary")
    val_acc = torchmetrics.Accuracy("binary")
    train_confusion_matrix = torchmetrics.ConfusionMatrix("binary")
    val_confusion_matrix = torchmetrics.ConfusionMatrix("binary")

    mrr = EvaluationQueries(args.model_name, args.val_qp_path, args.batch_size)

    '''
    Training loop over epochs
    '''
    save_model_dir = os.path.join(save_path, 'weights')
    os.mkdir(save_model_dir)
    model_save_path_template = 'model_{model_name}_epoch_{epoch:04d}_ndcg_{mrr:.3f}.pth'
    template_train_step = '\nTrain Step {}: \nLoss: {}, Acc: {}, Confusion matrix:\n{}\n'
    template_val_epoch = '\nVal Epoch {}: \nLoss: {}, Acc: {}, Confusion matrix:\n{}\n'
    previus_ndcg = 0.0
    log_file = open(os.path.join(save_path, 'log.txt'), 'w')

    for epoch in range(epochs):
        train_loss.reset()
        train_acc.reset()
        train_confusion_matrix.reset()
        val_loss.reset()
        val_acc.reset()
        val_confusion_matrix.reset()

        """
        train
        """
        model.train()
        training_step = 0
        for inputs, gold in tqdm(train_dataset, desc="Training in progress", total=int(train_length/batch_size+1)):
            training_step += 1
            optimizer.zero_grad()
            loss = one_step(model, inputs, gold, train_loss,
                            train_acc, train_confusion_matrix)
            loss.backward()
            optimizer.step()
            '''
            Validation loop every XXXX steps
            '''
            if training_step % 100 == 0:
                log.append_num1('train_loss', train_loss.compute().numpy())
                log.append_num1('train_acc', train_acc.compute().numpy())
                s = template_train_step.format(training_step,
                                               train_loss.compute(),
                                               train_acc.compute(),
                                               train_confusion_matrix.compute(),
                                               )
                train_loss.reset()
                train_acc.reset()
                train_confusion_matrix.reset()
                print(s)
                log_file.write(s)

        model.eval()
        for inputs, gold in tqdm(val_dataset, desc="Validation in progress", total=int(val_length/batch_size+1)):
            one_step(model, inputs, gold, val_loss,
                     val_acc, val_confusion_matrix)
        s = template_val_epoch.format(epoch+1,
                                      val_loss.compute(),
                                      val_acc.compute(),
                                      val_confusion_matrix.compute()
                                      )
        log.append_num1('val_loss', val_loss.compute())
        log.append_num1('val_acc', val_acc.compute())
        val_loss.reset()
        val_acc.reset()
        val_confusion_matrix.reset()
        print(s)
        log_file.write(s)
        model_save_path = os.path.join(
            save_model_dir, 'epoch_{}.pth'.format(epoch))
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)
        if (epoch+1) % args.mrr_every == 0:
            mrr.score(model, args.candidate_path)
            res = compute_NDCG_from_files(
                args.val_qrels_path, args.candidate_path, 10)
            s = 'Queries ranked: {}, NDCG@10: {}'.format(
                res['querys_num'], res['NDCG'])
            print(s)
            log_file.write(s)
            log.append_num1('NDCG@10', res['NDCG'])
            if res['NDCG'] > previus_ndcg:
                previus_ndcg = res['NDCG']
                model_save_path = os.path.join(save_model_dir, model_save_path_template.format(
                    model_name=model_name, epoch=epoch, mrr=previus_ndcg))
                print('Saving: ', model_save_path)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for the model
    '''
    parser.add_argument("--model_name", type=str,
                        help="Name of the HugginFace Model", default="bert-base-uncased")
    parser.add_argument("--freeze", action='store_true')
    parser.add_argument("--device", type=str,
                        help="Type of device", default="cuda")
    parser.add_argument("--ngpus", type=int,
                        help="Number of GPUs", default=1)
    '''
    Variables for dataset
    '''
    parser.add_argument("--train_query_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/queries.train.sampled.tsv")
    parser.add_argument("--train_passage_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/collection.train.sampled.tsv")
    parser.add_argument("--train_triple_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/qidpidtriples.train.sampled.tsv")
    parser.add_argument("--val_qp_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/msmarco-passagetest2019-43-top1000.tsv")
    parser.add_argument("--val_qrels_path", type=str, help="path to the train .tsv file",
                        default="./IR_2021_Project/2019qrels-pass.txt")
    parser.add_argument("--max_length", type=int,
                        help="max length of the tokenized input", default=256)
    parser.add_argument("--batch_size", type=int,
                        help="batch size", default=12)
    parser.add_argument("--num_classes", type=int,
                        help="number of output score class", default=2)
    parser.add_argument("--num_samples", type=int,
                        help="number of samples", default=50000)

    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int,
                        help="number of epochs", default=5)
    parser.add_argument("--learning_rate", type=float,
                        help="learning rate", default=1e-5)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=1e-8)
    parser.add_argument("--save_path", type=str,
                        help="path to the save folder", default="logs")

    '''
    Variables for evaluation
    '''
    parser.add_argument("--bm25_path", type=str, help="path to the BM25 run .tsv file",
                        default="data/evaluation/bm25/run.dev.small.tsv")
    parser.add_argument("--passages_path", type=str, help="path to the BM25 passages .json file",
                        default="data/passages/passages.bm25.small.json")
    parser.add_argument("--queries_path", type=str, help="path to the BM25 queries .tsv file",
                        default="data/queries/queries.dev.small.tsv")
    parser.add_argument(
        "--n_top", type=int, help="number of passages to re-rank after BM25", default=10)
    parser.add_argument("--n_queries_to_evaluate", type=int,
                        help="number of queries to evaluate for MMR", default=1000)
    parser.add_argument("--mrr_every", type=int,
                        help="number of epochs between mrr eval", default=1)
    parser.add_argument("--candidate_path", type=str,
                        help="path to the candidate run .tsv file", default="data/evaluation/model/run.tsv")

    '''
    Run main
    '''
    args = parser.parse_args()
    now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    save_path = os.path.join(args.save_path, args.model_name, now_time)
    os.makedirs(os.path.join(save_path))
    main(args, args.model_name, args.max_length, args.batch_size,
         args.num_samples, args.epochs, args.learning_rate, args.epsilon, save_path)
