import pandas as pd
from tqdm import tqdm


class EvaluationQuery():
    def __init__(self, model_name, qid, pids, query, passages, batch_size):
        self.model_name = model_name
        self.qid = qid
        self.pids = pids
        self.query = query
        self.passages = passages
        self.batch_size = batch_size

    def __str__(self):
        return "<{} qid:{}, pids:{}>".format(type(self), self.qid, self.passages.keys())

    def score(self, scorer):
        self.scores = scorer.score_query_passages(
            self.query, self.passages, self.batch_size)
        score_str = ""
        for rank, (score, pid) in enumerate(sorted(
                list(zip(self.scores, self.pids)), key=lambda x: x[0], reverse=True)):
            score_str = score_str + \
                "{}\tQ0\t{}\t{}\t{}\t{}\n".format(
                    self.qid, pid, rank+1, score, self.model_name)
        return score_str


class EvaluationQueries():
    def __init__(self, model_name, qp_path, batch_size):
        df = pd.read_csv(qp_path, sep='\t', header=None, names=[
            'qid', 'pid', 'queries', 'passages'])

        self.evaluation_queries = []
        for qid in df['qid'].unique():
            pids = df[df['qid'] == qid]['pid'].to_list()
            queries = df[df['qid'] == qid]['queries'].to_list()
            passages = df[df['qid'] == qid]['passages'].to_list()
            self.evaluation_queries.append(
                EvaluationQuery(model_name, qid, pids, queries[0], passages, batch_size))

    def __str__(self):
        s = '<EvaluationQueries '
        for i, evaluation_query in enumerate(self.evaluation_queries):
            s += evaluation_query.__str__()
            if i == 9:
                s += '...'
                break
        s += ' />'
        return s

    def score(self, scorer, output_path):
        score_str = ""
        print('Evaluation on {} queries'.format(
            len(self.evaluation_queries)))
        for evaluation_query in tqdm(self.evaluation_queries, desc="Evaluation in progress"):
            score_str = score_str + evaluation_query.score(scorer)
        f = open(output_path, "w")
        f.write(score_str)
        f.close()
