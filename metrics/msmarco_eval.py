import sys
import numpy as np
from collections import Counter

def load_reference(path_to_reference):
    qids_to_passages_rels = {}
    with open(path_to_reference, 'r') as f:
        for l in f:
            try:
                l = l.strip('\n').split('\t')
                qid = int(l[0])
                pid = int(l[2])
                if qid not in qids_to_passages_rels:
                    qids_to_passages_rels[qid] = {}
                qids_to_passages_rels[qid][pid] = int(l[3])
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_passages_rels


def load_candidate(path_to_candidate):
    qid_to_ranked_candidate_passages = {}
    with open(path_to_candidate, 'r') as f:
        for l in f:
            try:
                l = l.strip('\n').split('\t')
                qid = int(l[0])
                pid = int(l[2])
                rank = int(l[3])
                if qid not in qid_to_ranked_candidate_passages:
                    # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                    tmp = [0] * 1000
                    qid_to_ranked_candidate_passages[qid] = tmp
                qid_to_ranked_candidate_passages[qid][rank-1] = pid
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(
            qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def calculate_dcg(rels, K):
    dcg = 0
    for i in range(K):
        dcg += rels[i]/np.log2(i+2)
    return dcg


def compute_NDCG(qids_to_passageids_rels, qids_to_ranked_candidate_passages, K=10):
    NDCGs = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_passageids_rels:
            dcg = 0
            rels = []
            ranked_rels = []
            candidate_pids = qids_to_ranked_candidate_passages[qid]
            for i in range(len(candidate_pids)):
                pid = candidate_pids[i]
                if pid in qids_to_passageids_rels[qid]:
                    rel = qids_to_passageids_rels[qid][pid]
                    rels.append(rel)
                else:
                    rels.append(0)
            for pid in qids_to_passageids_rels[qid]:
                ranked_rels.append(qids_to_passageids_rels[qid][pid])
            if len(ranked_rels) < len(rels):
                ranked_rels.extend([0]*(len(rels)-len(ranked_rels)))
            dcg = calculate_dcg(rels, K)
            ranked_rels = sorted(ranked_rels, reverse=True)
            idcg = calculate_dcg(ranked_rels, K)
            if idcg == 0:
                ndcg = 0
            else:
                ndcg = dcg/idcg
            NDCGs.append(ndcg)
    NDCG = sum(NDCGs)/len(NDCGs)
    querys_num = len(NDCGs)
    res = {
        "NDCG": NDCG,
        "NDCGs": NDCGs,
        "querys_num": querys_num,
    }
    return res


def compute_NDCG_from_files(path_to_reference, path_to_candidate, K=10, perform_checks=True):
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(
            qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '':
            print(message)
    return compute_NDCG(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, K)


def main():
    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        metrics = compute_NDCG_from_files(
            path_to_reference, path_to_candidate)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')

    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        exit()


if __name__ == '__main__':
    main()
