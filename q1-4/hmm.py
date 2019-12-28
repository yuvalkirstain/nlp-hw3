import os
import random
import time
from data import *
from collections import defaultdict, Counter
# yuvalk imports and constants
import numpy as np
BEFORE_WORD_MARK = '*'
END_SENT_MARK = 'STOP'

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """
    print("Start training")
    total_tokens = 0
    # yuvalk - was defaultdict(lambda: defaultdict(int)) - should be ok
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = [defaultdict(int) for i in range(5)]
    ### YOUR CODE HERE
    for sent in sents:
        sent = [(BEFORE_WORD_MARK, BEFORE_WORD_MARK)] * 2 + sent + [(END_SENT_MARK, END_SENT_MARK)]
        q_uni_counts[BEFORE_WORD_MARK] += 2
        q_bi_counts[(BEFORE_WORD_MARK, BEFORE_WORD_MARK)] += 1
        total_tokens += 2
        sent_len = len(sent)
        for i in range(2, sent_len):
            x_i, y_i = sent[i]
            y_im1 = sent[i - 1][1]
            y_im2 = sent[i - 2][1]
            e_word_tag_counts[(x_i, y_i)] += 1
            q_uni_counts[y_i] += 1
            q_bi_counts[(y_im1, y_i)] += 1
            q_tri_counts[(y_im2, y_im1, y_i)] += 1
            total_tokens += 1
    ### YOUR CODE HERE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, q_uni_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["-"] * (len(sent))
    ### YOUR CODE HERE
    lambda3 = 1 - (lambda1 + lambda2)
    tags = [key for key in q_uni_counts if q_uni_counts[key] > 0]
    tags.remove(BEFORE_WORD_MARK)
    tags.remove(END_SENT_MARK)
    # get probs - slide 32 lec 4
    def transition(y_im2, y_im1, y_i):
        uni = float(q_uni_counts[y_i]) / total_tokens
        bi = 0  if q_uni_counts[y_im1] == 0         else    float(q_bi_counts[(y_im1, y_i)]) / q_uni_counts[y_im1]
        tri = 0 if q_bi_counts[(y_im2, y_im1)] == 0 else    float(q_tri_counts[(y_im2, y_im1, y_i)]) / q_bi_counts[(y_im2, y_im1)]
        res = lambda3 * uni + lambda2 * bi + lambda1 * tri
        return np.log(res) if res > 0 else float('-inf')

    def emission(word, tag):
        if q_uni_counts[tag] == 0:
            return 0
        res = float(e_word_tag_counts[(word, tag)]) / q_uni_counts[tag]
        return np.log(res) if res > 0 else float('-inf')

    # viterbi - slide 48 lec 4
    pi = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    pi[-1][(BEFORE_WORD_MARK, BEFORE_WORD_MARK)] = 0 # for base case
    bp = {}
    q_cache = {}
    e_cache = {}
    l_sent = len(sent)

    # build pi, bp
    for k in range(l_sent):
        bp[k] = {}
        word = sent[k][0]
        for y_i in tags:
            if (word, y_i) not in e_cache:
                e_cache[(word, y_i)] = emission(word, y_i)
            e = e_cache[(word, y_i)]
            if e == float('-inf'):
                continue # if emission prob is zero then prob is 0. The tags don't matter.
            for y_im2, y_im1 in pi[k-1]: # we don't iterate through all tags, just those that are a part of a possible path
                if (y_im2, y_im1, y_i) not in q_cache:
                    q_cache[(y_im2, y_im1, y_i)] = transition(y_im2, y_im1, y_i)
                q = q_cache[(y_im2, y_im1, y_i)]
                p = pi[k - 1][(y_im2, y_im1)]
                res = p + q + e
                if res > pi[k][(y_im1, y_i)]:
                    pi[k][(y_im1, y_i)] = res
                    bp[k][(y_im1, y_i)] = y_im2
        # yuvalk - hack for case that all tags have zero prob
        if len(bp[k]) == 0:
            for (y_im2, y_im1), res in pi[k-1].items():
                for y_i in tags:
                    pi[k][(y_im1, y_i)] = res
                    bp[k][(y_im1, y_i)] = 'O'

    # update last and before last
    max_res = float('-inf')
    for y_im1, y_i in pi[l_sent - 1]:
        res = pi[l_sent - 1][(y_im1, y_i)] + transition(END_SENT_MARK, y_im1, y_i)
        if res > max_res:
            max_res = res
            predicted_tags[l_sent - 1] = y_i
            if l_sent > 1:
                predicted_tags[l_sent - 2] = y_im1

    # update the rest
    for k in range(len(sent) - 3, -1, -1):
        y_kp1 = predicted_tags[k + 1]
        y_kp2 = predicted_tags[k + 2]
        predicted_tags[k] = bp[k+2][(y_kp1, y_kp2)]

    ### YOUR CODE HERE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append(tuple(hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, 0.12, 0.6)))
        ### YOUR CODE HERE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


def lambdas_search(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                   e_word_tag_counts, e_tag_counts, grid=True):
    # need to add lambda1 and lambda2 parameters to hmm_eval
    maxf1 = 0
    maxl1 = -1
    maxl2 = -1
    if grid:
        for lambda1 in [i * 0.04 for i in range(25)]:
            for lambda2 in [j * 0.04 for j in range(25)]:
                if lambda1 + lambda2 > 1:
                    break
                print(lambda1, lambda2)
                token_cm, t = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                       e_word_tag_counts, e_tag_counts, lambda1, lambda2)
                p, r, f1 = t
                if f1 > maxf1:
                    maxl1 = lambda1
                    maxl2 = lambda2
                    maxf1 = f1
                    print(lambda1, lambda2, f1)
    else:
        for _ in range(1000):
            lambda1 = random.uniform(0, 1)
            lambda2 = random.uniform(0, 1 - lambda1)
            token_cm, t = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                 e_word_tag_counts, e_tag_counts, lambda1, lambda2)
            p, r, f1 = t
            if f1 > maxf1:
                maxl1 = lambda1
                maxl2 = lambda2
                maxf1 = f1
                print(lambda1, lambda2, f1)
    print(maxl1, maxl2, maxf1)
    return maxl1, maxl2, maxf1

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    # best lambdas are 0.12, 0.6
    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    # unmark to do grid or random search
    # print(lambdas_search(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
    #                      e_word_tag_counts, e_tag_counts))

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
