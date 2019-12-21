import os
from data import *
from collections import defaultdict, Counter # yuvalk I added counter

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    word2tags2count = defaultdict(lambda: Counter())
    for sent in train_data:
        for word, tag in sent:
            word2tags2count[word][tag] += 1

    word2max_tag = {}
    for word, counter in word2tags2count.items():
        tag, _ = counter.most_common(1)[0]
        word2max_tag[word] = tag

    return word2max_tag
    ### YOUR CODE HERE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append([pred_tags[word] for word, _ in sent])
        ### YOUR CODE HERE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs) # yuvalk F1 is 0.80


if __name__ == "__main__":
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

