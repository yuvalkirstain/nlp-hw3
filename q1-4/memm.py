from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE

    ### YOUR CODE HERE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    pref = {'word_pref_' + str(i) : curr_word[:i] for i in range(1, 6) if i < len(curr_word)}
    suf = {'word_suf_' + str(i): curr_word[-i:] for i in range(1, 6) if i < len(curr_word)}
    features.update(pref)
    features.update(suf)

    def pipe_strings(s1, s2):
        return s1 + '|' + s2
            
    # features['tag_unigram'] = '' todo what is the purpose?
    features['next'] = next_word
    features['tag_bigram'] = prev_tag
    features['tag_trigram'] = pipe_strings(prevprev_tag, prev_tag)
    features['word_tag_prev'] = pipe_strings(prev_word, prev_tag)
    features['word_tag_prevprev'] = pipe_strings(prevprev_word, prevprev_tag)
    
    ### YOUR CODE HERE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["-"] * (len(sent))
    ### YOUR CODE HERE
    prev_word = prev_prev_word = '<st>'
    prev_tag = prev_prev_tag = '*'
    for k in range(len(sent)):
        curr_word = sent[k]
        next_word = sent[k + 1] if k + 1 < len(sent) else '</s>'
        features = extract_features_base(curr_word, next_word, prev_word, prev_prev_word, prev_tag,
                                         prev_prev_tag)
        vectorized_sent = vectorize_features(vec, features)
        index = logreg.predict(vectorized_sent)[0]
        predicted_tags[k] = index_to_tag_dict[index]
        prev_prev_tag = prev_tag
        prev_tag = predicted_tags[k]
        prev_prev_word = prev_word
        prev_word = curr_word
    ### YOUR CODE HERE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["-"] * (len(sent)) # todo change back
    ### YOUR CODE HERE
    def pipe_strings(s1, s2):
        return s1 + '|' + s2

    tag_to_index_dict = {tag : ind for ind, tag in index_to_tag_dict.items() if tag != '*'}
    pi = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    pi[-1][('*', '*')] = 0  # for base case
    bp = {}
    l_sent = len(sent)
    tags = ['O', 'ORG', 'MISC', 'PER', 'LOC']
    # build pi, bp
    prev_word = prev_prev_word = '<st>'
    for k in range(l_sent):
        bp[k] = {}
        curr_word = sent[k]
        next_word = sent[k + 1] if k + 1 < len(sent) else '</s>'
        features = []
        curr_features = extract_features_base(curr_word, next_word, prev_word, prev_prev_word, '', '')
        tags2ind = {}
        for prev_prev_tag, prev_tag in pi[k - 1]:  # we don't iterate through all tags, just those that are a part of a possible path
            p = pi[k - 1][(prev_prev_tag, prev_tag)]
            if p <= float('-inf'):
                continue
            curr_features['tag_bigram'] = prev_tag
            curr_features['tag_trigram'] = pipe_strings(prev_prev_tag, prev_tag)
            curr_features['word_tag_prev'] = pipe_strings(prev_word, prev_tag)
            curr_features['word_tag_prevprev'] = pipe_strings(prev_prev_word, prev_prev_tag)
            tags2ind[(prev_prev_tag, prev_tag)] = len(features)
            features.append(dict(curr_features))
        vectorized_sent = vec.transform(features)
        q_probs = logreg.predict_proba(vectorized_sent)
        q_probs = np.log(q_probs)
        q_probs[q_probs < -2] = float('-inf')  # pruning
        for prev_prev_tag, prev_tag in pi[k - 1]:
            p = pi[k - 1][(prev_prev_tag, prev_tag)]
            if p <= float('-inf'):
                continue
            q_prob = q_probs[tags2ind[(prev_prev_tag, prev_tag)]]
            for cur_tag in tags:
                q = q_prob[tag_to_index_dict[cur_tag]]
                res = p + q
                if res > pi[k][(prev_tag, cur_tag)]:
                    pi[k][(prev_tag, cur_tag)] = res
                    bp[k][(prev_tag, cur_tag)] = prev_prev_tag
        # yuvalk - hack for case that all tags have zero prob
        if len(bp[k]) == 0:
            for (prev_prev_tag, prev_tag), res in pi[k - 1].items():
                for cur_tag in tags:
                    pi[k][(prev_tag, cur_tag)] = res
                    bp[k][(prev_tag, cur_tag)] = 'O'
        prev_prev_word = prev_word
        prev_word = curr_word

    # update last and before last
    max_res = float('-inf')
    for prev_tag, cur_tag in pi[l_sent - 1]:
        res = pi[l_sent - 1][(prev_tag, cur_tag)]
        if res > max_res:
            max_res = res
            predicted_tags[l_sent - 1] = cur_tag
            if l_sent > 1:
                predicted_tags[l_sent - 2] = prev_tag

    # update the rest
    for k in range(len(sent) - 3, -1, -1):
        y_kp1 = predicted_tags[k + 1]
        y_kp2 = predicted_tags[k + 2]
        predicted_tags[k] = bp[k + 2][(y_kp1, y_kp2)]

    ### YOUR CODE HERE
    return predicted_tags

def should_log(sentence_index):
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        greedy_pred_tag_seqs.append(memm_greedy(words, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        viterbi_pred_tag_seqs.append(memm_viterbi(words, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        ### YOUR CODE HERE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
