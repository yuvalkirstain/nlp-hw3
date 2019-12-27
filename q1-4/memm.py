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
    for i in range(1, 6):
        feature_name = 'word_pref_' + str(i)
        features[feature_name] = ''
        if i <= len(curr_word):
            features[feature_name] = curr_word[:i]
            
    for i in range(1, 6):
        feature_name = 'word_suf_' + str(i)
        features[feature_name] = ''
        if i <= len(curr_word):
            features[feature_name] = curr_word[-i:]
            
    def pipe_strings(s1, s2):
        return s1 + '|' + s2
            
    features['tag_unigram'] = ''
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
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    for i in range(len(sent)):
        sentence = list(zip(sent, predicted_tags))
        features = extract_features(sentence, i)
        vectorized_sent = vectorize_features(vec, features)
        index = logreg.predict(vectorized_sent)[0]
        predicted_tags[i] = index_to_tag_dict[index]
    ### YOUR CODE HERE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    num_tags = len(index_to_tag_dict) - 1
    PI = np.zeros([len(sent), num_tags, num_tags])
    BP_ix = np.zeros([len(sent), num_tags, num_tags])
    for i in range(len(sent)):
        q = np.zeros([num_tags, num_tags, num_tags])
        for prev_tag_index in range(num_tags):
            for prev_prev_tag_index in range(num_tags):
                tags = ["O"] * (len(sent))
                if i > 0:
                    tags[i - 1] = index_to_tag_dict[prev_tag_index]
                if i > 1:
                    tags[i - 2] = index_to_tag_dict[prev_prev_tag_index]
                sentence = list(zip(sent, tags))
                features = extract_features(sentence, i)
                vectorized_sent = vectorize_features(vec, features)
                q[prev_prev_tag_index, prev_tag_index, :] = logreg.predict_proba(vectorized_sent)
        if i > 0:
            last_PI = PI[i-1]
        else:
            last_PI = np.ones([num_tags, num_tags])
        last_PI = np.repeat(last_PI[:, :, np.newaxis], num_tags, axis=2)
        r = last_PI * q
        BP_ix[i] = np.argmax(r, axis=0)
        PI[i] = np.max(r, axis=0)
    last_PI = PI[len(sent) - 1]
    argmax = last_PI.argmax()
    curr_tag_index = argmax % num_tags
    prev_tag_index = int(argmax / num_tags)
    predicted_tags[-1] = index_to_tag_dict[curr_tag_index]
    if len(sent) > 1:
        predicted_tags[-2] = index_to_tag_dict[prev_tag_index]
    for i in range(len(sent) - 1, 1, -1):
        prev_prev_tag_index = BP_ix[i, prev_tag_index, curr_tag_index]
        predicted_tags[i - 2] = index_to_tag_dict[prev_prev_tag_index]
        curr_tag_index = prev_tag_index
        prev_tag_index = curr_tag_index
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
