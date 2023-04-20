#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
from problem1 import load_word_index_dict
from problem2 import evaluate_toy_corpus
import codecs


def get_counts(word_index_dict, corpus_file_path):
    # initialize numpy 0s matrix
    counts = np.zeros((len(word_index_dict), len(word_index_dict)))
    # iterate through file and update counts
    with codecs.open(corpus_file_path) as f:
        for line in f:
            words = line.lower().strip().split()
            # previous word will be updated continuously
            previous_word = '<s>'
            for word in words[1:]:
                counts[word_index_dict[previous_word],
                       word_index_dict[word]] += 1
                previous_word = word
    return counts


def write_bigram_probs(probs, word_index_dict, output_file_path, bigrams):
    with codecs.open(output_file_path, "w") as output_file:
        for bigram in bigrams:
            prob = probs[word_index_dict[bigram[0]],
                         word_index_dict[bigram[1]]]
            output_file.write(f"p({bigram[1]} | {bigram[0]}) = {prob}\n")


def main():
    word_index_dict = load_word_index_dict("brown_vocab_100.txt")

    counts = get_counts(word_index_dict, "brown_100.txt")

    # normalize counts
    probs = normalize(counts, norm='l1', axis=1)

    # writeout bigram probabilities
    bigrams = [("all", "the"), ("the", "jury"),
               ("the", "campaign"), ("anonymous", "calls")]
    write_bigram_probs(probs, word_index_dict, "bigram_probs.txt", bigrams)

    # evaluate toy corpus for assignment 6
    evaluate_toy_corpus(probs, word_index_dict, "bigram_eval.txt")

if __name__ == "__main__":
    main()
