#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs


word_index_dict = {}
with codecs.open("brown_vocab_100.txt") as vocab:
    for i, line in enumerate(vocab):
        word = line.strip().lower()
        word_index_dict[word] = i


# initialize numpy 0s matrix
counts = np.zeros((len(word_index_dict), len(word_index_dict)))

# iterate through file and update counts
with codecs.open("brown_100.txt") as f:
    for line in f:
        words = line.strip().split()

        # previous word will be updated continuously
        previous_word = '<s>'

        for word in words[1:]:
            word = word.lower()
            # increment count for bigram
            counts[word_index_dict[previous_word], word_index_dict[word]] += 1
            previous_word = word

# normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
# Write specific bigram probabilities to bigram_probs.txt
with codecs.open("bigram_probs.txt", "w") as output_file:
    bigrams = [("all", "the"), ("the", "jury"), ("the", "campaign"), ("anonymous", "calls")]

    for bigram in bigrams:
        prob = probs[word_index_dict[bigram[0]], word_index_dict[bigram[1]]]
        output_file.write(f"p({bigram[1]} | {bigram[0]}) = {prob}\n")


