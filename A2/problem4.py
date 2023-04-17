#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
from problem2 import load_word_index_dict
from problem3 import get_counts, write_bigram_probs
import random
import codecs


def main():
    word_index_dict = load_word_index_dict("brown_vocab_100.txt")
    # initialize numpy 0s matrix
    counts = get_counts(word_index_dict, "brown_100.txt")

    # Add-α smoothing with α = 0.1
    counts += 0.1

    # normalize counts
    probs = normalize(counts, norm='l1', axis=1)

    # writeout bigram probabilities
    bigrams = [("all", "the"), ("the", "jury"),
                   ("the", "campaign"), ("anonymous", "calls")]
    write_bigram_probs(probs, word_index_dict, "smooth_probs.txt", bigrams)

if __name__ == "__main__":
    main()
