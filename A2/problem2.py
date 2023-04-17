#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE

# load the indices dictionary


def load_word_index_dict(file_path):
    word_index_dict = {}
    with open(file_path) as vocab:
        for i, line in enumerate(vocab):
            word = line.strip().lower()
            word_index_dict[word] = i
    return word_index_dict


def main():
    word_index_dict = load_word_index_dict("brown_vocab_100.txt")

    # Initialize the numpy vector of counts with zeros
    counts = np.zeros(len(word_index_dict))

    # Read the corpus file
    with open("brown_100.txt") as f:
        for line in f:
            # Split the sentence into a list of words and convert each word to lowercase
            words = line.strip().lower().split()

            # Iterate through the words and increment counts for each of the words they contain
            for word in words:
                if word in word_index_dict:
                    index = word_index_dict[word]
                    counts[index] += 1

    probs = counts / np.sum(counts)
    # write to unigram_probs.txt
    np.savetxt("unigram_probs.txt", probs)

    print(counts)
