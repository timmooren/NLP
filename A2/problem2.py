#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
from problem1 import load_word_index_dict


def evaluate_toy_corpus(probs, word_index_dict, filename, model):
    with open("toy_corpus.txt") as toy_corpus, open(filename, "w") as outfile:
        for line in toy_corpus:
            words = line.strip().lower().split()

            # initialize sentprob to 1
            sentprob = 1

            if model == "unigram":
                for word in words:
                    if word in word_index_dict:
                        sentprob *= probs[word_index_dict[word]]
            elif model == "bigram":
                for i in range(1, len(words)):
                    sentprob *= probs[word_index_dict[words[i - 1]],
                                      word_index_dict[words[i]]]
            elif model == "trigram":
                for i in range(2, len(words)):
                    sentprob *= probs[word_index_dict[words[i - 2]],
                                      word_index_dict[words[i - 1]],
                                      word_index_dict[words[i]]]

            # calculate perplexity using assignment formula
            sent_len = len(words)
            perplexity = 1 / (pow(sentprob, 1.0 / sent_len))

            outfile.write(f"{perplexity}\n")


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
                index = word_index_dict[word]
                counts[index] += 1

    print(counts)

    probs = counts / np.sum(counts)
    # write to unigram_probs.txt
    np.savetxt("unigram_probs.txt", probs)

    # evaluate toy corpus for assignment 6
    evaluate_toy_corpus(probs, word_index_dict, "unigram_eval.txt", "unigram")

    # Generate sentences using unigram model
    with open("unigram_generation.txt", "w") as f:
        for i in range(10):
            generated_sentence = GENERATE(
                word_index_dict, probs, "unigram", 50, None)
            f.write(generated_sentence + "\n")


if __name__ == "__main__":
    main()
