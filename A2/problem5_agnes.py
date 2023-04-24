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
import random
import codecs
from problem6_agnes import compute_perplexity
smooth = True


def get_counts_bigram(word_index_dict, corpus_file_path):
    # initialize numpy 0s matrix (2d)
    counts = np.zeros((len(word_index_dict), len(word_index_dict)))
    # iterate through file and update counts
    with codecs.open(corpus_file_path) as f:
        for line in f:
            words = line.strip().split()
            # previous word will be updated continuously
            previous_word = '<s>'
            for word in words[1:]:
                word = word.lower()
                counts[word_index_dict[previous_word],
                       word_index_dict[word]] += 1
                previous_word = word
    return counts

def convert_counts_to_probs(counts_trigram, counts_bigram):
    
    # itterate over each trigram in the counts matrix
    for i in range(counts_trigram.shape[0]):
        for j in range(counts_trigram.shape[1]):
            # find correponding count in digram matrix
            count_bi = counts_bigram[i, j]
            if count_bi != 0:
                # divide every bigram count in k by the corresponding bigram count
                for k in range(counts_trigram.shape[2]):
                    counts_trigram[i, j, k] /= count_bi
    return counts_trigram
    
def get_counts_trigram(word_index_dict, corpus_file_path):
    # initialize numpy 0s matrix (3d)
    counts = np.zeros((len(word_index_dict), len(word_index_dict), len(word_index_dict)))
    # itterate trough file and update counts
    with codecs.open(corpus_file_path) as f:
        
        for line in f:
            words = line.strip().split()
            
            # previous word will be updated continuously
            word_1 = words[0].lower()
            word_2 = words[1].lower()
            
            for word in words[2:]:
                word = word.lower()
                counts[word_index_dict[word_1],
                        word_index_dict[word_2],
                        word_index_dict[word]] += 1
                word_1 = word_2
                word_2 = word
    return counts
                

def write_bigram_probs(probs, word_index_dict, output_file_path, bigrams):
    with codecs.open(output_file_path, "w") as output_file:
        for bigram in bigrams:
            prob = probs[word_index_dict[bigram[0]],
                         word_index_dict[bigram[1]]]
            output_file.write(f"p({bigram[1]} | {bigram[0]}) = {prob}\n")
            
def write_trigram_probs(probs, word_index_dict, output_file_path, trigrams):
    with codecs.open(output_file_path, "w") as output_file:
        for trigram in trigrams:
            prob = probs[word_index_dict[trigram[0][0]],
                         word_index_dict[trigram[0][1]],
                         word_index_dict[trigram[1]]]
            output_file.write(f"p({trigram[1]} | {trigram[0][0]}, {trigram[0][1]}) = {prob}\n")


def main():
    word_index_dict = load_word_index_dict("brown_vocab_100.txt")

    counts_bi = get_counts_bigram(word_index_dict, "brown_100.txt")

    # normalize counts
    probs = normalize(counts_bi, norm='l1', axis=1)
    
    print(len(word_index_dict))

    # writeout bigram probabilities
    bigrams = [("all", "the"), ("the", "jury"),
               ("the", "campaign"), ("anonymous", "calls")]
    
    write_bigram_probs(probs, word_index_dict, "bigram_probs.txt", bigrams)
    
    # compute perplexity
    compute_perplexity("toy_corpus.txt", "bigram_eval.txt", probs, word_index_dict, "bigram")
    
    counts_tri = get_counts_trigram(word_index_dict, "brown_100.txt")
    
    # smoothen counts by 0.1
    if smooth == True:
        counts_tri += 0.1
        counts_bi += 0.1*len(word_index_dict)
    
    
    probs = convert_counts_to_probs(counts_tri, counts_bi)
    
    trigrams = [
        (("in", "the"), "past"),
        (("in", "the"), "time"),
        (("the", "jury"), "said"),
        (("the", "jury"), "recommended"),
        (("jury", "said"), "that"),
        (("agriculture", "teacher"), ","),
    ] 
    
    write_trigram_probs(probs, word_index_dict, "trigram_probs.txt", trigrams)
    compute_perplexity("toy_corpus.txt", "trigram_eval.txt", probs, word_index_dict, "trigram")
    
    return

if __name__ == "__main__":
    main()
