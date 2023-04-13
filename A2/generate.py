#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np

'''
HOW TO USE:

word_index_dict: your dictionary mapping words to their indices in the probability vector/matrix

probs: either a probability vector for the unigram model, or a probability matrix from the bigram model. 

Make sure the matrix is arranged so each row represents the probability of a word given the 
seen word. i.e. if we had a 3 word vocabulary and our probability matrix looked like this: 

          a     b      c  
        |----------------|
      a |    |     |     |
        |----------------|
      b |    |     |     |      Then cell x would represent p(b | c)
        |----------------|
      c |    |  x  |     |
        |----------------|

model_type: pass either 'unigram' or 'bigram' for which model you are generating from

max_words: how many words the generator should generate before it automatically stops. The 
generator stops itself once it generates "</s>," but this leads to extremely long "sentences"

start_word: only used in the bigram model, the word it starts generating from. I'd suggest using "<s>". 


'''


def GENERATE(word_index_dict, probs, model_type, max_words, start_word):
    returnSTR = ""
    index_word_dict = {v: k for k, v in word_index_dict.items()}
    num_words = 0

    #been passed a list of probabilities
    if model_type == "unigram":

        #using https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
        while(True):
            wordIndex = np.random.choice(len(word_index_dict), 1, p=list(probs))
            word = index_word_dict[wordIndex[0]]
            returnSTR += word + " "
            num_words +=1
            if word == "</s>" or num_words == max_words:
                break

        return returnSTR

    #been passed a matrix of probabilities, where each row is the previous word. 
    if model_type == "bigram":
        returnSTR = start_word + " "
        prevWord = start_word
        while(True):
            wordIndex = np.random.choice(len(word_index_dict), 1, p=list(probs[word_index_dict[prevWord]]))
            word = index_word_dict[wordIndex[0]]
            returnSTR += word + " "
            num_words +=1
            prevWord = word
            if word == "</s>" or num_words == max_words:
                break

        return returnSTR




