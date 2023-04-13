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


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary

f = codecs.open("brown_100.txt")


counts = #TODO: initialize numpy 0s array


#TODO: iterate through file and update counts

#TODO: normalize counts


#TODO: writeout bigram probabilities



f.close()