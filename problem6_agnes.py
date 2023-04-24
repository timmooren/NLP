import numpy as np
from generate import GENERATE


def compute_perplexity(file_input, file_output, probs, word_index_dict, model="unigram"):

    with open(file_input, "r") as f:
        toy_corpus = f.readlines()

        for sent in toy_corpus:

            # reset sent_prob
            sent_prob = 1

            # split the sentence into a list of words
            sent = sent.strip().lower().split()
            sent_len = len(sent)
            if model == "unigram":
                # iterate through each word in the sentence
                for word in sent:

                    # multiply the probabilities of each word in the sentence
                    sent_prob *= probs[word_index_dict[word]]
                perplexity = 1 / (pow(sent_prob, 1.0 / sent_len))

            # bigram has different prob with 2x2 matrix
            elif model == "bigram":

                # iterate through each word in the sentence
                for i in range(len(sent) - 1):

                    # multiply the probabilities of each word in the sentence
                    sent_prob *= probs[word_index_dict[sent[i]],
                                       word_index_dict[sent[i+1]]]

                perplexity = 1 / (pow(sent_prob, 1.0 / (sent_len-1)))

            # trigram has different prob with 3x3 matrix
            elif model == "trigram":
                # iterate through each word in the sentence
                for i in range(len(sent) - 2):

                    # multiply the probabilities of each word in the sentence
                    sent_prob *= probs[word_index_dict[sent[i]],
                                       word_index_dict[sent[i+1]], word_index_dict[sent[i+2]]]

                perplexity = 1 / (pow(sent_prob, 1.0 / (sent_len-2)))

            print(perplexity)

            # write perplexity to unigram_eval.txt
            f = open(file_output, "a")
            f.write(str(perplexity) + "\n")
    return
