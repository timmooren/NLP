#!/usr/bin/env python3
import numpy as np
from generate import GENERATE
from problem1 import load_word_index_dict
from problem6 import compute_perplexity

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

    probs = counts / np.sum(counts)
    # write to unigram_probs.txt
    np.savetxt("unigram_probs.txt", probs)

    # evaluate toy corpus for assignment 6
    compute_perplexity("toy_corpus.txt", "unigram_eval.txt", probs, word_index_dict, "unigram")

    # Generate sentences using unigram model
    with open("unigram_generation.txt", "w") as f:
        for i in range(10):
            generated_sentence = GENERATE(word_index_dict, probs, "unigram", 50, None)
            f.write(generated_sentence + "\n")


if __name__ == "__main__":
    main()