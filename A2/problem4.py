#!/usr/bin/env python3

from sklearn.preprocessing import normalize
from generate import GENERATE
from problem1 import load_word_index_dict
from problem6 import compute_perplexity
from problem3 import get_counts, write_bigram_probs


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

    # assignment 6 evaluate toy corpus
    compute_perplexity("toy_corpus.txt", "smoothed_bigram_eval.txt", probs, word_index_dict, "bigram")

    # Generate 10 sentences using smoothed bigram model
    with open("smoothed_generation.txt", "w") as f:
        for i in range(10):
            generated_sentence = GENERATE(word_index_dict, probs, "bigram", 50, "<s>")
            f.write(generated_sentence + "\n")



if __name__ == "__main__":
    main()
