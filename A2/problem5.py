from collections import Counter
from typing import Tuple


def count_bigrams(corpus_file_path):
    bigram_counts = Counter()

    with open(corpus_file_path) as f:
        for line in f:
            words = line.lower().strip().split()[1:]
            bigrams = zip(words[:-1], words[1:])
            bigram_counts.update(bigrams)

    return bigram_counts


def count_trigrams(corpus_file_path):
    trigram_counts = Counter()

    with open(corpus_file_path) as f:
        for line in f:
            words = line.lower().strip().split()[1:]
            trigrams = zip(words[:-2], words[1:-1], words[2:])
            trigram_counts.update(trigrams)

    return trigram_counts


def trigram_probability(bigram: Tuple[str, str], next_word: str, bigram_counts: Counter, trigram_counts: Counter, alpha: float = 0) -> float:
    """
    Calculate the probability of a trigram given bigram and trigram counts.

    Args:
        bigram (tuple): A tuple containing the first two words of the trigram.
        next_word (str): The third word of the trigram.
        bigram_counts (Counter): A Counter object containing bigram counts.
        trigram_counts (Counter): A Counter object containing trigram counts.
        alpha (float, optional): Smoothing parameter, default is None.

    Returns:
        float: Probability of the trigram.
    """
    trigram = bigram + (next_word.lower(),)

    # add smoothing if alpha is provided
    trigram_count = trigram_counts[trigram] + alpha
    bigram_count = bigram_counts[bigram] + alpha * 813

    if bigram_count == 0:
        return 0

    # probability of a trigram given bigram and trigram counts
    return trigram_count / bigram_count


def main():
    alpha = 0.1
    text_file = "brown_100.txt"
    bigram_counts = count_bigrams(text_file)
    trigram_counts = count_trigrams(text_file)
    trigrams_to_check = [
        (("in", "the"), "past"),
        (("in", "the"), "time"),
        (("the", "jury"), "said"),
        (("the", "jury"), "recommended"),
        (("jury", "said"), "that"),
        (("agriculture", "teacher"), ","),
    ]

    print("Unsmoothed Trigram Probabilities:")
    for bigram, next_word in trigrams_to_check:

        probability = trigram_probability(
            bigram, next_word, bigram_counts, trigram_counts)
        print(f"p({next_word} | {bigram}): {probability}")

    print("\nSmoothed Trigram Probabilities:")
    for bigram, next_word in trigrams_to_check:
        probability = trigram_probability(
            bigram, next_word, bigram_counts, trigram_counts, alpha=alpha)
        print(f"p({next_word} | {bigram}): {probability}")


if __name__ == "__main__":
    main()