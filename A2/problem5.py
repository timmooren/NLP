import codecs
from collections import Counter
from typing import Tuple
"""
NOT WORKING YET IT RETURNS THE WRONG VALUES FOR THE SMOOTHED PROBABILITIES
"""


def get_counts(text_file: str) -> Tuple[Counter, Counter]:
    """
    Get bigram and trigram counts from a text file.

    Args:
        text_file (str): Path to the text file.

    Returns:
        tuple: A tuple containing bigram and trigram counts as Counter objects.
    """
    bigram_counts = Counter()
    trigram_counts = Counter()

    with codecs.open(text_file, "r", "utf-8") as f:
        for line in f:
            words = line.strip().lower().split()
            for i in range(1, len(words) - 1):
                # consider word before
                bigram = (words[i - 1], words[i])
                # consider word before and after
                trigram = (words[i - 1], words[i], words[i + 1])

                bigram_counts[bigram] += 1
                trigram_counts[trigram] += 1

    return bigram_counts, trigram_counts


def trigram_probability(bigram: Tuple[str, str], next_word: str, bigram_counts: Counter, trigram_counts: Counter, alpha: float = None) -> float:
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
    if alpha:
        trigram_count = trigram_counts[trigram] + alpha
        bigram_count = bigram_counts[bigram] + alpha * len(bigram_counts)
    else:
        trigram_count = trigram_counts[trigram]
        bigram_count = bigram_counts[bigram]

    if bigram_count == 0:
        return 0

    # probability of a trigram given bigram and trigram counts
    return trigram_count / bigram_count

def main():
    alpha = 0.1
    text_file = "brown_100.txt"
    bigram_counts, trigram_counts = get_counts(text_file)
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
        probability = trigram_probability(bigram, next_word, bigram_counts, trigram_counts)
        print(f"p({next_word} | {bigram}): {probability}")

    print("\nSmoothed Trigram Probabilities:")
    for bigram, next_word in trigrams_to_check:
        probability = trigram_probability(bigram, next_word, bigram_counts, trigram_counts, alpha=alpha)
        print(f"p({next_word} | {bigram}): {probability}")


if __name__ == "__main__":
    main()