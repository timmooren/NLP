import nltk
from nltk.corpus import brown
from collections import Counter
from math import log2
from tabulate import tabulate


def calculate_pmi(corpus, min_freq=10):
    """
    Calculate the pointwise mutual information (PMI) for all successive pairs of words in the given corpus.

    Args:
        corpus (nltk.corpus.reader): The input corpus.
        min_freq (int, optional): Minimum frequency for words to be considered. Defaults to 10.

    Returns:
        dict: A dictionary of word pairs (bigrams) and their corresponding PMI values.
    """
    words = [word.lower() for word in corpus.words()]
    bigrams = nltk.bigrams(words)

    # absolute frequencies
    word_freq = Counter(words)
    bigram_freq = Counter(bigrams)

    # N is size of corpus
    N = len(words)

    pmi_dict = {}
    # iterate over the word pairs and calculate PMI
    for (w1, w2), bigram_count in bigram_freq.items():
        w1_count, w2_count = word_freq[w1], word_freq[w2]

        # words that hat occur in the corpus less than min_freq times are ignored
        if w1_count >= min_freq and w2_count >= min_freq:
            # calculate probabilities
            p_w1, p_w2 = w1_count / N, w2_count / N
            p_bigram = bigram_count / (N - 1)
            # calculate PMI
            pmi = log2(p_bigram / (p_w1 * p_w2))
            pmi_dict[(w1, w2)] = pmi

    return pmi_dict


def top_n_pairs(pmi_dict, n, highest=True):
    """
    Find the top n pairs with the highest or lowest PMI values in a PMI dictionary.

    Args:
        pmi_dict (dict): A dictionary of word pairs and their corresponding PMI values.
        n (int): The number of pairs to return.
        highest (bool, optional): Whether to return the highest (True) or lowest (False) PMI values. Defaults to True.

    Returns:
        list: A list of the top n word pairs with the highest or lowest PMI values.
    """
    sorted_pairs = sorted(
        pmi_dict.items(), key=lambda x: x[1], reverse=highest)
    return sorted_pairs[:n]


corpus = brown
pmi_dict = calculate_pmi(corpus)

top_20_highest_pmi = top_n_pairs(pmi_dict, 20, highest=True)
top_20_lowest_pmi = top_n_pairs(pmi_dict, 20, highest=False)

print("Top 20 word pairs with the highest PMI values:")
for pair, pmi in top_20_highest_pmi:
    print(f"{pair}: {pmi:.4f}")
table_highest = [[pair[0], pair[1], f"{pmi:.4f}"] for pair, pmi in top_20_highest_pmi]
print(tabulate(table_highest, headers=['Word 1', 'Word 2', 'PMI'], tablefmt='latex_booktabs'))


print("\nTop 20 word pairs with the lowest PMI values:")
for pair, pmi in top_20_lowest_pmi:
    print(f"{pair}: {pmi:.4f}")

table_lowest = [[pair[0], pair[1], f"{pmi:.4f}"] for pair, pmi in top_20_lowest_pmi]
print(tabulate(table_lowest, headers=['Word 1', 'Word 2', 'PMI'], tablefmt='latex_booktabs'))
