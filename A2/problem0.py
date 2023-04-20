"""
To access the full Brown corpus, import it from NLTK. Compute a list of unique words sorted by
descending frequency for (i) the whole corpus and (ii) two different genres of your choice.
Before moving on, extract the following information (should be visible in your code and output files):
number of tokens; number of types; number of words; average number of words per sentence;
average word length. You should also run a default part-of-speech tagger on the dataset and identify
the ten most frequent POS tags.
Next, Use the Python library matplotlib to plot the frequency curves for the corpus and two
genres you choose: i.e. x-axis is position in the frequency list, y-axis is frequency. Provide both a plot
with linear axes and one with log-log axes
"""
import nltk
from nltk.corpus import brown
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np


def plot_frequency_curve(freq_dist, color, label, loglog=False):
    freqs = sorted(freq_dist.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))

    if loglog:
        plt.loglog(ranks, freqs, linestyle='-',
                   marker='o', color=color, label=label)
        plt.xlabel("Log(rank)")
        plt.ylabel("Log(frequency)")
    else:
        plt.plot(ranks, freqs, linestyle='-',
                 marker='o', color=color, label=label)
        plt.xlabel("Rank")
        plt.ylabel("Frequency")
    
    plt.title("Frequency Curve")
    plt.legend()
    # save
    plt.savefig(f"plots/frequency_curve_loglog={loglog}.png")


def analyze_corpus(category=None):
    name = category if category else 'corpus'
    name2color = {
        'corpus': 'blue',
        'news': 'red',
        'romance': 'green'
    }

    corpus = brown.words(categories=category)
    # only keep words that contain at least one letter
    words = [word for word in corpus if any(c.isalpha() for c in word)]
    # number of tokens
    num_tokens = len(words)
    # number of types
    num_types = len(set(words))
    # number of words
    num_words = sum(len(word) for word in words)

    # Create frequency distributions
    corpus_fd = FreqDist(w.lower() for w in words)
    # Compute a list of unique words sorted by descending frequency
    most_common = corpus_fd.most_common(10)

    # plot frequency curve
    plot_frequency_curve(corpus_fd, color=name2color[name], label=name)
    plot_frequency_curve(
        corpus_fd, color=name2color[name], label=name, loglog=True)

    # average number of words per sentence
    sents = brown.sents(categories=category)

    num_sents = len(sents)
    avg_words_per_sent = num_tokens / num_sents
    avg_word_length = sum(len(word) for word in words) / num_tokens

    pos_tags = nltk.pos_tag(words)
    pos_tags_fd = FreqDist(tag for word, tag in pos_tags)
    most_common_pos = pos_tags_fd.most_common(10)

    print(f"{name.capitalize()} Data:")
    print(f"Most common words: {most_common}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of types: {num_types}")
    print(f"Number of words: {num_words}")
    print(f"Number of sentences: {num_sents}")
    print(f"Average number of words per sentence: {avg_words_per_sent:.2f}")
    print(f"Average word length: {avg_word_length:.2f}")
    print(f"Ten most frequent POS tags: {most_common_pos}\n")


def main():
    nltk.download('brown')
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")

    genres = [None, 'news', 'romance']

    for genre in genres:
        analyze_corpus(genre)

if __name__ == '__main__':
    main()
