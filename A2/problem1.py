#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""


def load_word_index_dict(file_path):
    word_index_dict = {}
    # read brown_vocab_100.txt into word_index_dict
    with open(file_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
            word = line.rstrip().lower()
            word_index_dict[word] = idx
    return word_index_dict


def main():
    word_index_dict = load_word_index_dict("brown_vocab_100.txt")
    # write word_index_dict to word_to_index_100.txt
    with open("word_to_index_100.txt", "w") as f:
        f.write(str(word_index_dict))

    print(word_index_dict['all'])
    print(word_index_dict['resolution'])
    print(len(word_index_dict))


if __name__ == "__main__":
    main()
