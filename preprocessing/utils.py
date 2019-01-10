#%%

import random
from numpy.random import shuffle
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%% These functions are used for data preprocessing

def to_pairs(doc):

    '''
    split a loaded document into sentences

    :param doc: wiktionary input line
    :return: list of input --> output pairs
    '''

    pairs = [[line['word'], line['X_SAMPA']] for line in doc]
    return pairs

# Train, dev, test splits
def cv_splits(pairs, ndev, ntest, seed):

    '''
    Create cross-validation splits

    :param pairs: input data
    :param ndev: number of observations in dev
    :param ntest: number of observations in test
    :param seed: seed to use for pseudo-random number generator
    :return: (train, dev, test) data as tuple
    '''

    # Define train/dev/test set
    n_train = len(pairs) - ntest - ndev
    n_dev = len(pairs) - ntest
    n_test = len(pairs)
    # random shuffle
    random.seed(seed)
    shuffle(pairs)
    # split into train/test
    return pairs[:n_train], pairs[n_train:n_dev], pairs[n_dev: n_test]

# Filter out homophones from the dataset
def filter_homophone(pairs, dev_pronunciation_words, test_pronunciation_words):

    '''
    Filter homophones from the train data

    :param pairs: input data
    :param dev_pronunciation_words: pronunciations that occur in the dev set
    :param test_pronunciation_words: pronunciations that occur in the test set
    :return: (train, dev, test) splits for homophones
    '''

    # Subset data
    dev = [pair for pair in pairs if pair[1] in dev_pronunciation_words]
    tst = [pair for pair in pairs if pair[1] in test_pronunciation_words]

    # Make word list that should be removed from the data
    remove = dev_pronunciation_words + test_pronunciation_words

    # Subset
    trn = np.array([list(pair) for pair in pairs if pair[1] not in remove])

    # Return
    return ((trn, dev, tst))
