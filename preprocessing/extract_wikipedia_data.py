# Import pywiktionary

#from pywiktionary import Wiktionary
import numpy as np
import urllib.request
import os
from preprocessing.utils import cv_splits, to_pairs, filter_homophone
import re
from numpy.random import shuffle
import math
from collections import Counter # for word frequencies
from pywiktionary import Wiktionary

#%% ------ Preprocessing functions

def preprocess_data(data, language, max_word_length=False, one_per_observation=False):
    '''
    Take wikt2pron output (list containing pronunciations in English), and:
     - Remove 'IPA not present' / 'Language not found'
     - Turn the data into a rectangular format

    @param data result of wikit2pron
    @param language language for which to filter (ex. 'en')
    @param max_word_length remove tail ends from word lengths; defaults to False; calculates the sd of word lengths and removes mean+-2*sd
    @param one_per_observation every element in the wikt2pron output can contain multiple pronunciations (e.g. for different dialects). It is not specified for which dialect these occur. If True, this parameter takes on the first observation of the list of possible values.

    @return pandas data frame containing n * k_n rows and 5 columns.
     tuple containing :
        1. the result (list containing a dict per one word-pronunciation mapping)
        2. number of languages found in the master dataset
    '''

    # Filter results where pronunciation is a string --> these are 'not found' messages
    data_filter = [co for co in data if not isinstance(co["pronunciation"], str)]

    # If maximum length, calculate upper bound
    if max_word_length:
        wl = [len(df['title']) for df in data_filter]
        upper_bound = np.mean(wl) + 1 * np.sqrt(np.var(wl))

        # Subset data
        data_filter = [entry for entry in data_filter if len(entry['title']) <= upper_bound]

    # Unlist each pronunciation result s.t. it is associated with a word and unique id
    data_flattened = [flatten_word_list(df, one_per_observation=one_per_observation) for df in data_filter]

    # Flatten the list of lists that is the output of col_decomp
    res = [word_list for entry in data_flattened for word_list in entry]

    # Retrieve languages found in results
    langs = set([elem["lang"] for elem in res])

    # Filter for reference language
    res_lang_filter = [elem for elem in res if elem["lang"] == language]

    # If length == 0, then raise error
    if len(res_lang_filter) == 0:
        raise phonormException("Language {} not present in data".format(language))

    # Return
    return ((res, langs))

def flatten_word_list(word_list, one_per_observation=False):
    '''
    Unnest the wikt2pron data

    @param word_list one element of original wikt2pron dataset
    @param one_per_observation see same parameter under 'preprocess_data()' function

    @return: list containing dictionary entry for each word-pronunciation mapping found
    '''

    # Save word name + id
    _id = word_list["id"]
    word = word_list["title"].lower()

    # Check length of pronunciation
    # If one per observation, then only return first one
    if len(word_list["pronunciation"]) == 1 or one_per_observation:

        lang = word_list["pronunciation"][0]["lang"]
        IPA = word_list["pronunciation"][0]["IPA"]
        X_SAMPA = "\t" + word_list["pronunciation"][0]["X-SAMPA"] + "\n"

        # Create dict
        res = {
            "_id": _id,
            "word": word,
            "lang": lang,
            "IPA": IPA,
            "X_SAMPA": X_SAMPA
        }

        # Return
        return ([res])

    # Else multiple entries
    else:

        # Open results
        res = [None] * len(word_list["pronunciation"])

        # Unroll dict
        max_len = len(word_list["pronunciation"])

        # For each element, ...
        for elem_iter in range(0, max_len):
            # Subset
            elem = word_list["pronunciation"][elem_iter]

            # Compile new dict
            res_current = {
                "_id": _id,
                "word": word,
                "lang": elem["lang"],
                "IPA": elem["IPA"],
                "X_SAMPA": "\t" + elem["X-SAMPA"] + "\n"
            }

            # Save to res
            res[elem_iter] = res_current

        # Return
        return (res)

#%%

'''
Download wikipedia dump file and preprocess if it does not already exist in the data folder

See general url for wiki dumps: https://dumps.wikimedia.org/enwiktionary/

'''

# If not exists, download
if "wikt2pron.npy" in os.listdir("data/raw"):

    pron = np.load("data/raw/wikt2pron.npy")

elif not "enwiktionary-20181101-pages-meta-current.xml.bz2" in os.listdir("data/raw"):

    print("Downloading wikipedia data. This takes approximately 10 minutes")

    # Point to wikipedia dump file
    file = "https://dumps.wikimedia.org/enwiktionary/20181101/enwiktionary-20181101-pages-meta-current.xml.bz2"

    # Download
    urllib.request.urlretrieve(file, "data/raw/enwiktionary-20181101-pages-meta-current.xml.bz2")

    # Set file path
    fp = "data/raw/enwiktionary-20181101-pages-meta-current.xml.bz2"

    '''Extract english words + pronunciation. This takes a LONG time'''

    # Extract only English words
    wikt = Wiktionary(lang="English", XSAMPA=True)

    # Save dict to python object
    pron = wikt.extract_IPA(fp)

    np.save("data/raw/wikt2pron.npy", pron)

#%% ------ Preprocess data and save to file

# Filter for this language
language = "en"
# Words beyond this length are cut
max_word_length = False
# Only one word per observation (not multiple for different dialects, e.g. american v. british english)
one_per_example = True

# Prep datset
prep = preprocess_data(pron, language,
                       max_word_length=max_word_length,
                       one_per_observation=one_per_example)

# Take words with special characters and remove these
prep = np.array(prep[0])
pattern = '[^a-zA-Z]'

# Filter
mask = np.array([bool(re.search(pattern, item['word'])) for item in prep])
prep_masked = prep[mask == False]

# To from / to pairs and save
import random

# To numpy array
pairs = to_pairs(prep_masked)
pairs = np.array(pairs)

# Check a random subset of the data
dl = np.arange(0, len(pairs)-1)
shuffle(dl)
for i in dl[:100]:
    print('[%s] => [%s]' % (pairs[i,0], pairs[i,1]))

# Save
np.save('data/raw/wikt2pron.npy', pairs)

#%% Make train/dev/test splits

pairs = np.load('data/raw/wikt2pron.npy')

for pair in pairs:
    pair[1] = pair[1].replace("<SOS>", "\t").replace("<EOS>", "\n")

#%%

import random

# Get pairs with equal output value
words = [pair[1] for pair in pairs]

# Make a counter for frequencies
freqs = Counter(words)

# If value for a word > 1, append to the homophone list
hmphones = []
for key in freqs:
	if freqs[key] > 1:
		hmphones.append(key)

# Proportion used for test/dev
devobs = math.floor(0.1 * len(hmphones))
testobs = math.floor(0.1 * len(hmphones))

# Shuffle
random.seed(297)
shuffle(hmphones)

# Filter homophones and split to dev/test
trn, dev, tst = filter_homophone(pairs, hmphones[:testobs], hmphones[testobs:(devobs+testobs)])

# Check
if not len(pairs) == (len(trn) + len(dev) + len(tst)):
	print("Warning. The length of the original data does not match the length of the individual splits")

#%%

# Save
np.save("data/preprocessed/wikt2pron_homophone_dev.npy", dev)
np.save("data/preprocessed/wikt2pron_homophone_tst.npy", tst)

# Make normal train/test split

# Split data
train, dev, test = cv_splits(trn, 300, 600, 500)

# Save
np.save("data/preprocessed/wikt2pron_dev.npy", dev)
np.save("data/preprocessed/wikt2pron_test.npy", test)
np.save("data/preprocessed/wikt2pron_train.npy", train)

