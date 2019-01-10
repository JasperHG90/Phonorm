## Preprocess cmudict data

import re
import numpy as np
from numpy.random import shuffle
import math
import random
from preprocessing.utils import cv_splits, to_pairs, filter_homophone
from collections import Counter

#%%

data = list()
with open("data/raw/cmudict/cmudict_SPHINX_40.txt") as lines:
	for line in lines:
		data.append(line.replace("\n", "").split("\t"))

#%%

# Remove first part of the data
data = data[64:]
# Remove last entries
data = data[:(133026 - 64)]

# Remove entries with numbers in them
nums = re.compile('[0-9]')
pairs = np.array([line for line in data if nums.search(line[0]) == None])

# Add SOS and EOS
for pair in pairs:
	pair[1] = "\t " + pair[1] + " \n"

# Lower strings
for pair in pairs:
	pair[0] = pair[0].lower()
	pair[1] = pair[1].lower()

# Check a random subset of the data
dl = np.arange(0, len(pairs)-1)
shuffle(dl)
for i in dl[:100]:
	print('[%s] => [%s]' % (pairs[i,0], pairs[i,1].strip("\t").strip("\n")))

#%%

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
devobs = math.floor(0.02 * len(hmphones))
testobs = math.floor(0.02 * len(hmphones))

# Shuffle
random.seed(297)
shuffle(hmphones)

# Filter homophones and split to dev/test
trn, dev, tst = filter_homophone(pairs, hmphones[:testobs], hmphones[testobs:(devobs+testobs)])

# Check
if not len(pairs) == (len(trn) + len(dev) + len(tst)):
	print("Warning. The length of the original data does not match the length of the individual splits")

# Print number of homophones
print("Dev homophones.: " + str(devobs) + ", Dev obs.: " + str(len(dev)))
print("Test homophones.: " + str(testobs) +  ", Test obs.: " + str(len(tst)))

#%%

# Save
np.save("data/preprocessed/cmudict_multichar_homophone_dev.npy", dev)
np.save("data/preprocessed/cmudict_multichar_homophone_tst.npy", tst)

#%%

# Make normal train/test split

# Split data
train, dev, test = cv_splits(trn, 400, 600, seed=4712)

# Save
np.save("data/preprocessed/cmudict_multichar_dev.npy", dev)
np.save("data/preprocessed/cmudict_multichar_test.npy", test)
np.save("data/preprocessed/cmudict_multichar_train.npy", train)

#%% Paste all characters together for the second dataset

# Join output data
for pair in pairs:
	pair[1] = "\t" + "".join(pair[1].split()) + "\n"

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
devobs = math.floor(0.02 * len(hmphones))
testobs = math.floor(0.02 * len(hmphones))

#%%

# Shuffle
random.seed(297)
shuffle(hmphones)

# Filter homophones and split to dev/test
trn, dev, tst = filter_homophone(pairs, hmphones[:testobs], hmphones[testobs:(devobs+testobs)])

# Check
if not len(pairs) == (len(trn) + len(dev) + len(tst)):
	print("Warning. The length of the original data does not match the length of the individual splits")

# Print number of homophones
print("Dev homophones.: " + str(devobs) + ", Dev obs.: " + str(len(dev)))
print("Test homophones.: " + str(testobs) +  ", Test obs.: " + str(len(tst)))

# Save
np.save("data/preprocessed/cmudict_singlechar_homophone_dev.npy", dev)
np.save("data/preprocessed/cmudict_singlechar_homophone_tst.npy", tst)

# Split data
train, dev, test = cv_splits(trn, 400, 600, seed=4712)

# Save
np.save("data/preprocessed/cmudict_singlechar_dev.npy", dev)
np.save("data/preprocessed/cmudict_singlechar_test.npy", test)
np.save("data/preprocessed/cmudict_singlechar_train.npy", train)