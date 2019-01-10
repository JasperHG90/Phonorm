# Preprocess wikipedia misspelled words
# Jasper Ginn
# 28/12/18

# -------

'''
Import the misspelled words data, select 100 randomly chosen words and label these as follows:

1. If the mispelled word is phonetically similar to the word that was intended, label with 1
2. Else, label as 0
'''

import re

msw = []
# Set pattern to search for
pattern = '[^a-zA-Z]'
# File
file = "data/extra/wikipedia_misspelled_words.txt"
with open(file) as inFile:
    for line in inFile:
        # Split words
        inWord = line.split("\\t")[0].lower()
        outWord = line.split("\\t")[1].replace("\n", "").lower()
        # Remove words with special characters
        if bool(re.search(pattern, inWord)) or bool(re.search(pattern, outWord)):
            continue
        msw.append([inWord, outWord])

#%%

# Shuffle words
from random import Random

Random(9065).shuffle(msw)

# Select 100 and save to file
pairs = msw[:100]

#%%

# Should sound the same or not?
sound = [1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
         0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
         1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
         0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
         0, 0, 0, 1, 0, 1, 0, 1, 0, 0]

so = sum(sound)

#%%

final = []
# To one ds
for pair, sound in zip(pairs, sound):
    final.append([pair, sound])

print(str(so) + " marked as 'should be pronounced the same'")

#%%

# Save
import numpy as np
np.save("data/preprocessed/wikipedia_misspelled.npy", final)