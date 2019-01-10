# Phonorm

[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org) [![lifecycle](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://www.tidyverse.org/lifecycle/#stable)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the following files

```text
+-- data
  | +-- extra
      - contains wikipedia dataset with commonly misspelled words
  | +-- preprocessed
      - contains preprocessed datasets
  | +-- raw
      - contains raw data (not preprocessed)
+-- docs
  - Contains presentation and writeup
+-- modeling
  - Contains Jupyter notebooks used for modeling
+-- models
  - Contains saved models
+-- phonorm
  - Contains utilities and code for modeling
+-- preprocessing
  - Contains utilities and code for preprocessing data
+-- .gitignore
+-- README.md
+-- requirements.txt
```

# CFI phonetic normalization

This repository contains all necessary documents/scripts etc. for the 'phonetic normalization' project at the Centre for Innovation @ Leiden University.

## Project documents

You can find:

- The [datasets overview](https://docs.google.com/document/d/1t7gr5FqlqpbCCpqlIAHmqN2tkgVAWNp3kDaN6UhLZ7g/edit?usp=sharing) here
- The [project overview](https://docs.google.com/document/d/1WoyhyAnES8HbDhgiUsLFy0G7z_eu-yCfJwfwD02-LQg/edit?usp=sharing) here
- The [research document](https://docs.google.com/document/d/1gaI7TUZnQYrR1PI8Cg_ejRpm5EgsXa1b2uunEhhNNKI/edit?usp=sharing) here

## Basic information

It would help to read up on [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) and [Extended Speech Assessment Methods Phonetic Alphabet](https://en.wikipedia.org/wiki/X-SAMPA)

Also check out the [phonetic alphabet for English dialects](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet_chart_for_English_dialects)

## Folders in this project

- **code**: contains python code to extract phonetic data from wikipedia dumps & to train NN on resulting data
- **readings**: academic articles relevant to the project
- **img**: images, plots etc. relevant to the project
- **data**: contains dataset used for this project
- **env**: dockerfiles used to (re)-create the project environment
- **phonorm**: python module containing reproducible code and examples for the phonetic normalizer

## Prerequisites

- Python 3
- [Wikt2Pron](https://github.com/abuccts/wikt2pron) module (used to extract phonetic data from wikipedia)
- Jupyter (to run jupyter notebooks)
- [Wikipedia dump](https://dumps.wikimedia.org/enwiktionary/) (extracted)

## Further reading

Additional notes, comments and descriptions can be found in the Jupyter notebooks or README files in the individual folders.
