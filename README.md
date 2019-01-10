# Phonorm

[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org) [![lifecycle](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://www.tidyverse.org/lifecycle/#stable)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**phonorm** is an exploratory project in which we apply a Recurrent Neural Network to the problem of [phonetic normalization](http://mlwiki.org/index.php/Phonetic_Normalization). The need for such a model arose from the type of conversations we observed in our chatbot [ChitChat](https://bitbucket.org/arvid/chitchat), as we observed a lot of text that is written much like it is spoken. Current phonetic algorithms, such as [Soundex](https://en.wikipedia.org/wiki/Soundex) are too aggressive and do not work well in our use case.

You can find our writeup of the project [here](https://github.com/JasperHG90/phonorm/docs/writeup/phonorm_writeup.pdf). Comments are welcome and can either be left in the issues section or can be sent to jasperginn[at]gmail.com

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
  - Contains pre-trained models
+-- phonorm
  - Contains utilities and code for modeling
+-- preprocessing
  - Contains utilities and code for preprocessing data
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## A note on training the model

If you want to retrain the model using the data in this repository, be aware that training will be **slow** on CPUs.

## Setting up

At a minimum, you need a python 3 installation. However, it would be best to use [Anaconda](https://www.anaconda.com/). The steps below assume that you are using anaconda for this project.

1. Create a new environment called 'phonorm'

```shell
conda create -n phonorm python=3.6 anaconda
```

2. Activate the environment

```shell
source activate phonorm  
```

on Windows:

```shell
conda activate phonorm
```

3. Install dependencies

```shell
conda install --yes --file requirements.txt
```

4. Install 'pywiktionary' from git

5. (optional) install `tensorflow-gpu` if you are using a GPU





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
