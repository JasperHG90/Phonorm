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

If you want to retrain the model using the data in this repository, be aware that training will be **slow** on CPUs. You should consider using a GPU.

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

4. (optional) Install 'pywiktionary' from git

```shell
pip install git+https://github.com/abuccts/wikt2pron.git
```

5. (optional) install `tensorflow-gpu` if you are using a GPU

```shell
conda install tensorflow-gpu
```

At this point, your environment ready to be used.

## Using phonorm

If you want to train your own models, you should check out the [modeling](https://github.com/JasperHG90/phonorm/tree/master/modeling) folder for examples.

If you want to use the pre-trained models, please see the [examples](https://github.com/JasperHG90/phonorm/tree/master/examples/) folder.
