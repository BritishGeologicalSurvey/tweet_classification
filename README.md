## Automatic tweet classification for geohazards

This repository contains several techniques that can be used to perform
binary classification with text. In this case we train each classifier
to learn whether a tweet is referring to a hazard-event/phenomenon
(earthquakes, volcanoes, floods, the aurora) or not.

The methods used in this implementation are the following:

-   Convolutional Neural Network
-   Recurrent Neural Network
    -   Baseline RNN
    -   GRU
    -   LSTM

Prerequisites
-------------

-   Python 3

Install dependencies
--------------------

    pip3 install requirements.txt

Training
--------

There are two modes of training for each algorithm:

1.  Training with random word vectors  
2.  Training with pre-trained word2vec vectors

-   Random-word vectors for CNN

<!-- -->

    python train_cnn.py

To train the RNN-LSTM with random word vectors, we have to use the
following flag

    python train_rnn.py --cell_type "LSTM"

using `python train_rnn.py` without a flag will train the baseline RNN.

-   Pre-trained word vectors (word2vec):

<!-- -->

    python train_rnn.py --word2vec 'word2vec_twitter_model.bin'

After the training finishes the top five models with the best accuracy
are saved in a folder with the format `runs/DAY_MMM_DD_HH_MM_SS_YYYY`

Evaluation
----------

For the evaluation we have to use the following flag to point to the
directory that we saved the checkpoints

    python eval.py --checkpoint_dir 'runs/DAY_MMM_DD_HH_MM_SS_YYYY/models'

This function will print the metrics that we use to measure the
performance of the classifier.

    ##  Accuracy: 0.863636 
    ##  F1: 0.8683602771362586 
    ##  Specificity: 0.79
    ##  Precision: 0.8744186046511628 
    ##  Recall: 0.8623853211009175

To use tensorboard

    tensorboard --logdir runs/DAY_MMM_DD_HH_MM_SS_YYYY/summaries
