## Functions for evaluation of model performance

## TODO: references to source for each function where applicable
## TODO: write documentation for each function

# Modules
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import numpy as np

def decode_sequence(input_seq, encoder_model, decoder_model, mapping_input, mapping_output):

    '''
    Take input as one-hot encoded vector and predict the output.

    :param input_seq: one-hot encoded input word
    :param encoder_model: trained model encoder (see 'inference' in seq2seq.py)
    :param decoder_model: trained model decoder
    :param mapping_input: hash tables from character --> integer and vice versa
    :param mapping_output: hash tables from character --> integer and vice versa
    :return: predicted pronunciation

    :adapted from: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    '''
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, mapping_output.n_chars))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, mapping_output.char2index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = mapping_output.index2char[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > mapping_output.max_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, mapping_output.n_chars))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip("\n")

def evaluate_bleu(reference, prediction):

    '''
    Calculate the 1, 2, 3, and 4-gram BLEU score

    :param reference: actual output
    :param prediction: predicted output
    :return: numpy array containing 4 rows
    '''
    
    ## Numpy array
    out = np.zeros(
        (4, 1),
        dtype="float32"
    )
    
    ## For each
    out[0] = sentence_bleu([list(reference)], list(prediction), weights=(1, 0,0,0))
    out[1] = sentence_bleu([list(reference)], list(prediction), weights=(0.5, 0.5,0,0))
    out[2] = sentence_bleu([list(reference)], list(prediction), weights=(0.33, 0.33,0.33,0))
    out[3] = sentence_bleu([list(reference)], list(prediction), weights=(0.25, 0.25,0.25,0.25))
    
    # Return
    return out

## Plot bleu score function
def plot_bleu(data):

    '''
    Plot the bleu score for a given n-gram

    :param data: list of bleu scores
    :return: matplotlib histogram
    '''

    # %matplotlib inline
    plt.hist(data, normed=True, bins=15)
    plt.ylabel('BLEU score')
    plt.show()

def plot_model_history(history):
    
    '''
    Given Keras model history as input, show a plot with train / validation split loss

    :param history: keras train history saved from <model_name>.history
    :return: matplotlib plot with epochs on x-axis, train and dev loss on the y-axis

    :from: Chollet, Francois. Deep learning with python. Manning Publications Co., 2017.
    '''
    
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

