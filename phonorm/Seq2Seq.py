## This file contains the Encoder, Decoder and the Seq2Seq implementation

from keras import Input, Model
from keras.models import save_model, load_model
from keras.layers import Dense, LSTM, Bidirectional, Dot, Concatenate
from keras.optimizers import Adam
import pickle

from phonorm.utilities import one_hot_encode, decode_from_ohe
from phonorm.evaluate import plot_model_history, decode_sequence, evaluate_bleu

## Seq2seq setup
class Seq2Seq:

    '''
    This is the encoder/decoder model we are using for phonorm.

    It is based on: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    Also: see Chollet, Francois. Deep learning with python. Manning Publications Co., 2017.
    '''
    
    def __init__(self, hidden_dim, mapping_input, mapping_output):
        
        '''
        :param hidden_dim: number of hidden units
        :param mapping_input: charmap object containing mapping and inverse mapping for the input words
        :param mapping_output: charmap object containing mapping and inverse mapping for the output words
        '''
        
        self.mapping_input = mapping_input
        self.mapping_output = mapping_output
        
        self.hidden_dim = hidden_dim
        self.model = None
        self.encoder_model = None
        
        ## Define concatenator
        self.concat = Concatenate()
        ## This is shared by the layers so we might as well define it here
        
    def Encoder(self, vocab_length, dropout_prop = 0.2, recurrent_dropout_prop = 0.2):

        '''
        Encoder model

        :param vocab_length: length of the input charmap
        :param dropout_prop: probability of masking inputs
        :param recurrent_dropout_prop: probability of masking connections between recurrent units
        '''
        
        ## Set data
        self.encoder_vocab_length = vocab_length
        
        # Specify input
        self.encoder_inputs = Input(shape=(None, vocab_length))

        ## Specify the encoder
        encoder = Bidirectional(LSTM(self.hidden_dim, activation = "tanh", return_state = True, 
                                     dropout = dropout_prop, recurrent_dropout = recurrent_dropout_prop))

        ## Get outputs
        encoder_outputs, forward_hidden, forward_memcell, backward_hidden, backward_memcell = encoder(self.encoder_inputs)

        ## Concatename the forward & backward hidden cells
        self.state_hidden = self.concat([forward_hidden, backward_hidden])
        self.state_memcell = self.concat([forward_memcell, backward_memcell])
        
    def Decoder(self, vocab_length, dropout_prop = 0.2, recurrent_dropout_prop = 0.2):

        '''
        Decoder model

        :param vocab_length: number of characters in the output charmap
        :param dropout_prop: probability of masking inputs
        :param recurrent_dropout_prop: probability of masking connections between recurrent units
        :return:
        '''
        
        ## Set data
        self.decoder_vocab_length = vocab_length
        
        ## Use encoder states as the initial states as the initial states
        self.decoder_inputs = Input(shape = (None, vocab_length))

        ## Encoder LSTM is bidirectional so we need to multiply this
        self.decoder_lstm = LSTM(self.hidden_dim * 2, return_sequences=True, return_state=True,
                            dropout = dropout_prop, recurrent_dropout = recurrent_dropout_prop)

        ## Save the outputs
        decoder_outputs = self.decoder_lstm(self.decoder_inputs, initial_state = [self.state_hidden, self.state_memcell])

        ## Discard elements 2 and 3 using underscore
        self.decoder_outputs, _, _ = decoder_outputs

        ## Propagate through densely connected layer
        self.decoder_dense = Dense(vocab_length, activation = "softmax")

        ## Save outputs
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        
    def compile_model(self,
                      optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss = "categorical_crossentropy",
                      print_summary = True):

        '''
        Compile the Keras encoder/decoder model

        :param optimizer: Optimizer to use for gradient descent. Defaults to Adam. See https://keras.io/optimizers/
        :param loss: Loss to optimize. Defaults to crossentropy loss
        :param print_summary: Whether to print a summary of the model. Defaults to True
        '''
        
        ## Define the model
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        
        ## Compile
        self.model.compile(optimizer = optimizer,
                           loss = loss)
        
        ## Print summary
        if print_summary:
            self.model.summary()
            
    def fit(self, data_in, data_out, batch_size = 64, epochs = 10, validation_split = 0.05,
            plot_loss = True):
        
        self.fit_opts = {
            "batch_size" : batch_size,
            "epochs" : epochs,
            "validation_split" : validation_split,
            "hidden_dim" : self.hidden_dim
        }
        
        '''
        @param data_in list containing ecoder inputs & decoder inputs
        @param data_out one-hot encoded outputs for decoder
        '''
        
        ## If none, raise error
        if self.model == None:
            
            raise ValueError("You must compile the model before calling 'fit'")
        
        # Fit model and plot
        history = self.model.fit(data_in, data_out,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split)
        
        ## Save history
        self.history = history.history
        
        ## Plot historical loss
        if plot_loss:
            plot_model_history(self.history)
        
    def plot_model_history(self):
        
        '''Plot history of loss'''
        
        plot_model_history(self.history)
        
    def inference(self):
        
        '''
        Inference setup

        See: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        '''
        
        # Define the model
        encoder_states = [self.state_hidden, self.state_memcell]
        self.encoder_model = Model(self.encoder_inputs, encoder_states)

        # Get decoder states
        decoder_state_input_h = Input(shape=(self.hidden_dim * 2,))
        decoder_state_input_c = Input(shape=(self.hidden_dim * 2,))

        # Feed to the decoder
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # Outputs to the decoder
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        
        # Decoder states
        decoder_states = [state_h, state_c]

        # Densor for the outputs
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Model
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        
    def predict(self, word):

        '''
        Predict the pronunciation of an input word

        :param word: word to predict
        :return: pronunciation of input word
        '''
        
        if self.encoder_model is None:
            ## Inference setup if not exists
            self.inference()
        
        # One-hot encode the word
        word_ohe = one_hot_encode([word], self.mapping_input)
        
        # Shape should be (1, 33, 34)
        #print(word_ohe.shape)
        
        # Predict output
        return(decode_sequence(word_ohe, self.encoder_model, self.decoder_model, self.mapping_input, self.mapping_output))
    
    def save(self, pathname = "models/model.h5"):

        '''
        Save the trained model to disk

        :param pathname: path to store model. Defaults to 'models/model.h5'
        '''
        
        '''Save the model'''
        
        self.model.save(pathname)
        
        ## Save settings as json
        mappings_out_name = pathname.strip(".h5") + "_mappings.p"
        fitopts_out_name = pathname.strip(".h5") + "_fit_opts.p"
        mhist_out_name = pathname.strip(".h5") + "_history.p"
        
        ## Concat mappings
        mappings = [self.mapping_input, self.mapping_output]
        
        ## Save mappings as pickle files
        with open(mappings_out_name, "wb") as outFile:
            pickle.dump(mappings, outFile, protocol=pickle.HIGHEST_PROTOCOL)
            
        ## Save fit options
        with open(fitopts_out_name, "wb") as outFile:
            pickle.dump(self.fit_opts, outFile,  protocol=pickle.HIGHEST_PROTOCOL)
            
        ## Save history
        with open(mhist_out_name, "wb") as outFile:
            pickle.dump(self.history, outFile, protocol = pickle.HIGHEST_PROTOCOL)
            
    def load(self, pathname = "models/model.h5"):

        '''
        Load a model saved on disk

        :param pathname: path where model is stored
        '''
        
        self.model = load_model(pathname)
        
        ## Save settings as json
        mappings_in_name = pathname.strip(".h5") + "_mappings.p"
        fitopts_in_name = pathname.strip(".h5") + "_fit_opts.p"
        mhist_in_name = pathname.strip(".h5") + "_history.p"
        
        ## Retrieve mappings
        with open(mappings_in_name, "rb") as inFile:
            mappings = pickle.load(inFile)
            
        self.mapping_input = mappings[0]
        self.mapping_output = mappings[1]
        
        ## Retrieve fit options
        with open(fitopts_in_name, "rb") as inFile:
            self.fit_opts = pickle.load(inFile)
            
        ## Retrieve history
        with open(mhist_in_name, "rb") as inFile:
            self.history = pickle.load(inFile)
            
        ## Set up inference
        ## TODO: adapted from ???
        
        ## Load inputs & states
        encoder_inputs = self.model.input[0]   
        encoder_outputs, forward_hidden, forward_memcell, backward_hidden, backward_memcell = self.model.layers[1].output  

        ## Concatenate
        concat = Concatenate()
        ## Concatename the forward & backward hidden cells
        state_hidden = concat([forward_hidden, backward_hidden])
        state_memcell = concat([forward_memcell, backward_memcell])
        encoder_states = [state_hidden, state_memcell]
        
        # Encoder model
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        ## Decoder inputs
        decoder_inputs = self.model.input[1]

        ## Create inputs
        decoder_state_input_h = Input(shape=(self.fit_opts['hidden_dim'] * 2,),name='input_3')
        decoder_state_input_c = Input(shape=(self.fit_opts['hidden_dim'] * 2,),name='input_4')

        ## Save states for decoder and define model
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[4]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        
        ## Propagate through densor
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[5]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)




