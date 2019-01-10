## Useful functions

# Import
from phonorm.prepare import charmap
import numpy as np

def create_mapping(input_language_name,
                   output_language_name,
                   word_pairs,
                   split = False):

    '''Take word pairs and create input/output dictionaries'''

    # Create dictionaries
    input_mapping = charmap(input_language_name, split=False)
    output_mapping = charmap(output_language_name, split=split)

    # For each word in the word pairs, add to the charmap
    for pair in word_pairs:

        input_mapping.addWord(pair[0])
        output_mapping.addWord(pair[1])

    # Return
    return(input_mapping, output_mapping)

def index_from_word(mapping, word):

    '''Map characters in a word to their integer representation'''

    return [mapping.char2index.get(char, "<UNK>") for char in list(word)]

def tensor_from_word(mapping, word):

    '''Create a tensor from the character mappings'''

    indices = index_from_word(mapping, word)
    # Return
    return indices

def tensor_from_pair(pair, mappings):

    '''Create tensors for a given word pair'''

    input_tensor = tensor_from_word(mappings[0], pair[0])
    output_tensor = tensor_from_word(mappings[1], pair[1])

    ## Add padding
    if len(input_tensor) < mappings[0].max_length:

        input_tensor += [mappings[0].char2index["<UNK>"]] * (mappings[0].max_length - len(input_tensor))

    ## Add padding
    if len(output_tensor) < mappings[1].max_length:

        output_tensor += [mappings[1].char2index["<UNK>"]] * (mappings[1].max_length - len(output_tensor))

    # Return
    return(np.asarray(input_tensor), np.asarray(output_tensor))

def one_hot_encode(data, mapping, one_timestep_ahead = False, split = False):
    
    '''
    Create one-hot encoding 
    
    @param data list of input words of length N
    @param vocab mapping mapping created by create_mapping() function
    @param one_timestep_ahead if True, then the function will create a one-hot vector that is one timestep ahead
    
    @return numpy array of dimensions (N, max_word_length, vocab_length)
    
    @seealso create_mapping() for max_word_length, vocab_length and vocab_char2index arguments
            - max_word_length max length of the sequence (usually longest entry in data)
            - vocab_length number of elements in the vocabulary associated with data
            - vocab_char2index dictionary that maps characters to integer representation
    '''
    
    ## Retrieve values from the mapping
    max_word_length = mapping.max_length
    vocab_length = mapping.n_chars
    
    vocab_char2index = mapping.char2index
    
    ## Empty numpy array of output dimensions
    out_ohe = np.zeros(
        (len(data), max_word_length, vocab_length),
        dtype='float32'
    )
    
    # Populate the numpy array
    for entry_pos, entry in enumerate(data):
        
        ## char_pos is the index value of the character that is being processed.
        ##  e.g. if input is a word of length 4 then the first char_pos is 0 etc.
        
        ## char is the actual character value
        if split:

            for char_pos, char in enumerate(entry.split(" ")):

                ## Convert input character to integer representation
                char_integer_repr = vocab_char2index[char]

                ## Set value to 1
                if one_timestep_ahead:

                    if char_pos > 0:
                        out_ohe[entry_pos, char_pos - 1, char_integer_repr] = 1.

                else:

                    out_ohe[entry_pos, char_pos, char_integer_repr] = 1.

        else:

            for char_pos, char in enumerate(entry):

                ## Convert input character to integer representation
                char_integer_repr = vocab_char2index[char]

                ## Set value to 1
                if one_timestep_ahead:

                    if char_pos > 0:

                        out_ohe[entry_pos, char_pos - 1, char_integer_repr] = 1.

                else:

                    out_ohe[entry_pos, char_pos, char_integer_repr] = 1.
                
    ## Return
    return(out_ohe)

def decode_position(position, mapping):
    
    '''
    Lookup up a character by integer index in the mapping
    
    @param position integer index value. Should exist in mapping.index2char
    @mapping character mapping created by create_mapping()
    
    @return character associated with the position value
    '''
    
    return(mapping.index2char[position])


def decode_from_ohe(input_sample, mapping):
    
    '''
    Returns a decoded one-hot encoded sample 
    
    @input_sample one one-hot encoded sample --> e.g. encoder_in_ohe[0]
    @mapping character mapping created by create_mapping()
    
    @return decoded example
    '''
    
    ## Retrieve index2char mapping
    index2char = mapping.index2char
    
    ## For each input that is not zero, reverse-lookup the index in the dictionary and append
    decoded = []
    for i, array in enumerate(input_sample):
        
        ## If array is 0 then don't decode. Else we just end up with a bunch of padding
        ## This happens because the output shape is specified as the longest output sequence (e.g. Ty=30)
        ## If the sequence that is being translated is, say, 14 characters, then 16 characters will be 0 (or padding)
        if np.sum(array) > 0:
        
            decoded.append(decode_position(np.argmax(array), mapping))
    
    ## Return
    return("".join(decoded))


