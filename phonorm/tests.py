'''Tests go here'''

from phonorm.utilities import one_hot_encode, decode_from_ohe
import random

def check_ohe(pairs, input_mapping, output_mapping, n=10, split = False):
    
    '''
    Check if one-hot encoding functions work properly. 
    
    :param pairs: input pairs of length n
    :param input_mapping: charmap for input language
    :param output_mapping: charmap for output language
    :param n: number of words to evaluate. Defaults to 10
    :param split: if True, then dealing with phonemes. This is relevant for the cmudict (multiple) model.
    '''
    
    pairs_random = [random.choice(pairs) for i in range(n)]
    pairs_in = [pair[0] for pair in pairs_random]
    pairs_out = [pair[1] for pair in pairs_random]
    
    ## Encode
    encoder_in = one_hot_encode(pairs_in, input_mapping)
    decoder_in = one_hot_encode(pairs_out, output_mapping, split = split)
    decoder_target = one_hot_encode(pairs_out, output_mapping, one_timestep_ahead = True, split = split)
    
    print("decoded and 'target decoded' should be the same\n")
    
    ## For each example
    for i in range(n):
        
        input_word = pairs_in[i]
        output_word = pairs_out[i]
        
        tst_enc_input = encoder_in[i]
        tst_dec_input = decoder_in[i]
        tst_dec_target = decoder_target[i]
        
        ## Decode output
        print("---- Example " + str(i))
        print("Input: " + input_word + ", decoded: " + decode_from_ohe(tst_enc_input, input_mapping))
        print("Output: " + output_word.strip("\t").strip("\n") + 
              ", decoded: " + 
              decode_from_ohe(tst_dec_input, output_mapping).strip("\t").strip("\n") + 
              ", target decoded: " + 
              decode_from_ohe(tst_dec_target, output_mapping).strip("\t").strip("\n"))
        print("\n")
        