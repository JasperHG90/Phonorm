## Prep data functions

class charmap:

    '''
    Create an object containing hash map from character --> integer and vice-versa

    adapted from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    '''

    def __init__(self, name, split=False):

        # Language name
        self.name = name
        # To map characters --> numeric index value
        self.char2index = {'<PAD>': 0, '<UNK>': 1, '\t': 2, "\n": 3}
        # Reverse the index
        self.index2char = {0: '<PAD>', 1: '<UNK>', 2: "\t", 3: "\n"}
        # Character frequencies
        self.char2count = {}
        # Number of characters --> use to create char index
        self.n_chars = 4
        # Max length
        self.max_length = 0
        self.split = split

    def addWord(self, word):

        '''
        Add unique characters to the hash map

        :param word: single word to be evaluated
        '''

        '''For each character, add to character index + reverse'''

        ## Check length of word. If > than self.max_length, then replace

        # If the word contains spaces, split at spaces. Else, just create a list out the characters
        if self.split:

            if len("".join(word.split())) > self.max_length:

                self.max_length = len("".join(word.split()))

            for char in word.split(" "):
                self.addChar(char)

        else:

            if len(word) > self.max_length:

                self.max_length = len(word)

            for char in list(word):

                self.addChar(char)

    def addChar(self, char):

        '''
        Add character to the hash maps

        :param char: single character as string
        '''

        if not char in self.char2index:

            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1

        else:
            
            if char not in ["\t", "\n"]:

                self.char2count[char] += 1
