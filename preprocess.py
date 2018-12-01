'''
    The full process for preparing the data is:
        Read text file and split into lines, split lines into pairs
        Normalize text, filter by length and content
        Make word lists from sentences in pairs
'''
import re
import unicodedata
from io import open
import string
import random
# The cPickle module implements the same algorithm, in C instead of Python.
# It is many times faster than the Python implementation, 
# but does not allow the user to subclass from Pickle.
import _pickle as cPickle
from pickle import HIGHEST_PROTOCOL

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

'''
    We need a unique index per word to use as the inputs and targets of the 
    networks later. 
    Helper class Lang has word → index (word2index) 
    and index → word (index2word) dictionaries, 
    as well as a count of each word word2count to use to later replace 
    rare words.
'''
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Convert to lowercase, turn a Unicode string to plain ASCII, 
# trim most punctuation, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

'''
    To read the data file we will split the file into lines, 
    and then split lines into pairs. 
    The files are all English → Other Language, 
    so if we want to translate from Other Language → English 
    use the reverse flag to reverse the pairs.
'''
def readLangs(lang1, lang2, reverse=False):
    # Read the file and split into lines
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        pairs = [list(p) for p in pairs]
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

'''
    Since there are a lot of example sentences and we want to train 
    something quickly, we trim the data set to only relatively short
    and simple sentences. Here the maximum length is 10 words.
'''
def filterPairs(pairs, reverse):
    def filterPair(p, reverse):
        if reverse == True:
            return len(p[0].split(' ')) < MAX_LENGTH and \
                   len(p[1].split(' ')) < MAX_LENGTH and \
                   p[1].startswith(eng_prefixes)
        else:
            return len(p[0].split(' ')) < MAX_LENGTH and \
                   len(p[1].split(' ')) < MAX_LENGTH and \
                   p[0].startswith(eng_prefixes)
    
    return [pair for pair in pairs if filterPair(pair, reverse)]

'''
    To translate from French to English, set reverse True
'''
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

def main():    
    input_lang, output_lang, pairs = prepareData('eng', 'fra')
    random.shuffle(pairs)
    N_samples = len(pairs)
    N_train = int(0.80 * N_samples)
    # test pairs and train pairs:
    train_pairs = pairs[:N_train]
    test_pairs = pairs[N_train:] 
    # Saving the processed data:
    with open('processed_text_Data.save', 'wb') as f:  
        cPickle.dump([input_lang, output_lang, train_pairs, test_pairs], f, 
                     protocol=HIGHEST_PROTOCOL)
        f.close()


if __name__ == "__main__":
    main()