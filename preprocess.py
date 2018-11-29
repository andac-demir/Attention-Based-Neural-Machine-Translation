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
import pickle

SOS_token = 0
EOS_token = 1
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


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim most punctuation, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
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
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

'''
    Since there are a lot of example sentences and we want to train 
    something quickly, we trim the data set to only relatively short
    and simple sentences. Here the maximum length is 10 words.
'''
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

'''
    Dataset is prepared as sentences in French to English.
    To translate from English to French, set reverse True
'''
def prepareData(lang1, lang2, reverse=True):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def main():    
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    # Saving the processed data:
    with open('processed_text_Data.pkl', 'wb') as f:  
        pickle.dump([input_lang, output_lang, pairs], f)


if __name__ == "__main__":
    main()