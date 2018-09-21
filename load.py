import torch
import re
import os
import unicodedata

from config import MAX_LENGTH, save_dir, vocab_size
'''
数据需处理为：
你好  （post）  \t  你也好 (response)
吃了吗 （post） \t  吃了   (response)
'''
SOS_token = 1
EOS_token = 2
PAD_token = 0
UNK_token = 3

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # Count SOS and EOS

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

    def cutVocab(self):
        print(self.n_words)
        temp = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)
        temp = temp[:vocab_size-4]
        self.word2index = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = vocab_size  # Count SOS and EOS
        for i in range(len(temp)):
            self.word2index[temp[i][0]] = i+4
            self.index2word[i+4] = temp[i][0]
        print(len(self.word2index))
        print(len(self.index2word))
        self.word2count = dict(temp)


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    '''
    lines = [x.strip() for x in content]
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    pairs = [[x, next(it)] for x in it]
    '''
    lines = [x.strip('\n').strip() for x in content]
    pairs = []
    for line in lines:
        post, response = line.split('\t')
        pairs.append([post, response])
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name):
    voc, pairs = readVocs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    voc.cutVocab()
    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(corpus, corpus_name)
    return voc, pairs
