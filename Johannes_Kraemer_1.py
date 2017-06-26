from collections import Counter
from math import log2
import numpy as np
import nltk
import os
from texttable import Texttable

#returns list of lists of words
def tokenize(path):
    f = open(path, encoding='utf8').read()
    return [nltk.word_tokenize(s) for s in nltk.sent_tokenize(f)]

class Ngram(object):
    __slots__ = 'n', 'counts', 'n_1gram', 'vocab', 'start', 'end', 'pplexity'

    def __init__(self, n=2, start='<$>', end='</$>'):
        self.n = n
        self.counts = Counter() #ngram freq
        self.n_1gram = Counter() #n-1gram freq
        self.vocab = set() #all tokens ---> V
        self.start = start
        self.end = end
        self.pplexity = 0

    #update the ngrams frequency
    def update(self, sequence):
        if not isinstance(sequence, list):
            raise TypeError('Expected argument of type list!')
        self.vocab.update(sequence)
        for ngram in self.ngrams(sequence):
            #update ngram freq
            if ngram in self.counts:
                self.counts[ngram] += 1
            else:
                self.counts[ngram] = 1
            #update n-1gram freq
            if ngram[:-1] in self.n_1gram:
                self.n_1gram[ngram[:-1]] += 1
            else:
                self.n_1gram[ngram[:-1]] = 1


    #get ngrams from list of tokens (helper method for update)
    def ngrams(self, sequence):
        t = [self.start] * (self.n - 1) #n-1 starting symbols
        t += sequence
        t.append(self.end)
        return zip(*[t[i:] for i in range(self.n)])

    #returns the relative freq (MLE probability) of the given sequence.
    def prob_mle(self, sequence):
        if isinstance(sequence, str):
            sequence = nltk.word_tokenize(sequence)
        mle = 0.0
        for ngram in self.ngrams(sequence):
            #check if n-1gram has p > 0
            if self.n_1gram[ngram[:-1]] > 0:
                mle += np.log2(self.counts[ngram]) -np.log2(self.n_1gram[ngram[:-1]])
        return  mle

    #returns the probability of the given sequence with additive smoothing.
    def prob_add(self, sequence, alpha=1.0):
        if isinstance(sequence, str):
            sequence = nltk.word_tokenize(sequence)
        mle = 0
        for ngram in self.ngrams(sequence):
            if alpha == 0:
                # check if n-1gram has p > 0
                if self.n_1gram[ngram[:-1]] > 0:
                    if self.counts[ngram] == 0:
                        return float('-inf')
                    mle += np.log2((self.counts[ngram] + alpha)) -np.log2(self.n_1gram[ngram[:-1]] + (alpha * len(self.vocab)))
            else:
                mle += np.log2((self.counts[ngram] + alpha)) -np.log2(self.n_1gram[ngram[:-1]] + (alpha * len(self.vocab)))
        return mle

    #estimate of cross entropy
    def cross_entropy(self, sequence, alpha=1.0):
        seq_length = 0
        mle = 0
        for sent in sequence:
            seq_length += len(sent)+1
            mle += self.prob_add(sent, alpha)
        return -(1 / seq_length) * mle

    def perplexity(self, sequence, alpha=1.0):
        return pow(2, self.cross_entropy(sequence, alpha))

    def estimate_alpha(self, sequence):
        lowest = 1
        prev_estimate = self.cross_entropy(sequence, 1)
        for alpha in [x * 0.1 for x in range(11)]:
            estimate = self.cross_entropy(sequence, alpha)
            if estimate < prev_estimate:
                lowest = alpha
            prev_estimate = estimate
        return lowest



class BackoffNgram(object):
    __slots__ = 'n', 'counts', 'n_1gram', 'n_2gram', 'singletons_n_grams', 'lmbda', 'singletons_n1_grams', 'beta',\
                'vocab', 'start', 'end', 'estimated_alpha'

    def __init__(self, n=3, alpha=1, start='<$>', end='</$>'):
        self.n = n
        self.counts = Counter() #ngram freq
        self.n_1gram = Counter() #n-1gram freq
        self.n_2gram = Counter() #n-2gram freq
        self.vocab = set() #all tokens ---> V
        self.start = start
        self.end = end
        self.singletons_n_grams = set() #ngrams that appear only ones in the counts
        self.singletons_n1_grams = set() #n-1grams that appear only ones in the n_1grams
        self.lmbda = 0 #lambda ---> good turing discount for trigrams (for ngrams)
        self.beta = 0 #beta ---> good turing discout for bigrams (for n-1grams)
        self.estimated_alpha = alpha #use the alpha as given parameter (which was estimated from the unigram model)

    def updateAlpha(self, alpha):
        self.estimated_alpha = alpha

    def update(self, sequence):
        if not isinstance(sequence, list):
            raise TypeError('Expected argument of type list!')
        #update vocabulary
        self.vocab.update(sequence)
        for ngram in self.ngrams(sequence):
            #update counts
            if ngram in self.counts:
                self.counts[ngram] += 1
            else:
                self.counts[ngram] = 1
            #update n-1gram
            if ngram[:-1] in self.n_1gram:
                self.n_1gram[ngram[:-1]] += 1
            else:
                self.n_1gram[ngram[:-1]] = 1
            #update n-2 gram
            if ngram[:-2] in self.n_2gram:
                self.n_2gram[ngram[:-2]] += 1
            else:
                self.n_2gram[ngram[:-2]] = 1

    def update_singletons(self):
        #update singletons for ngrams
        self.singletons_n_grams = set()
        for ngram in self.counts:
            if self.counts[ngram] == 1:
                self.singletons_n_grams.add(ngram)
        #update singletons for n-1grams
        self.singletons_n1_grams = set()
        for n1gram in self.n_1gram:
            if self.n_1gram[n1gram] == 1:
                self.singletons_n1_grams.add(n1gram)

    def update_hyperparameters(self):
        #update lambda
        self.lmbda = len(self.singletons_n_grams) / sum(self.counts.values())
        #update beta
        self.beta = len(self.singletons_n1_grams) / sum(self.n_1gram.values())

    # get ngrams from list of tokens (helper method for update)
    def ngrams(self, sequence):
        t = [self.start] * (self.n - 1)  # n-1 starting symbols
        t += sequence
        t.append(self.end)
        return zip(*[t[i:] for i in range(self.n)])

    def prob(self, sequence):
        p = 0
        for ngram in self.ngrams(sequence):
            if self.counts[ngram] > 0:
                p += np.log2((1-self.lmbda)) + np.log2((self.counts[ngram]))  -np.log2(self.n_1gram[ngram[:-1]])
            elif self.n_1gram[ngram[:-1]] > 0:
                p += np.log2(self.lmbda) + (1-self.beta) * np.log2((self.n_1gram[ngram[:-1]])) -np.log2(self.n_2gram[ngram[:-2]])
            else:
                p += np.log2(self.lmbda) + self.beta * np.log2(((self.n_2gram[ngram[:-2]] + self.estimated_alpha))) -np.log2(sum(self.n_2gram.values()) + (self.estimated_alpha * len(self.vocab)))
        return p

    def cross_entropy(self, sequence):
        seq_length = 0
        mle = 0
        for sent in sequence:
            seq_length += len(sent) + 1
            mle += self.prob(sent)
        return -(1 / seq_length) * mle

    def perplexity(self, sequence):
        return pow(2, self.cross_entropy(sequence))


#    def estimate_alpha(self, sequence):
#        lowest = 1
#        prev_estimate = self.cross_entropy(sequence, 1.0)
#        for alpha in [x * 0.1 for x in range(11)]:
#            estimate = self.cross_entropy(sequence, alpha)
#            if estimate < prev_estimate:
#                lowest = alpha
#            prev_estimate = estimate
#        return lowest



############################################################################
#PUTTING IT ALL TOGETHER
############################################################################

#result
#should be a table of this format:
# [ [files: c1gl: c1ga: ...]
#   [c00    900.0 850.0 ...]
#   [d00    900.0 850.0 ...]
#   .
#   .
#   .
#   [t08    900.0 850.0 ...] ]
score_table = []

#testset
c00 = tokenize('assignment1-data/c00.txt')
d00 = tokenize('assignment1-data/d00.txt')
t01 = tokenize('assignment1-data/t01.txt')
t02 = tokenize('assignment1-data/t02.txt')
t03 = tokenize('assignment1-data/t03.txt')
t04 = tokenize('assignment1-data/t04.txt')
t05 = tokenize('assignment1-data/t05.txt')
t06 = tokenize('assignment1-data/t06.txt')
t07 = tokenize('assignment1-data/t07.txt')
t08 = tokenize('assignment1-data/t08.txt')

#get filenames
file_names = []
with open('assignment1-data/list', 'r') as f:
    for line in f:
        line = line.strip("\n")
        file_names.append(line)

#unigrams with the c-files
n1_c = Ngram(1)
for file_name in file_names:
    if str(file_name).startswith('c') and str(file_name) != 'c00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n1_c.update(sent)
c1l = []
c1l.append(n1_c.perplexity(c00))
c1l.append(n1_c.perplexity(d00))
c1l.append(n1_c.perplexity(t01))
c1l.append(n1_c.perplexity(t02))
c1l.append(n1_c.perplexity(t03))
c1l.append(n1_c.perplexity(t04))
c1l.append(n1_c.perplexity(t05))
c1l.append(n1_c.perplexity(t06))
c1l.append(n1_c.perplexity(t07))
c1l.append(n1_c.perplexity(t08))

#estimate alpha seperatly in order to reuse it in the backoff ngram implementation
c_est_c00 = n1_c.estimate_alpha(c00)
c_est_d00 = n1_c.estimate_alpha(d00)
c_est_t01 = n1_c.estimate_alpha(t01)
c_est_t02 = n1_c.estimate_alpha(t02)
c_est_t03 = n1_c.estimate_alpha(t03)
c_est_t04 = n1_c.estimate_alpha(t04)
c_est_t05 = n1_c.estimate_alpha(t05)
c_est_t06 = n1_c.estimate_alpha(t06)
c_est_t07 = n1_c.estimate_alpha(t07)
c_est_t08 = n1_c.estimate_alpha(t08)

c1a = []
c1a.append(n1_c.perplexity(c00, c_est_c00))
c1a.append(n1_c.perplexity(d00, c_est_d00))
c1a.append(n1_c.perplexity(t01, c_est_t01))
c1a.append(n1_c.perplexity(t02, c_est_t02))
c1a.append(n1_c.perplexity(t03, c_est_t03))
c1a.append(n1_c.perplexity(t04, c_est_t04))
c1a.append(n1_c.perplexity(t05, c_est_t05))
c1a.append(n1_c.perplexity(t06, c_est_t06))
c1a.append(n1_c.perplexity(t07, c_est_t07))
c1a.append(n1_c.perplexity(t08, c_est_t08))



#unigrams with the d-files
n1_d = Ngram(1)
for file_name in file_names:
    if str(file_name).startswith('d') and str(file_name) != 'd00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n1_d.update(sent)

d1l = []
d1l.append(n1_d.perplexity(c00))
d1l.append(n1_d.perplexity(d00))
d1l.append(n1_d.perplexity(t01))
d1l.append(n1_d.perplexity(t02))
d1l.append(n1_d.perplexity(t03))
d1l.append(n1_d.perplexity(t04))
d1l.append(n1_d.perplexity(t05))
d1l.append(n1_d.perplexity(t06))
d1l.append(n1_d.perplexity(t07))
d1l.append(n1_d.perplexity(t08))

#estimate alpha seperatly in order to reuse it in the backoff ngram implementation
d_est_c00 = n1_d.estimate_alpha(c00)
d_est_d00 = n1_d.estimate_alpha(d00)
d_est_t01 = n1_d.estimate_alpha(t01)
d_est_t02 = n1_d.estimate_alpha(t02)
d_est_t03 = n1_d.estimate_alpha(t03)
d_est_t04 = n1_d.estimate_alpha(t04)
d_est_t05 = n1_d.estimate_alpha(t05)
d_est_t06 = n1_d.estimate_alpha(t06)
d_est_t07 = n1_d.estimate_alpha(t07)
d_est_t08 = n1_d.estimate_alpha(t08)

d1a = []
d1a.append(n1_d.perplexity(c00, d_est_c00))
d1a.append(n1_d.perplexity(d00, d_est_d00))
d1a.append(n1_d.perplexity(t01, d_est_t01))
d1a.append(n1_d.perplexity(t02, d_est_t02))
d1a.append(n1_d.perplexity(t03, d_est_t03))
d1a.append(n1_d.perplexity(t04, d_est_t04))
d1a.append(n1_d.perplexity(t05, d_est_t05))
d1a.append(n1_d.perplexity(t06, d_est_t06))
d1a.append(n1_d.perplexity(t07, d_est_t07))
d1a.append(n1_d.perplexity(t08, d_est_t08))


#bigrams with c-files
n2_c = Ngram(2)
for file_name in file_names:
    if str(file_name).startswith('c') and str(file_name) != 'c00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n2_c.update(sent)

c2l = []
c2l.append(n2_c.perplexity(c00))
c2l.append(n2_c.perplexity(d00))
c2l.append(n2_c.perplexity(t01))
c2l.append(n2_c.perplexity(t02))
c2l.append(n2_c.perplexity(t03))
c2l.append(n2_c.perplexity(t04))
c2l.append(n2_c.perplexity(t05))
c2l.append(n2_c.perplexity(t06))
c2l.append(n2_c.perplexity(t07))
c2l.append(n2_c.perplexity(t08))

c2a = []
c2a.append(n2_c.perplexity(c00, n2_c.estimate_alpha(c00)))
c2a.append(n2_c.perplexity(d00, n2_c.estimate_alpha(d00)))
c2a.append(n2_c.perplexity(t01, n2_c.estimate_alpha(t01)))
c2a.append(n2_c.perplexity(t02, n2_c.estimate_alpha(t02)))
c2a.append(n2_c.perplexity(t03, n2_c.estimate_alpha(t03)))
c2a.append(n2_c.perplexity(t04, n2_c.estimate_alpha(t04)))
c2a.append(n2_c.perplexity(t05, n2_c.estimate_alpha(t05)))
c2a.append(n2_c.perplexity(t06, n2_c.estimate_alpha(t06)))
c2a.append(n2_c.perplexity(t07, n2_c.estimate_alpha(t07)))
c2a.append(n2_c.perplexity(t08, n2_c.estimate_alpha(t08)))


#bigrams with the d-files
n2_d = Ngram(2)
for file_name in file_names:
    if str(file_name).startswith('d') and str(file_name) != 'd00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n2_d.update(sent)

d2l = []
d2l.append(n2_d.perplexity(c00))
d2l.append(n2_d.perplexity(d00))
d2l.append(n2_d.perplexity(t01))
d2l.append(n2_d.perplexity(t02))
d2l.append(n2_d.perplexity(t03))
d2l.append(n2_d.perplexity(t04))
d2l.append(n2_d.perplexity(t05))
d2l.append(n2_d.perplexity(t06))
d2l.append(n2_d.perplexity(t07))
d2l.append(n2_d.perplexity(t08))

d2a = []
d2a.append(n2_d.perplexity(c00, n2_d.estimate_alpha(c00)))
d2a.append(n2_d.perplexity(d00, n2_d.estimate_alpha(d00)))
d2a.append(n2_d.perplexity(t01, n2_d.estimate_alpha(t01)))
d2a.append(n2_d.perplexity(t02, n2_d.estimate_alpha(t02)))
d2a.append(n2_d.perplexity(t03, n2_d.estimate_alpha(t03)))
d2a.append(n2_d.perplexity(t04, n2_d.estimate_alpha(t04)))
d2a.append(n2_d.perplexity(t05, n2_d.estimate_alpha(t05)))
d2a.append(n2_d.perplexity(t06, n2_d.estimate_alpha(t06)))
d2a.append(n2_d.perplexity(t07, n2_d.estimate_alpha(t07)))
d2a.append(n2_d.perplexity(t08, n2_d.estimate_alpha(t08)))


#trigrams with the c-files
n3_c = Ngram(3)
for file_name in file_names:
    if str(file_name).startswith('c') and str(file_name) != 'c00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n3_c.update(sent)

c3l = []
c3l.append(n3_c.perplexity(c00))
c3l.append(n3_c.perplexity(d00))
c3l.append(n3_c.perplexity(t01))
c3l.append(n3_c.perplexity(t02))
c3l.append(n3_c.perplexity(t03))
c3l.append(n3_c.perplexity(t04))
c3l.append(n3_c.perplexity(t05))
c3l.append(n3_c.perplexity(t06))
c3l.append(n3_c.perplexity(t07))
c3l.append(n3_c.perplexity(t08))

c3a = []
c3a.append(n3_c.perplexity(c00, n3_c.estimate_alpha(c00)))
c3a.append(n3_c.perplexity(d00, n3_c.estimate_alpha(d00)))
c3a.append(n3_c.perplexity(t01, n3_c.estimate_alpha(t01)))
c3a.append(n3_c.perplexity(t02, n3_c.estimate_alpha(t02)))
c3a.append(n3_c.perplexity(t03, n3_c.estimate_alpha(t03)))
c3a.append(n3_c.perplexity(t04, n3_c.estimate_alpha(t04)))
c3a.append(n3_c.perplexity(t05, n3_c.estimate_alpha(t05)))
c3a.append(n3_c.perplexity(t06, n3_c.estimate_alpha(t06)))
c3a.append(n3_c.perplexity(t07, n3_c.estimate_alpha(t07)))
c3a.append(n3_c.perplexity(t08, n3_c.estimate_alpha(t08)))


#trigrams with the d-files
n3_d = Ngram(3)
for file_name in file_names:
    if str(file_name).startswith('d') and str(file_name) != 'd00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            n3_d.update(sent)

d3l = []
d3l.append(n3_d.perplexity(c00))
d3l.append(n3_d.perplexity(d00))
d3l.append(n3_d.perplexity(t01))
d3l.append(n3_d.perplexity(t02))
d3l.append(n3_d.perplexity(t03))
d3l.append(n3_d.perplexity(t04))
d3l.append(n3_d.perplexity(t05))
d3l.append(n3_d.perplexity(t06))
d3l.append(n3_d.perplexity(t07))
d3l.append(n3_d.perplexity(t08))

d3a = []
d3a.append(n3_d.perplexity(c00, n3_d.estimate_alpha(c00)))
d3a.append(n3_d.perplexity(d00, n3_d.estimate_alpha(d00)))
d3a.append(n3_d.perplexity(t01, n3_d.estimate_alpha(t01)))
d3a.append(n3_d.perplexity(t02, n3_d.estimate_alpha(t02)))
d3a.append(n3_d.perplexity(t03, n3_d.estimate_alpha(t03)))
d3a.append(n3_d.perplexity(t04, n3_d.estimate_alpha(t04)))
d3a.append(n3_d.perplexity(t05, n3_d.estimate_alpha(t05)))
d3a.append(n3_d.perplexity(t06, n3_d.estimate_alpha(t06)))
d3a.append(n3_d.perplexity(t07, n3_d.estimate_alpha(t07)))
d3a.append(n3_d.perplexity(t08, n3_d.estimate_alpha(t08)))


#backoff ngram with the c-files
backoff_c = BackoffNgram()
for file_name in file_names:
    if str(file_name).startswith('c') and str(file_name) != 'c00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            backoff_c.update(sent)

#i excluded the updating of the parameters from the update method
backoff_c.update_singletons()
backoff_c.update_hyperparameters()

backoff_c_list = []
backoff_c.updateAlpha(c_est_c00)
backoff_c_list.append(backoff_c.perplexity(c00))
backoff_c.updateAlpha(c_est_d00)
backoff_c_list.append(backoff_c.perplexity(d00))
backoff_c.updateAlpha(c_est_t01)
backoff_c_list.append(backoff_c.perplexity(t01))
backoff_c.updateAlpha(c_est_t02)
backoff_c_list.append(backoff_c.perplexity(t02))
backoff_c.updateAlpha(c_est_t03)
backoff_c_list.append(backoff_c.perplexity(t03))
backoff_c.updateAlpha(c_est_t04)
backoff_c_list.append(backoff_c.perplexity(t04))
backoff_c.updateAlpha(c_est_t05)
backoff_c_list.append(backoff_c.perplexity(t05))
backoff_c.updateAlpha(c_est_t06)
backoff_c_list.append(backoff_c.perplexity(t06))
backoff_c.updateAlpha(c_est_t07)
backoff_c_list.append(backoff_c.perplexity(t07))
backoff_c.updateAlpha(c_est_t08)
backoff_c_list.append(backoff_c.perplexity(t08))

#backoff ngram with the d-files
backoff_d = BackoffNgram()
for file_name in file_names:
    if str(file_name).startswith('d') and str(file_name) != 'd00':
        for sent in tokenize('assignment1-data/' + file_name + '.txt'):
            backoff_d.update(sent)

#i excluded the updating of the parameters from the update method
backoff_d.update_singletons()
backoff_d.update_hyperparameters()

backoff_d_list = []
backoff_d.updateAlpha(d_est_c00)
backoff_d_list.append(backoff_d.perplexity(c00))
backoff_d.updateAlpha(d_est_d00)
backoff_d_list.append(backoff_d.perplexity(d00))
backoff_d.updateAlpha(d_est_t01)
backoff_d_list.append(backoff_d.perplexity(t01))
backoff_d.updateAlpha(d_est_t02)
backoff_d_list.append(backoff_d.perplexity(t02))
backoff_d.updateAlpha(d_est_t03)
backoff_d_list.append(backoff_d.perplexity(t03))
backoff_d.updateAlpha(d_est_t04)
backoff_d_list.append(backoff_d.perplexity(t04))
backoff_d.updateAlpha(d_est_t05)
backoff_d_list.append(backoff_d.perplexity(t05))
backoff_d.updateAlpha(d_est_t06)
backoff_d_list.append(backoff_d.perplexity(t06))
backoff_d.updateAlpha(d_est_t07)
backoff_d_list.append(backoff_d.perplexity(t07))
backoff_d.updateAlpha(d_est_t08)
backoff_d_list.append(backoff_d.perplexity(t08))


###################################################################################
#PRINT THE WHOLE THING
###################################################################################
table1 = Texttable()
table1.add_rows([['files: ', 'c-1g-l: ', 'c-1g-a: ', 'd-1g-l: ', 'd-1g-a: '],
                 ['c00', c1l[0], c1a[0], d1l[0], d1a[0]],
                 ['d00', c1l[1], c1a[1], d1l[1], d1a[1]],
                 ['t01', c1l[2], c1a[2], d1l[2], d1a[2]],
                 ['t02', c1l[3], c1a[3], d1l[3], d1a[3]],
                 ['t03', c1l[4], c1a[4], d1l[4], d1a[4]],
                 ['t04', c1l[5], c1a[5], d1l[5], d1a[5]],
                 ['t05', c1l[6], c1a[6], d1l[6], d1a[6]],
                 ['t06', c1l[7], c1a[7], d1l[7], d1a[7]],
                 ['t07', c1l[8], c1a[8], d1l[8], d1a[8]],
                 ['t08', c1l[9], c1a[9], d1l[9], d1a[9]]])
print(table1.draw())
print()
print()
print()


table2 = Texttable()
table2.add_rows([['files: ', 'c-2g-l: ', 'c-2g-a: ', 'd-2g-l: ', 'd-2g-a: '],
                 ['c00', c2l[0], c2a[0], d2l[0], d2a[0]],
                 ['d00', c2l[1], c2a[1], d2l[1], d2a[1]],
                 ['t01', c2l[2], c2a[2], d2l[2], d2a[2]],
                 ['t02', c2l[3], c2a[3], d2l[3], d2a[3]],
                 ['t03', c2l[4], c2a[4], d2l[4], d2a[4]],
                 ['t04', c2l[5], c2a[5], d2l[5], d2a[5]],
                 ['t05', c2l[6], c2a[6], d2l[6], d2a[6]],
                 ['t06', c2l[7], c2a[7], d2l[7], d2a[7]],
                 ['t07', c2l[8], c2a[8], d2l[8], d2a[8]],
                 ['t08', c2l[9], c2a[9], d2l[9], d2a[9]]])
print(table2.draw())
print()
print()
print()

table3 = Texttable()
table3.add_rows([['files: ', 'c-3g-l: ', 'c-3g-a: ', 'd-3g-l: ', 'd-3g-a: '],
                 ['c00', c3l[0], c3a[0], d3l[0], d3a[0]],
                 ['d00', c3l[1], c3a[1], d3l[1], d3a[1]],
                 ['t01', c3l[2], c3a[2], d3l[2], d3a[2]],
                 ['t02', c3l[3], c3a[3], d3l[3], d3a[3]],
                 ['t03', c3l[4], c3a[4], d3l[4], d3a[4]],
                 ['t04', c3l[5], c3a[5], d3l[5], d3a[5]],
                 ['t05', c3l[6], c3a[6], d3l[6], d3a[6]],
                 ['t06', c3l[7], c3a[7], d3l[7], d3a[7]],
                 ['t07', c3l[8], c3a[8], d3l[8], d3a[8]],
                 ['t08', c3l[9], c3a[9], d3l[9], d3a[9]]])
print(table3.draw())
print()
print()
print()

table4 = Texttable()
table4.add_rows([['files: ', 'c-backoff: ', 'd-backoff: '],
                 ['c00', backoff_c_list[0], backoff_d_list[0]],
                 ['d00', backoff_c_list[1], backoff_d_list[1]],
                 ['t01', backoff_c_list[2], backoff_d_list[2]],
                 ['t02', backoff_c_list[3], backoff_d_list[3]],
                 ['t03', backoff_c_list[4], backoff_d_list[4]],
                 ['t04', backoff_c_list[5], backoff_d_list[5]],
                 ['t05', backoff_c_list[6], backoff_d_list[6]],
                 ['t06', backoff_c_list[7], backoff_d_list[7]],
                 ['t07', backoff_c_list[8], backoff_d_list[8]],
                 ['t08', backoff_c_list[9], backoff_d_list[9]]])
print(table4.draw())
print()
print()
print()

###########################################################################
#Authors of the documents
#c*   Wilkie Collins
#d*   Charles Dickens
#t01  Charles Dickens
#t02  Wilkie Collins
#t03  Wilkie Collins
#t04  Charles Dickens
#t05  Wilkie Collins
#t06  Mark Twain
#t07  Lewis Carroll (Charles Lutwidge Dodgson)
#t08  Jane Austen

