from collections import Counter
from math import log2
import numpy as np
import nltk
import os

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
        mle = 1.0
        for ngram in self.ngrams(sequence):
            #check if n-1gram has p > 0
            if self.n_1gram[ngram[:-1]] > 0:
                mle *= self.counts[ngram] / self.n_1gram[ngram[:-1]]
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
        for alpha in [x * 0.01 for x in range(101)]:
            estimate = self.cross_entropy(sequence, alpha)
            if estimate < prev_estimate:
                lowest = alpha
            prev_estimate = estimate
        return lowest






#n = Ngram()

#path = "assignment1-data"
#for filename in os.listdir(path):
#    if filename is not "c00.txt" and filename is not "d00.txt":


#for sent in tokenize("assignment1-data/c01.txt"):
#    n.update(sent)
#for sent in tokenize("assignment1-data/c02.txt"):
#    n.update(sent)
#for sent in tokenize("assignment1-data/c03.txt"):
#    n.update(sent)
#for sent in tokenize("assignment1-data/c04.txt"):
#    n.update(sent)
#for sent in tokenize("assignment1-data/c05.txt"):
#    n.update(sent)

# x = tokenize("assignment1-data/c06.txt")
# print(n.perplexity(x))
#
# estm = n.estimate_alpha(x)
# print(estm)
#
# print(n.perplexity(x, estm))









class BackoffNgram(object):
    __slots__ = 'n', 'counts', 'n_1gram', 'n_2gram', 'singletons_n_grams', 'lmbda', 'singletons_n1_grams', 'beta',\
                'vocab', 'start', 'end'

    def __init__(self, n=3, start='<$>', end='</$>'):
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

    def prob(self, sequence, alpha=1.0):
        p = 0
        for ngram in self.ngrams(sequence):
            if self.counts[ngram] > 0:
                p += (1-self.lmbda) * np.log2((self.counts[ngram]))  -np.log2(self.n_1gram[ngram[:-1]])
            elif self.n_1gram[ngram[:-1]] > 0:
                p += self.lmbda * (1-self.beta) * np.log2((self.n_1gram[ngram[:-1]])) -np.log2(self.n_2gram[ngram[:-2]])
            else:
                p += self.lmbda * self.beta * np.log2(((self.n_2gram[ngram[:-2]] + alpha))) -np.log2(sum(self.n_2gram.values()) + (alpha * len(self.vocab)))
        return p

    def cross_entropy(self, sequence, alpha=1.0):
        seq_length = 0
        mle = 0
        for sent in sequence:
            seq_length += len(sent) + 1
            mle += self.prob(sent, alpha)
        return -(1 / seq_length) * mle

    def perplexity(self, sequence, alpha=1.0):
        return pow(2, self.cross_entropy(sequence, alpha))

    def estimate_alpha(self, sequence):
        lowest = 1
        prev_estimate = self.cross_entropy(sequence, 1.0)
        for alpha in [x * 0.01 for x in range(101)]:
            estimate = self.cross_entropy(sequence, alpha)
            print(alpha)
            print(estimate)
            print()
            if estimate < prev_estimate:
                lowest = alpha
            prev_estimate = estimate
        return lowest




b = BackoffNgram()

for sent in tokenize("assignment1-data/c01.txt"):
    b.update(sent)
#for sent in tokenize("assignment1-data/c02.txt"):
#    b.update(sent)
#for sent in tokenize("assignment1-data/c03.txt"):
#    b.update(sent)
#for sent in tokenize("assignment1-data/c04.txt"):
#    b.update(sent)
#for sent in tokenize("assignment1-data/c05.txt"):
#    b.update(sent)
#
b.update_singletons()
b.update_hyperparameters()

x = tokenize("assignment1-data/c06.txt")
print(b.perplexity(x))

x2 = b.perplexity([['He', 'stooped', 'and', 'kissed', 'her', '.']])
print(x2)

print(b.estimate_alpha(x))
