from __future__ import division
from corpus import Document, NamesCorpus, ReviewCorpus
from maxent import MaxEnt
from unittest import TestCase, main, skip
from random import shuffle, seed
from collections import defaultdict
from nltk.corpus import stopwords
import sys
import nltk
import random


class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""

        pattern = r'\w+'

        return nltk.regexp_tokenize(self.data.lower(), pattern)


class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]] 


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        #print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        print("%.2d%% " % (100 * sum(correct) / len(correct)), file=verbose)
    return float(sum(correct)) / len(correct)


class MaxEntTest(TestCase):
    u"""Tests for the MaxEnt classifier."""

    def get_review_vocab(self, dataset):
        """Get list of vocab words to serve as features for review model"""

        words = defaultdict(int)

        stops = set(stopwords.words('english'))

        # get frequencies of vocab words (non-stopwords) in text
        for doc in dataset:
            text = doc.features()
            for word in text:
                if word not in stops:
                    words[word] += 1

        return [w for w in words if words[w] >= 5]

    def get_names_vocab(self, dataset):
        """Get list of letter features for names model"""

        vocab = []

        for doc in dataset:
            if doc.features()[0] not in vocab:
                vocab.append(doc.features()[0])
            if doc.features()[1] not in vocab:
                vocab.append(doc.features()[1])

        return vocab

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training, dev, and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:5000], names[5000:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, dev, test = self.split_names_corpus()
        classifier = MaxEnt()
        classifier.labels = ['female', 'male']
        classifier.vocab = self.get_names_vocab(train)
        classifier.feature_vectors(train + dev + test)
        classifier.train(train, dev)

        acc = accuracy(classifier, test)
        self.assertGreater(acc, 0.70)

    def split_review_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
        seed(hash("reviews"))
        shuffle(reviews)
        return (reviews[:10000], reviews[10000:11000], reviews[11000:14000])

    def test_reviews_bag(self):
        """Classify sentiment using bag-of-words"""
        train, dev, test = self.split_review_corpus(BagOfWords)
        classifier = MaxEnt()
        classifier.labels = ['positive', 'negative', 'neutral']
        classifier.vocab = self.get_review_vocab(train)
        print('...creating sparse feature vectors...')
        classifier.feature_vectors(train + dev + test)

        print('...training...')
        classifier.train(train, dev)
        self.assertGreater(accuracy(classifier, test), 0.55)


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

