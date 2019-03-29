# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from scipy.special import logsumexp
from random import shuffle
import numpy as np
import math


class MaxEnt(Classifier):

    def __init__(self):
        self.labels = []
        self.weights = []
        self.vocab = []
        super(Classifier, self).__init__()

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def feature_vectors(self, dataset):
        """Construct sparse feature vector for each document in dataset"""

        for doc in dataset:
            for word in self.vocab:
                if word in doc.features():
                    doc.feature_vector.append(1)
                else:
                    doc.feature_vector.append(0)
            doc.feature_vector.append(1)    # for bias term

    def chop_up(self, instances, batch_size):
        """split given dataset into equal sized batches for gradient descent"""

        shuffle(instances)
        chopped = []

        i = 0
        while i < len(instances):
            chopped.append(instances[i:i+batch_size])
            i += batch_size

        return chopped

    def compute_gradient(self, minibatch, batch_size):
        """get average gradient matrix for given minibatch"""

        gradient_matrix = np.zeros([len(self.labels), len(self.vocab)+1], dtype=float)

        for doc in minibatch:
            for label_idx in range(len(self.labels)):
                # add row to gradient matrix
                if self.labels[label_idx] == doc.label:
                    gradient_matrix[label_idx] += np.subtract(doc.feature_vector, np.multiply(doc.feature_vector, self.posterior(doc, label_idx)))
                else:
                    gradient_matrix[label_idx] += np.subtract(0, np.multiply(doc.feature_vector, self.posterior(doc, label_idx)))

        # get average of gradients for each doc in minibatch
        return np.divide(gradient_matrix, batch_size)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""

        self.weights = np.zeros([len(self.labels), len(self.vocab)+1], dtype=float)

        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient Descent.  Convergence
        determined by performance on dev set"""

        minibatches = self.chop_up(train_instances, batch_size)

        converged = False
        prev_accuracy = 0
        iteration_count = 0
        datapoint_count = 0
        first_accuracy = 0
        count_converge = 0

        while not converged:
            print('iteration', iteration_count)

            for minibatch in minibatches:
                datapoint_count += 1
                gradient = self.compute_gradient(minibatch, batch_size)
                self.weights += (gradient * learning_rate)

            current_accuracy = self.accuracy(dev_instances)

            if iteration_count == 0:
                first_accuracy = current_accuracy

            print('prev_accuracy', prev_accuracy)
            print('current_accuracy', current_accuracy)
            print('difference between iterations:', abs(prev_accuracy - current_accuracy))

            if current_accuracy > first_accuracy and round(abs(prev_accuracy - current_accuracy), 4) <= .001:
                count_converge += 1
            else:
                count_converge = 0

            # converge once the difference between accuracies is small 40 times in a row
            if count_converge >= 40:
                converged = True
                print('...training has converged...')

            iteration_count += 1
            prev_accuracy = current_accuracy

    def accuracy(self, test_data):
        correct = [self.classify(x) == x.label for x in test_data]

        return float(sum(correct)) / len(correct)

    def posterior(self, doc, label_index):
        """get posterior probability of the given document being in a certain class"""

        numerator_score = np.dot(self.weights[label_index], doc.feature_vector)
        denominator_score = []

        # get sum of dot products of feature vector with weight vectors for each class
        for i in range(len(self.labels)):
            denominator_score.append(np.dot(self.weights[i], doc.feature_vector))

        return math.exp(numerator_score - logsumexp(denominator_score))

    def classify(self, instance):
        """predict class of a given document by computing probability of that document being in each class
        and select the class with the highest probability"""

        class_probs = []
        for label_idx in range(len(self.labels)):
            class_probs.append(self.posterior(instance, label_idx))

        best_class = int(np.argmax(class_probs))

        return self.labels[best_class]
