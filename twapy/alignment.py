"""This module handles everything having to do with aligning models and solving temporal word
analogies.

The `Alignment` class takes two models and fits a linear regression transformation between them,
and also applies this transformation to produce a third model.

The `Analogy` class solves a temporal word analogy. Its inputs are the LHS word, the LHS model,
and the RHS model, and its output is the RHS word. In its implementation, an `Analogy` must have
a corresponding `Alignment` which contains the models and does most of the math. The `Analogy`
just holds the input word and the results.

An example of how to use these classes to solve a temporal word analogy is:

>>> from twapy import Alignment
>>> a = Alignment('/path/to/models/1987.bin', '/path/to/models/1997.bin')
>>> print(a.analogy('reagan'))
1987 : reagan :: 1997 : clinton

"""

import random
import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
from . import VectorSpaceModel
from . import error, warn, info, debug



class Alignment(object):

    """An Alignment is based on two vector space models. It can fit a transformation from one
    space onto the other, and apply that transformation to produce a third, aligned, vector space
    model.

    """

    def __init__(self, model1, model2, collection=None, samplesize=0.5):
        """There are three ways to initialize an `Alignment`:
        1) With two VectorSpaceModel instances
        2) With two filenames (which can be loaded to VectorSpaceModel instances)
        3) With two model names and a ModelCollection instance
        """
        if collection is not None:
            self.model1 = collection[model1]
            self.model2 = collection[model2]
        else:
            if type(model1) is str:
                self.model1 = VectorSpaceModel.load(model1)
            else:
                self.model1 = model1
            if type(model2) is str:
                self.model2 = VectorSpaceModel.load(model2)
            else:
                self.model2 = model2
        self.model3 = None
        self.name = "{:}->{:}".format(self.model1.name, self.model2.name)
        self.samplesize = samplesize
        self.regression = None
        debug("Initialized {:}".format(self))
        self.fit_transform()
        return

    def __repr__(self):
        return("<Alignment of {:} onto {:}>".format(self.model1.name, self.model2.name))

    # @classmethod
    # def from_collection(cls, modelname1, modelname2, collection):
    #     return cls(model1, model2, modelname1, modelname2, collection)
    #
    # @classmethod
    # def from_files(cls, filename1, filename2):
    #     model1 = VectorSpaceModel.load(filename1)
    #     model2 = VectorSpaceModel.load(filename2)
    #     modelname1 = os.path.basename(filename1)
    #     modelname2 = os.path.basename(filename2)
    #     return cls(model1, model2, modelname1, modelname2)
    #
    def fit_transform(self):
        """Fit the regression that aligns model1 and model2."""
        debug("Fitting regression from '{:}' to '{:}'".format(self.model1.name, self.model2.name))
        self.regression = fit_w2v_regression(self.model1.m, self.model2.m, self.samplesize)
        debug("Applying transformation {:}".format(self.name))
        self.model3 = VectorSpaceModel(name=self.name)
        self.model3.m = apply_wv2_regression(self.model1.m, self.regression)
        return

    def compare_nns(self, word, topn=10, allow_keyerror=True):
        """This function will return a three-tuple of the lists of the topn nearest neighbors to
        the given word in each of the three alignment models."""
        try:
            nn1 = self.model1.most_similar(word, topn=topn)
        except KeyError:
            if allow_keyerror:
                nn1 = [("", 0.0)] * topn
            else:
                raise
        try:
            nn2 = self.model2.most_similar(word, topn=topn)
        except KeyError:
            if allow_keyerror:
                nn2 = [("", 0.0)] * topn
            else:
                raise
        try:
            nn3 = self.model3.most_similar(word, topn=topn)
        except KeyError:
            if allow_keyerror:
                nn3 = [("", 0.0)] * topn
            else:
                raise
        return nn1, nn2, nn3

    def analogy(self, word):
        return Analogy(word1=word, alignment=self)

    def print_analogy(self, word):
        result = self.analogy(word)
        print("{:>10s} : {:>20s} <=> {:<20s} : {:<10s}".format(
            self.modelname1, word, result, self.modelname2))
        return

    def print_compare_nns(self, word):
        nn1, nn2, nn3 = self.compare_nns(word)
        linewidth = 80
        s = ""
        s += "=" * linewidth + "\n"
        s += "     word: '{:}'\n".format(word)
        s += "-" * linewidth + "\n"
        s += "{:5}{:25}{:25}{:25}\n".format(
            "", "Model "+self.modelname1, "Model "+self.modelname2, "Model "+self.modelname3)
        s += "-" * linewidth + "\n"
        for i in range(len(nn1)):
            s += "{:<5d}{:25}{:25}{:25}\n".format(
                i+1, nn1[i][0], nn2[i][0], nn3[i][0])
        s += "-" * linewidth + "\n"
        print(s)


class Analogy(object):

    """An analogy is.... An analogy requires an alignment. If on is not passed to the
    constructor, then an `Alignment` will be created. """

    def __init__(self, word1, model1=None, model2=None, collection=None, alignment=None):
        if alignment is None:
            self.alignment = Alignment(model1, model2, collection)
        else:
            self.alignment = alignment
        self.word1 = word1
        self.word2 = None
        self.solve()
        return

    def solve(self):
        debug("Solving analogy...")
        vec1 = self.alignment.model1[self.word1]
        vec3 = self.alignment.model3[self.word1]
        self.neighbors1 = self.alignment.model1.most_similar(vec1)
        self.neighbors2 = self.alignment.model2.most_similar(vec3)
        self.word2 = self.neighbors2[0][0]
        return

    def __str__(self):
        return "{:} : {:} :: {:} : {:}".format(self.alignment.model1.name, self.word1,
                                               self.alignment.model2.name, self.word2)

    def __repr__(self):
        return "<Analogy {:} {:}->{:}>".format(repr(self.word1), self.model1.name, self.model2.name)


def fit_w2v_regression(model1, model2, samplesize=0.5):
    """Given two gensim Word2Vec models, fit a regression model using a subset of the vocabulary.
    The size of this subset is given by the samplesize parameter, which can specify either a
    percentage of the common vocab to use, or the number of words to use.

    ::param model1:: a gensim `KeyedVectors` instance for the LHS
    ::param model2:: a gensim `KeyedVectors` instance for the RHS
    ::param samplesize:: a float or int specifying how much of the vocab to use to fit the model.
    ::returns:: a `sklearn.linear_model.LinearRegression` object.
    """
    common_vocab = set(model1.vocab.keys()).intersection(set(model2.vocab.keys()))
    if "</s>" in common_vocab:
        common_vocab.remove("</s>")
    debug("{:,} words in model 1".format(len(model1.vocab)))
    debug("{:,} words in model 2".format(len(model2.vocab)))
    debug("{:,} words common to both models".format(len(common_vocab)))
    if type(samplesize) == float:
        samplesize = int(samplesize * len(common_vocab))
    debug("Sampling {:,} words from the common vocab".format(samplesize))
    d1 = model1.vector_size
    d2 = model2.vector_size
    sample = random.sample(common_vocab, samplesize)
    X = np.ndarray((samplesize, d1), dtype=np.float32)
    Y = np.ndarray((samplesize, d2), dtype=np.float32)
    # Pretty sure this loop is not an efficient way to sample things...
    for i, word in enumerate(sample):
        X[i, :] = model1[word]
        Y[i, :] = model2[word]
    debug("Fitting linear regression with {:,} samples".format(samplesize))
    regression = LinearRegression()
    regression.fit(X, Y)
    return regression

def apply_wv2_regression(model, regression):
    """Given a word2vec model and a linear regression, apply that regression to all the vectors
    in the model.

    ::param model:: A gensim `KeyedVectors` or `Word2Vec` instance
    ::param regression:: A `sklearn.linear_model.LinearRegression` instance
    ::returns:: A gensim `KeyedVectors` instance
    """
    debug("Applying transformation")
    model_t = KeyedVectors() # Word2Vec()
    model_t.wv.vocab = model.vocab.copy()
    model_t.wv.vector_size = model.vector_size
    model_t.wv.index2word = model.index2word
    # model_t.reset_weights()
    debug("Transforming {:,} vectors".format(len(model.syn0)))
    # N.B. Somehow I get float64 here and that's not what I want, so I'm explicitly casting to float32
    model_t.syn0 = regression.predict(model.syn0).astype(np.float32)
    return model_t
