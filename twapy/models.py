"""Vector-space models for temporal word analogies.

This module defines two main classes:

 * `VectorSpaceModel` is a single VSM and is mainly just a wrapper around the Gensim
 `KeyedVectors` class. It can read/write models from/to disk, and this class is used when fitting
 transformations across models.

 * `ModelCollection` is a container of VSMs. It helps manage a group of models contained in the
 specified directory, but it will not load all models into memory at once.

For example, below are two different ways of loading the 1987 model from disk:

>>> from twapy.models import VectorSpaceModel, ModelCollection
>>> collection = ModelCollection('/path/to/models')
>>> model = collection['1987']
>>> model = VectorSpaceModel.load('/path/to/models/1987.bin')

"""

import os
import pickle
import re

from twapy import info, debug, warn


try:
    from gensim.models import KeyedVectors
except:
    warn("Gensim import failed. Please ensure that gensim verison 2 or greater is installed.")
    import sys
    sys.exit(1)


class VectorSpaceModel(object):

    """Base class for models that represent words as vectors.

    For now, this really is just a wrapper around the Gensim KeyedVectors / Word2Vec class.

    """

    def __init__(self, name=None):
        self.name = name
        self.m = KeyedVectors()
        return

    @classmethod
    def load(cls, filename, modelname=None, **kwargs):
        if filename.endswith('.pkl'):
            model = cls.load_pickle(filename, modelname=modelname, **kwargs)
        else:
            model = cls.load_w2v(filename, modelname=modelname, **kwargs)
        return model

    @classmethod
    def load_pickle(cls, filename, **kwargs):
        debug("Loading pickled model from file {:}".format(filename))
        model = pickle.load(filename)
        return model

    @classmethod
    def load_w2v(cls, filename, modelname=None, **kwargs):
        """Load the model from disk."""
        debug("Loading word2vec model from file {:}".format(filename))
        if filename.endswith(".bin"):
            m = KeyedVectors.load_word2vec_format(filename, binary=True)
        else:
            m = KeyedVectors.load_word2vec_format(filename)
        model = cls()
        model.m = m
        if modelname is None:
            modelname = os.path.basename(filename)
            modelname = re.sub('.bin', '', modelname)
        model.name = modelname
        return model

    def save_pickle(self, filename):
        debug("Saving model {:} to pickle file {:}".format(self.name, filename))
        pickle.dump(self, filename)
        return

    def __getitem__(self, word):
        return(self.m[word])

    def most_similar(self, query, k=5):
        """Return the most similar words to the query. `query` can be either a string or a
        vector. If it is a string, then its vector will be looked up in the current VSM.
        """
        if type(query) is str:
            results = self.m.most_similar(query, topn=k)
        else:
            results = self.m.similar_by_vector(query, topn=k)
        return results

    def __repr__(self):
        return "<VectorSpaceModel {:} with {:,} vectors>".format(repr(self.name), self.m.syn0.shape[0])


# class Word2VecModel(VectorSpaceModel):
#
#     """Vector space model based on word2vec vectors.
#
#     This is mainly just a wrapper around the `gensim` `Word2Vec`
#     class.
#
#     """
#
#     def __init__(self, model, binary=True):
#         if gm and type(model) == gm.Word2Vec:
#             self.m = model
#         elif type(model) == str:
#             self.m = gm.KeyedVectors.load_word2vec_format(
#                 model, binary=binary)
#         return
#
#     @classmethod
#     def load(cls, filename):
#         binary = False
#         if filename.endswith(".bin"):
#             binary = True
#         model = KeyedVectors.load_word2vec_format(modelpath, binary=binary)


# class BaselineModel(VectorSpaceModel):
#
#     def most_similar(self, query):
#         return query
#
#
# class CooccurrenceModel(VectorSpaceModel):
#
#     """Vector space model using word co-occurrence counts as the basis for
#     the vectors.
#
#     """
#
#     def __init__(self):
#         return
#

class ModelCollection(object):

    """Collection of vector space models.

    This can be instantiated from a directory containing multiple vector space models. This class
    facilitates any operations that involve more than one vector space model.

    """

    def __init__(self, directory=None, lazy=True):

        self._models = {}
        self._directory = None
        self._lazy = lazy

        if directory is not None:
            self._load_directory(directory, lazy=lazy)

        return

    def add_model(self, model, modelname=None):
        if modelname is None:
            try:
                modelname = model.name
            except AttributeError:
                i = 0
                while True:
                    modelname = "model{:d}".format(i)
                    if modelname not in self._models.keys():
                        break
                    i += 1
        if modelname in self._models.keys():
            warn("Overwriting existing model '{:}'.".format(modelname))
        self._models[modelname] = model
        return

    def _load_directory(self, directory, lazy=True):
        '''Load all of the models from the given directory.

        If `lazy` is True, then this will simply load the model names, but it won't load the
        models themselves. Instead, the file paths will be stored and the models will be loaded
        on-demand as needed. This is very useful (actually, essential) when working with
        directories containing many models.
        '''
        debug("Loading models from directory {:}".format(directory))
        for fn in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fn)
            modelname = fn.rsplit(".", 1)[0]
            if lazy:
                debug("Lazy-loaded model from file {:}".format(fn))
                model = fpath
            else:
                try:
                    debug("Loaded model from file {:}".format(fn))
                    model = load_model(fpath, lazy=lazy)
                except:
                    warn("Unable to load model from file {:}".format(fn))
                    model = None
            self._models[modelname] = model
        debug("Loaded {:d} models: {:}".format(len(self._models), list(self._models.keys())))
        return

    def __getitem__(self, modelname):
        model = self._models[modelname]
        if type(model) == str:
            model = VectorSpaceModel.load(filename=model, modelname=modelname)
        return model

    @property
    def modelnames(self):
        return sorted(self._models.keys())
