"""
This is the `twapy` module for computing temporal word analogies in Python (twa+py).

The code in this module facilitates aligning and comparing independent vector space models. This
module does not help with the task of training those models: you can use the `gensim` module,
the original word2vec implementation, or any other library to build word embeddings provided they
are in a format that can be read by gensim.

A temporal word analogy is a pair of words that have similar meanings at different points in
time. For exapmle, the word "reagan" in the year 1987 is like the word "clintion" in the year
1997, since both words reference the current sitting United States president. Assuming you have
trained two word2vec models, one trained on a corpus of 1987 text and one trained on a corpus of
1997 text, you can construct a word analogy with twapy like so:

>>> import twapy
>>> analogy = twapy.Analogy('reagan', 'models/1987.bin', 'models/1997.bin')
>>> analogy.word2
'clinton'

Alternatively, if you are interested in computing multiple word analogies for the same pair of
years, you can construct an `Alignment` object and use it to compute the analogies:

>>> alignment = twapy.Alignment('models/1987.bin', 'models/1997.bin')
>>> analogy = twapy.Analogy('reagan', alignment=alignment)
>>> analogy.word2
'clinton'
>>> analogy = twapy.Analogy('koch', alignment=alignment)
>>> analogy.word2
'giuliani'

If you want to compute analogies for many pairs of models, you can place your all of your
pre-trained embedding models together in a single directory, and access them using the
ModelCollection class.

>>> collection = twapy.ModelCollection(directory='models')
>>> analogy = twapy.Analogy('reagan', '1987', '1997', collection=collection)
>>> analogy.word2
'clinton'

Using a ModelCollection, you can easily automate tasks like looking at the analogues of a given
word over a large set of different models.

This module also includes a simple web server for interacting with these objects and solving
temporal word analogies using a browser-based interface. To launch the web server,
run the `runserver` script (there is a bash script for Mac/Linux, and a .bat script for Windows).

"""

__author__ = 'Terrence Szymanski'
__verion__ = '0.1'
__date__ = '2017-07-30'

import logging

###############################################################################
# LOGGING
#
# Configure logging. This is currently configured in a very simplistic
# way but it should be done better. All submodules should just import
# the logging functions from here, to enable a centralized
# configuration. (e.g. `from twapy import error, warn, info, debug`)
###############################################################################

logfile = None  # Change this to log to a file instead of the console
loglevel = 'DEBUG'  # Change this if desired. loglevel = 100 will disable logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logfile, level=loglevel)
log = logger.log
error = logger.error
warn = logger.warn
info = logger.info
debug = logger.debug


###############################################################################
# Top-level imports
#
# All of the `twapy` methods to be available via top-level imports
###############################################################################

from .models import ModelCollection, VectorSpaceModel
from .alignment import Alignment, Analogy
from .evaluate import Evaluation, score_evaluation_file
