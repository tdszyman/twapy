# twapy: Temporal Word Analogies in Python

This package contains Python code to build and evaluate models for studying temporal word analogies: pairs of words from
 different points in time that have similar meanings. An example of a temporal word analogy is:

    "Ronald Reagan" in 1987 is like "Bill Clinton" in 1997
    
In this example the strings `'Ronald Reagan'` and `'Bill Clinton'` both represent the semantic concept `president of the
 United States` at different points in time, and thus they constitute a temporal word analogy.

A discussion of the approach implemented in this package and the types of results it generates will appear at the 2017
Annual Meeting of the Association for Computational Linguistics (ACL 2017). If you use this code, please cite the
following paper:

> Terrence Szymanski. 2017. *Temporal Word Analogies: Identifying Lexical Replacement with Diachronic Word Embeddings*. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).

    @InProceedings{szymanski:2017,
    author    = {Szymanski, Terrence},
    title     = {Temporal Word Analogies: Identifying Lexical Replacement with Diachronic Word Embeddings},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
    month     = {August},
    year      = {2017},
    address   = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics}
    }

## Requirements

* Python 3
* gensim
* sklearn
* pandas (for the evaluation scripts)

This package was developed with Python 3.6, gensim version 2.2, sklearn version 0.18, and pandas version 0.20. It will
probably work with other versions of those packages, with the exception that it will probably not work with gensim
versions prior to version 2.0.

## Quickstart Instructions

1. Clone this repository to your computer.
2. Run the `download_models.py` script to download two example embedding models.
3. Run the `run_example.py` script to see that everything is working.
4. Run the `run_server.sh` (or `runserver.bat`) script to launch the web interface.
