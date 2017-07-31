"""This module handles the evaluation of temporal word analogies.

The `run_evaluation.py` script uses this module to run an evaluation over all pairs of models. The
output of running this script is the 'evaluation_output.tsv' file, which is included in the
github repository.

The `score_evaluation.py` script uses this module to score the accuracy of the outputs generated
by the `run_evaluation.py` script.



"""


import os
import pandas as pd
import random

from .models import ModelCollection
from .alignment import Alignment


class EvaluationError(Exception):
    pass


class Evaluation(object):

    """This is a class to run an evaluation of a set of models against a ground truth set of analogies.

    """

    def __init__(self, evals_fn, models_dir="", samplesize=0.5, output_fn="evaluation_output.tsv"):
        self.samplesize = samplesize
        self.collection = ModelCollection(models_dir)
        self.e = pd.read_csv(evals_fn, encoding="utf-8", index_col=0)
        # Indexes might look like ints (e.g. 1987), but treat them as strings:
        self.e.index = self.e.index.astype("str")
        # Remove any models that don't exist.
        self.e = self.e.loc[self.collection.modelnames]
        self.output_fn = output_fn
        if os.path.exists(output_fn):
            # Don't want to accidentally overwrite an existing file.
            raise EvaluationError("Output file already exists.")
        return


    def evaluate_all(self):
        print("Evaluation all pairs of:", self.e.index.values)
        for mn1 in self.e.index.values:
            for mn2 in self.e.index.values:
                if mn1 == mn2:
                    continue
                self.evaluate_pair(mn1, mn2)


    def evaluate_all_from(self, mn1):
        print("Evaluation all pairs from {:} to {:}".format(
            mn1, self.e.index.values))
        for mn2 in self.e.index.values:
            if mn1 == mn2:
                continue
            self.evaluate_pair(mn1, mn2)


    def evaluate_sample(self, n=10):
        sample = self.sample(n)
        for mn1, mn2 in sample:
            self.evaluate_pair(mn1, mn2)
        return


    def evaluate_pair(self, mn1, mn2):
        print("Evaluating {:} -> {:}".format(mn1, mn2))
        a = Alignment(mn1, mn2, collection=self.collection, samplesize=self.samplesize)
        with open(self.output_fn, "a", encoding="utf-8") as f:
            for col in self.e.columns:
                w1 = self.e.loc[mn1, col]
                w2 = self.e.loc[mn2, col]
                try:
                    w2_predicted = a.analogy(w1).word2
                except Exception as e:
                    print("WARNING! Something is wrong.")
                    print(e)
                    # Something happened
                    w2_predicted = "ERROR"
                f.write("\t".join([mn1, mn2, w1, w2, w2_predicted]) + "\n")
        return


    def sample(self, n=10):
        """Generate a sample of distinct pairs of models. 
        """
        sample = set()
        models = list(self.e.index.values)
        max_n = len(models)**2 - len(models)
        if n > max_n:
            print("Warning: the specified sample size is greater than the population size. Automatically reducing the sample size.")
            n = max_n
        while len(sample) < n:
            mn1 = random.choice(models)
            mn2 = mn1
            while mn1 == mn2:
                mn2 = random.choice(models)
            sample.add((mn1, mn2))
        return list(sample)


def score_evaluation_file(filename, min_diff=None, max_diff=None):
    """Once an evaluation file has been produced, this will summarize the
    results and compute the accuracy."""
    df = pd.read_table(filename, index_col=None, header=None, encoding="utf-8")
    df.columns = ['year1', 'year2', 'original', 'gold', 'predicted']
    if min_diff is not None:
        df = df[(df.year2-df.year1).abs() >= min_diff]
    if min_diff is not None:
        df = df[(df.year2-df.year1).abs() <= min_diff]
    correct = (df.gold == df.predicted).sum()
    baseline = (df.gold == df.original).sum()
    agree = (df.predicted == df.original).sum()
    null = (df.predicted=="NONE").sum()
    total = len(df)
    accuracy = 100.0 * correct / total
    nullpct = 100.0 * null / total
    basepct = 100.0 * baseline / total
    agreepct = 100.0 * agree /total
    print("{:5.1f}% ({:} out of {:}) correctly predicted.".format(
        accuracy, correct, total))
    print("{:5.1f}% ({:} out of {:}) predicted by baseline.".format(
        basepct, baseline, total))
    print("{:5.1f}% ({:} out of {:}) prediction agrees with baseline.".format(
        agreepct, agree, total))
    print("{:5.1f}% ({:} out of {:}) had no prediction.".format(
        nullpct, null, total))
    return correct, baseline, agree, null, total


def accuracy_over_time(filename):
    df = pd.read_table(filename, index_col=None, header=None, encoding="utf-8")
    df.columns = ['year1', 'year2', 'original', 'gold', 'predicted']
    spread = df.year1.max() - df.year1.min()
    ts = pd.DataFrame(index=range(1, spread))
    ts["prediction"] = 0.0
    ts["baseline"] = 0.0
    for i in range(1, spread):
        c, b, a, n, t = score_evaluation_file(filename, min_diff=i, max_diff=i)
        ts.loc[i, "prediction"] = c / t
        ts.loc[i, "baseline"] = b / t
    return ts
