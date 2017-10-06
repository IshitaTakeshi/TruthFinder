import numpy as np
from numpy.linalg import norm
import math
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.6108&rep=rep1&type=pdf

import warnings
warnings.filterwarnings("error")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TruthFinder(object):
    def __init__(self, implication,
                 dampening_factor=0.3, influence_related=0.5):
        """
        implication: function taking two arguments
            implication(f1, f2) should return
            `imp(f1 -> f2) <- [-1, 1]` in the original paper
        dampening_factor:
            gamma <- (0, 1) in the original paper
        influence_related:
            rho <- [0, 1] in the original paper
        """

        assert(0 < dampening_factor < 1)
        assert(0 <= influence_related <= 1)
        self.implication = implication
        self.dampening_factor = dampening_factor
        self.influence_related = influence_related

    def adjust_confidence(self, df):
        """Eq. 6"""

        update = {}
        for i, row1 in df.iterrows():
            f1 = row1["fact"]
            s = 0
            for j, row2 in df.drop_duplicates("fact").iterrows():
                f2 = row2["fact"]
                if f1 == f2:
                    continue
                s += row2["fact_confidence"] * self.implication(f2, f1)
            update[i] = self.influence_related * s + row1["fact_confidence"]

        for i, row1 in df.iterrows():
            df.set_value(i, "fact_confidence", update[i])

        return df

    def calculate_confidence(self, df):
        trustworthiness_score = lambda x: -math.log(1-x)  # Eq. 3

        """Calculate confidence for each fact"""
        for i, row in df.iterrows():
            # Eq. 5
            # trustworthiness of corresponding websites `W(f)`
            ts = df.loc[df["fact"] == row["fact"], "trustworthiness"]
            v = sum(trustworthiness_score(t) for t in ts)
            df.set_value(i, "fact_confidence", v)
        return df

    def compute_fact_confidence(self, df):
        f = lambda x: sigmoid(self.dampening_factor * x)
        for i, row in df.iterrows():
            df.set_value(i, "fact_confidence", f(row["fact_confidence"]))
        return df

    def update_fact_confidence(self, df):
        for object_ in df["object"].unique():
            indices = df["object"] == object_
            d = df.loc[indices]
            d = self.calculate_confidence(d)
            d = self.adjust_confidence(d)
            df.loc[indices] = self.compute_fact_confidence(d)
        return df

    def update_website_trustworthiness(self, df):
        for website in df["website"].unique():
            indices = df["website"] == website
            cs = df.loc[indices, "fact_confidence"]
            df.loc[indices, "trustworthiness"] = sum(cs) / len(cs)
        return df

    def iteration(self, df):
        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df

    def train(self, dataframe, n_iterations, initial_trustworthiness):
        dataframe["trustworthiness"] =\
                np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))

        for i in range(n_iterations):
            self.iteration(dataframe)
        return dataframe
