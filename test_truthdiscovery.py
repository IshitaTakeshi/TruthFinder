import math
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd

from truthdiscovery import TruthFinder


class TestTruthFinder(unittest.TestCase):
    def setUp(self):
        def implication(f1, f2):
            return 1

        self.dampening_factor = 0.1
        self.influence_related = 0.5
        self.finder = TruthFinder(
                implication,
                dampening_factor=self.dampening_factor,
                influence_related=self.influence_related)
        self.dataframe = pd.DataFrame([
            ["a", "Einstein", "Special relativity"],
            ["a", "Newton", "Universal gravitation"],
            ["b", "Einstein", "Special relativity"],
            ["b", "Galilei", "Heliocentrism"],
            ["c", "Newton", "Special relativity"],
            ["c", "Galilei", "Universal gravitation"],
            ["c", "Einstein", "Heliocentrism"]
        ],
        columns=["website", "fact", "object"])

        initial_trustworthiness = 0.9

        self.dataframe["trustworthiness"] =\
                np.ones(len(self.dataframe.index)) * initial_trustworthiness

        self.dataframe["fact_confidence"] = [
            0.5, 0.3, 0.6, 0.8, 0.1, 0.0, 0.8
        ]

    def test_calculate_confidence(self):
        df = self.dataframe.copy()

        df = df[df["object"] == "Special relativity"]

        df = self.finder.calculate_confidence(df)

        A1 = -2 * math.log(0.1)
        A2 = -math.log(0.1)

        assert_array_almost_equal(
            df[df["fact"] == "Einstein"]["fact_confidence"],
            A1
        )
        assert_array_almost_equal(
            df[df["fact"] == "Newton"]["fact_confidence"],
            A2
        )

        df = self.finder.adjust_confidence(df)

        C1 = A1 + self.influence_related * A2
        C2 = A2 + self.influence_related * A1

        indices1 = np.logical_and(
            df["object"] == "Special relativity",
            df["fact"] == "Einstein"
        )
        indices2 = np.logical_and(
            df["object"] == "Special relativity",
            df["fact"] == "Newton"
        )

        assert_array_almost_equal(df[indices1]["fact_confidence"], C1)
        assert_array_almost_equal(df[indices2]["fact_confidence"], C2)

        df = self.finder.compute_fact_confidence(df)

        assert_array_almost_equal(
            df[indices1]["fact_confidence"],
            1 / (1 + math.exp(-self.dampening_factor * C1))
        )
        assert_array_almost_equal(
            df[indices2]["fact_confidence"],
            1 / (1 + math.exp(-self.dampening_factor * C2))
        )

    def test_update_website_trustworthiness(self):
        df = self.dataframe.copy()

        df = self.finder.update_website_trustworthiness(df)

        assert_array_equal(
            df.loc[df["website"] == "a", "trustworthiness"],
            0.4
        )

        assert_array_equal(
            df.loc[df["website"] == "b", "trustworthiness"],
            0.7
        )

        assert_array_equal(
            df.loc[df["website"] == "c", "trustworthiness"],
            0.3
        )


if __name__ == "__main__":
    unittest.main()
