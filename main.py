import csv
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from truthdiscovery import TruthFinder


dataframe = pd.DataFrame([
        ["a", "Einstein", "Special relativity"],
        ["a", "Newton", "Universal gravitation"],
        ["b", "Albert Einstein", "Special relativity"],
        ["b", "Galileo Galilei", "Heliocentrism"],
        ["c", "Newton", "Special relativity"],
        ["c", "Galilei", "Universal gravitation"],
        ["c", "Einstein", "Heliocentrism"]
    ],
    columns=["website", "fact", "object"]
)


vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit(dataframe["fact"])


def similarity(w1, w2):
    V = vectorizer.transform([w1, w2])
    v1, v2 = np.asarray(V.todense())
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def implication(f1, f2):
    return similarity(f1.lower(), f2.lower())


finder = TruthFinder(implication, dampening_factor=0.8, influence_related=0.6)

print("Inital state")
print(dataframe)
dataframe = finder.train(dataframe)

print("Estimation result")
print(dataframe)
