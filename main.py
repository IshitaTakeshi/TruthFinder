import csv
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from truthdiscovery import TruthFinder

# with open("book.txt", "r") as f:
#     reader = csv.reader(f, delimiter="\t")
#     lines = [line for line in reader if len(line) == 4]
# dataframe = pd.DataFrame(lines, columns=["website", "isbn", "object", "fact"])

dataframe = pd.DataFrame([
        ["a", "Einstein", "Special relativity"],
        ["a", "Newton", "Universal gravitation"],
        ["b", "Einstein", "Special relativity"],
        ["b", "Galilei", "Heliocentrism"],
        ["c", "Newton", "Special relativity"],
        ["c", "Galilei", "Universal gravitation"],
        ["c", "Einstein", "Heliocentrism"]
    ],
    columns=["website", "fact", "object"]
)


vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit(dataframe["fact"])


def similarity(f1, f2):
    V = vectorizer.transform([f1.lower(), f2.lower()])
    v1, v2 = np.asarray(V.todense())
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def implication(f1, f2):
    return similarity(f1, f2)


finder = TruthFinder(implication)

dataframe = finder.train(dataframe, 4, 0.9)

print(dataframe)
