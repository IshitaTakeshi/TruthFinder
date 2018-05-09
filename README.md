# TRUTHFINDER

This is an implementation of TRUTHFINDER: Truth discovery with multiple conflicting information providers on the web, which is a model for finding true facts from a large amount of conflicting information.  

TRUTHFINDER can estimate:

* Trustworthiness of information providers (e.g. websites)
* Confidence of information which are claimed as facts by these information providers.

# Usage
As an example, consider that list of theorems and the names of the dircoverers are provided by multiple websites and we estimate the trustworthiness both of the websites and the provided information.  

| website (information provider) | fact (discoverer) |      object (theorem) |
|:-------------------------------|:------------------|:----------------------|
|                              a |          Einstein |    Special relativity |
|                              a |            Newton | Universal gravitation |
|                              b |   Albert Einstein |    Special relativity |
|                              b |   Galileo Galilei |         Heliocentrism |
|                              c |            Newton |    Special relativity |
|                              c |           Galilei | Universal gravitation |
|                              c |          Einstein |         Heliocentrism |

This model works on data represented in `pandas.DataFrame`. So let's represent the data in it.  

```python
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
```

Before estimating, we have to define the `implication` between facts, which is explained later.  

```python
vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit(dataframe["fact"])


def similarity(f1, f2):
    V = vectorizer.transform([f1.lower(), f2.lower()])
    v1, v2 = np.asarray(V.todense())
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def implication(f1, f2):
    return similarity(f1, f2)
```

Then we can estimate the trustworthiness of these information.  

```python
finder = TruthFinder(implication, dampening_factor=0.8, influence_related=0.6)
```

__result__

| website | trustworthiness |             fact |                object | fact_confidence |
|:--------|:----------------|:-----------------|:----------------------|:----------------|
| a       |        0.862090 |         Einstein |    Special relativity | 0.894279        |
| a       |        0.862090 |           Newton | Universal gravitation | 0.829901        |
| b       |        0.862090 |  Albert Einstein |    Special relativity | 0.894279        |
| b       |        0.862090 |  Galileo Galilei |         Heliocentrism | 0.829901        |
| c       |        0.754878 |           Newton |    Special relativity | 0.754878        |
| c       |        0.754878 |          Galilei | Universal gravitation | 0.754878        |
| c       |        0.754878 |         Einstein |         Heliocentrism | 0.754878        |

As we can see in the table, website `c` and fact provided by it is less reliable than others.  

# Method
Suppose that some websites provide information on some objects.  
Fo example, three websites provide information on the object "Who discovered the law of universal gravitation?". Website A and website B claim that "Newton discovered", and Website C claims that "Einstein discovered", as shown in the figure.  

Then TRUTHFINDER estimates trustworthiness of websites and confidence of factsunder the assumption below:

_A fact is likely to be true if it is provided by trustworthy web sites. A web site is trustworthy if most facts it provides are true._

Although this assumption contains interdependence, the algorithm estimates the turstworthness and the confidence iteratively.  

## Implication
Different facts about the same object may be conflicting. However, sometimes facts may be supportive to each other.  

| website |              fact |                object |
|:--------|:------------------|:----------------------|
|       a |          Einstein |    Special relativity |
|       b |   Albert Einstein |    Special relativity |
|       c |            Newton |    Special relativity |

In this case, while the facts provided by `a` and `c` are conflicting, those provided by `a` and `b` are suppotive.  
In order to represent such relationships, the concept of implication between facts is proposed. 
The implication from fact f1 to f2 `implication(f1, f2)` represents how much f2’s confidence should be increased according to f1’s confidence

## Advantages over simple majority voting
Suppose the case that three websites are providing information on three objects as shown in the table.  

| website   | fact     | object                |
|:----------|:---------|:----------------------|
| A         | Einstein | Special relativity    |
| B         | Einstein | Special relativity    |
| C         | Newton   | Special relativity    |
| A         | Newton   | Universal gravitation |
| C         | Galilei  | Universal gravitation |
| B         | Galilei  | Heliocentrism         |
| C         | Einstein | Heliocentrism         |

While in majority voting the websites that provide information on a object are equally evaluated whether reliable or not, TRUTHFINDER evaluates information based on the trustworthiness of the websites.  

For example, because majority voting evaluates the facts about the object `Universal gravitation` by the number of websites supporting each fact, it cannnot determine which fact is true `Newton` or `Galilei`. But TRUTHFINDER can evaluate facts based on the trustworthiness of the websites providing them. In the table above we can see that obviously website C provides incorrect information. Because of the mechanism of TRUTHFINDER, it can be determined that the facts provided by website `c` is less reliable.  

# Reference
Yin, Xiaoxin, Jiawei Han, and S. Yu Philip. "Truth discovery with multiple conflicting information providers on the web." IEEE Transactions on Knowledge and Data Engineering 20.6 (2008): 796-808.
