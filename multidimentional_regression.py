import numpy as np
from typing import List
from numpy import floating





test_squares = np.array([1450, 2200, 1800, 950, 2700,
                1300, 2500, 1600, 2100, 1900,
                1750, 2300, 1100, 2850, 1500,
                2400, 1650, 1950, 2250, 1250])

test_bathrooms = np.array([2, 3, 2, 1, 3,
                 2, 3, 2, 2, 2,
                 2, 3, 1, 3, 2,
                 3, 2, 2, 3, 1])

test_years = np.array([1995, 2010, 2005, 1985, 2018,
              2000, 2015, 1998, 2008, 2002,
              2003, 2012, 1990, 2020, 1997,
              2016, 2001, 2006, 2011, 1993])

test_prices = np.array([325000, 485000, 375000, 195000, 585000,
               275000, 525000, 315000, 425000, 355000,
               345000, 495000, 225000, 615000, 295000,
               505000, 335000, 365000, 455000, 255000])

test_features = np.column_stack([test_squares, test_years, test_bathrooms, np.ones(test_squares.shape)])



def predict(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    return data @ weights

def squared_error(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    diff = labels - predict(weights, features)
    return np.mean(diff**2)

def gradient_descent(weights: np.ndarray, features: np.ndarray, labels: np.ndarray, step: float, lambd: int, k: int):
    l1_gradient = np.where(weights > 0, 1, -1)
    l2_gradient = weights
    gradient = -2/len(labels) * (features.T @ (labels - predict(weights, features))) + lambd*k*l1_gradient + lambd*(1-k)*l2_gradient
    weights -= step*gradient


def linear_regression(features: np.ndarray, labels: np.ndarray, step: float, steps: int, lambd: int = 0, k: int = 1) -> np.ndarray:
    weights = np.random.uniform(-1, 1, size = features.shape[1])
    features_mean = features.mean(axis=0)
    features_std = np.where(features.std(axis=0) == 0, 1, features.std(axis=0))
    labels_mean = labels.mean()
    labels_std = labels.std() if labels.std() != 0 else 1
    features = (features - features_mean) / features_std
    labels = (labels - labels_mean) / labels_std
    for s in range(steps):
        gradient_descent(weights, features, labels, step, lambd, k)
        if s%1000 == 0:
            print(squared_error(weights, features, labels))
    feature_weights = weights[:-1].copy()
    weights[:-1] = weights[:-1]*(labels_std/features_std[:-1])
    weights[-1] = labels_mean + labels_std * (weights[-1] - np.sum(feature_weights / features_std[:-1]))
    return weights

if __name__ == '__main__':
    result = linear_regression(test_features, test_prices, 0.001, 10000)
    print(predict(result, test_features))