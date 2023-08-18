import pandas as pd
import statsmodels.api as sm
import warnings
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class PortfolioModelingManagement:
    def __init__(self, LoanData, responseVariable):
        self.LoanData = LoanData
        self.responseVariable = responseVariable

        self.tuningVariableList = {
            "categoricalList": [],
            "standardizedVariableList": [],
            "interaction": [],
        }

        self.covariateList = {"base": [],"test": []}

        self.RegressionResults = {
            "dummy": {
                "covariate": [],
                "categorical": [],
                "standardized": [],
                "modelRes": None,
                "notes": "dummy model",
            },
        }

        self.baseRes, self.testRes = None, None
