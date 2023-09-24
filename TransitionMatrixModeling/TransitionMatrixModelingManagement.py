# use applied logistic regression book method
import pandas as pd
import statsmodels.api as sm
import warnings
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class TransitionMatrixModelingManagement:
    def __init__(self, LoanData, responseVariable):
        self.LoanData = LoanData
        self.responseVariable = responseVariable

        self.tuningVariableList = {
            "categoricalList": [],
            "interaction": [],
        }

        self.covariateList = {
            "base": [],
            "test": [],
        }

        self.RegressionResults = {
            "dummy": {
                "covariate": [],
                "categorical": [],
                "modelRes": None,
                "notes": "dummy model",
            },
        }

        self.baseRes, self.testRes = None, None  # placeholder of base and test result

    def runTrainingTest(self):
        X = self.genX(self.covariateList["base"])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            self.LoanData[[self.responseVariable]],
            test_size=0.33,
            random_state=42,
        )
        # using statsmodel
        # model = sm.MNLogit(self.y_train, self.X_train).fit(method="bfgs", maxiter=1000)
        # y_prob_predict = model.predict(self.X_test)
        # lr_auc = roc_auc_score(self.y_test, y_prob_predict, multi_class="ovr")

        # using sklearn
        sklearnLRModel = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=1000000,
        )
        model = sklearnLRModel.fit(self.X_train, self.y_train)
        y_prob_predict = model.predict_proba(self.X_test)

        ovo_auc = roc_auc_score(
            np.squeeze(self.y_test),
            y_prob_predict,
            multi_class="ovo",
            average="macro",
        )
        ovr_auc = roc_auc_score(
            np.squeeze(self.y_test), y_prob_predict, multi_class="ovr", average="macro"
        )
        self.trainingTestRes = {
            "trainingModelRes": model,
            "AUC(ovo)": ovo_auc,
            "AUC(ovr)": ovr_auc,
        }

        return self

    def appendRegressionResults(self, resName, resNotes):
        if resName not in self.RegressionResults.keys():
            self.RegressionResults[resName] = {
                "covariate": self.covariateList["base"].copy(),
                "categorical": self.tuningVariableList["categoricalList"].copy(),
                "modelRes": self.baseRes,
                "notes": resNotes,
            }
        else:
            raise ("resName already exists")
        return self

    def calcChi2(self, model1, model2):
        # by default, assuming model2 is the model with more variable. could be base model or test model
        g2 = -2 * (model1.llf - model2.llf)
        p_value = 1 - scipy.stats.chi2.cdf(g2, model2.df_model - model1.df_model)
        model1ParamsDf, model2ParamsDf = model1.params, model2.params
        commonParams = model1ParamsDf.index.intersection(model2ParamsDf.index)

        model2ExtraParams = model2ParamsDf.index.difference(commonParams)
        model1ExtraParams = model1ParamsDf.index.difference(commonParams)

        if len(model2ExtraParams) * len(model1ExtraParams) > 0:
            # if both model has extra parameters
            confoundingImpact = "NA. model1 and model2 parameters difference is not subset of each other"

        elif len(model2ExtraParams) + len(model1ExtraParams) == 0:
            # if both model has no extra parameters
            confoundingImpact = "NA. model1 and model2 parameters are the same"

        else:
            # if only one model has extra parameters

            if len(model2ExtraParams) > 0:
                extraParams = model2ExtraParams
                baseParamsDf = model1ParamsDf
                alternativeParamsDf = model2ParamsDf

            if len(model1ExtraParams) > 0:
                extraParams = model1ExtraParams
                baseParamsDf = model2ParamsDf
                alternativeParamsDf = model1ParamsDf

            cols = baseParamsDf.columns

            confoundingImpact = pd.DataFrame(
                data=(
                    np.array(baseParamsDf.loc[commonParams, cols])
                    - np.array(alternativeParamsDf.loc[commonParams, cols])
                )
                / np.array(alternativeParamsDf.loc[commonParams, cols]),
                index=commonParams,
                columns=cols,
            )
            for item in extraParams:
                confoundingImpact.loc[item] = np.nan

        return {
            "g2": g2,
            "p_value": p_value,
            "confoundingImpact": confoundingImpact,
        }

    def runAgainstVariate(self, baseModelInput, testCovariate, remove=False):
        baseModel = baseModelInput["modelRes"]
        baseModelCovariate = baseModelInput["covariate"]

        self.covariateList["test"] = baseModelCovariate.copy()
        if remove:
            self.covariateList["test"].remove(testCovariate)
        else:
            self.covariateList["test"].append(testCovariate)

        self.runModel(targetCovariateGroup="test")

        if remove:
            model1 = self.testRes
            model2 = baseModel

        else:
            model1 = baseModel
            model2 = self.testRes

        return {
            "baseModel": baseModelInput,
            "alternativeModel": {
                "modelRes": self.testRes,
                "covariate": self.covariateList["test"],
            },
            "stats": self.calcChi2(model1, model2),
        }

    def poolingCategorical(self, col, newCol, map, setCategorical=True):
        self.LoanData[newCol] = self.LoanData[col].map(map)
        if setCategorical:
            self.updateInputList(
                {"add": [newCol]}, "tuningVariableList", "categoricalList"
            )

        return self

    def loopThroughUnitvariate(self):
        # generate univariate model of each varaible
        self.univariateRunSummary = pd.DataFrame(
            columns=["Variable", "p-value", "modelRes"]
        )
        # run all categorical and numeric
        for x in self.covariateList["base"]:
            self.setInputList([x], "covariateList", "test")
            self.runModel(targetCovariateGroup="test")

            self.univariateRunSummary.loc[len(self.univariateRunSummary), :] = [
                x,
                self.testRes.llr_pvalue,
                self.testRes,
            ]


        return self

    def addInteraction(self):
        for interactionTerms in self.tuningVariableList["interaction"]:
            term1 = interactionTerms[0]
            term2 = interactionTerms[1]
            crossTerm = term1 + " x " + term2
            if crossTerm not in self.LoanData.columns:
                self.LoanData.loc[:, crossTerm] = (
                    self.LoanData.loc[:, term1] * self.LoanData.loc[:, term2]
                )

        return self

    def genX(self, covariateList):
        X = self.LoanData[covariateList]

        for item in X.columns:
            if item in self.tuningVariableList["categoricalList"]:
                X.loc[:, item] = X.loc[:, item].astype("category")
                X = pd.get_dummies(X, columns=[item], drop_first=True)

        X.loc[:, "Intercept"] = 1
        return X

    def displayCoef(self, modelRes):
        pvalDf, paramsDf, oddsratioDf = (
            modelRes.pvalues,
            modelRes.params,
            np.exp(modelRes.params),
        )
        print("=" * 60, "P-Value", "=" * 60)
        print(pvalDf)
        print("=" * 60, "Coefficients", "=" * 60)
        print(paramsDf)
        print("=" * 60, "Odds Ratio", "=" * 60)
        print(oddsratioDf)

        return self

    def setInputList(self, inputList, targetVar, targetGroup):
        if targetVar == "covariateList":
            self.covariateList[targetGroup] = inputList
        elif targetVar == "tuningVariableList":
            self.tuningVariableList[targetGroup] = inputList

        return self

    def updateInputList(self, actionDict, targetVar, targetGroup):
        addAction = actionDict.get("add")
        removeAction = actionDict.get("remove")

        if addAction is not None:
            for item in actionDict.get("add"):
                if targetVar == "covariateList":
                    self.addVariable(item, self.covariateList[targetGroup])
                elif targetVar == "tuningVariableList":
                    self.addVariable(item, self.tuningVariableList[targetGroup])
        if removeAction is not None:
            for item in actionDict.get("remove"):
                if targetVar == "covariateList":
                    self.removeVariable(item, self.covariateList[targetGroup])
                elif targetVar == "tuningVariableList":
                    self.removeVariable(item, self.tuningVariableList[targetGroup])

        return self

    def addVariable(self, variable, variableList):
        if variable not in variableList:
            variableList.append(variable)
        return self

    def removeVariable(self, variable, variableList):
        if variable in variableList:
            variableList.remove(variable)
        return self

    def runInteractionTerms(self):
        # assuming all variables are in base
        # interaction terms are generated from base and stored in test
        self.runModel(targetCovariateGroup="base")
        baseRes = self.baseRes

        X = self.genX(self.covariateList["base"])
        X = X.drop(columns=["Intercept"])
        interaction = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        X_inter = interaction.fit_transform(X)
        X_inter_df = pd.DataFrame(
            data=X_inter,
            columns=interaction.get_feature_names_out(X.columns),
            index=X.index,
        )
        X_inter_df = X_inter_df.loc[
            :, [x for x in X_inter_df.columns if x not in X.columns]
        ]

        interactionCheck = pd.DataFrame(columns=["model", "llf", "g2", "pvalue"])
        interactionCheck.loc[len(interactionCheck)] = [
            "main effects",
            baseRes.llf,
            np.nan,
            np.nan,
        ]

        for interCol in X_inter_df.columns:
            self.setInputList(
                self.covariateList["base"] + [interCol], "covariateList", "test"
            )
            self.LoanData = self.LoanData.join(X_inter_df[[interCol]], how="left")

            self.runModel(targetCovariateGroup="test")

            model1, model2 = baseRes, self.testRes
            interactionStats = self.calcChi2(model1, model2)

            interactionCheck.loc[len(interactionCheck)] = [
                interCol,
                self.testRes.llf,
                interactionStats["g2"],
                interactionStats["p_value"],
            ]
            self.LoanData = self.LoanData.drop(columns=[interCol])

        return interactionCheck

    def runModel(self, covariateList=[], targetCovariateGroup='base'):
        # self.baseRes = None  # placeholder of base result
        # self.testRes = None  # placeholder of test result

        if targetCovariateGroup is not None:
            X = self.genX(self.covariateList[targetCovariateGroup])
            model = sm.MNLogit(self.LoanData[[self.responseVariable]], X).fit(
                method="bfgs", maxiter=1000, reference = 0
            )
            if targetCovariateGroup == "base":
                self.baseRes = None
                self.baseRes = model
            elif targetCovariateGroup == "test":
                self.testRes = None
                self.testRes = model
            else:
                raise Exception("targetCovariateGroup must be either base or test")
        else:
            raise Exception("need to specify targetCovariateGroup either base or test")
            # X = self.genX(covariateList)
            # model = sm.MNLogit(self.LoanData[[self.responseVariable]], X).fit(
            #     method="bfgs", maxiter=1000
            # )
            # return model

    def displayLogitOnSingle(self, continuousVariable, binsArg={"m": 4}):
        # display quartile plot of transitionTo against continuous variable
        bucketVar = continuousVariable + "Bucket"
        alternativeY = list(self.LoanData[self.responseVariable].unique())
        alternativeY.remove(0)
        graphData = self.LoanData[[continuousVariable, self.responseVariable]].copy()

        bins = binsArg.get("bins")
        if bins is None:
            bins = [
                np.quantile(graphData[continuousVariable], item / 100.0)
                for item in range(0, 100, round(100.0 / binsArg.get("m")))
            ] + [np.quantile(graphData[continuousVariable], 1)]

        graphData[bucketVar] = pd.cut(graphData[continuousVariable], bins=bins)
        temp = (
            pd.DataFrame(
                graphData.groupby([bucketVar])[[self.responseVariable]].value_counts(
                    normalize=True
                ),
                columns=["temp"],
            )
            .reset_index(drop=False)
            .pivot_table(
                index=bucketVar,
                columns=self.responseVariable,
                values="temp",
                aggfunc="sum",
            )
        )
        


        f, axs = plt.subplots(nrows=len(alternativeY), ncols=1, figsize=(18, 6), dpi=90)

        for i, yResponse in enumerate(alternativeY):
            logitDf = pd.DataFrame(
                np.log(temp.loc[:, yResponse] / temp.loc[:, 0]), columns=["logit"]
            )
            frenquencySeries = graphData[bucketVar].value_counts(normalize=True)
            frenquencyDf = pd.DataFrame(
                data=frenquencySeries.values,
                index=frenquencySeries.index,
                columns=["frquency"],
            )

            logitDf = logitDf.join(frenquencyDf, how="left")
            logitDf.plot(
                kind="line",
                y="logit",
                figsize=(10, 5 * len(alternativeY)),
                color="black",
                ax=axs[i],
            )
            axTwin = axs[i].twinx()
            logitDf.plot(
                kind="bar", y="frquency", ax=axTwin, rot=0, color="green", alpha=0.5
            )
            axs[i].set_ylabel("Logit")
            axTwin.set_ylabel("frequency")
            axs[i].set_title(
                "Logit plot of "
                + continuousVariable
                + "|"
                + str(yResponse)
                + " to "
                + str(0)
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return self
