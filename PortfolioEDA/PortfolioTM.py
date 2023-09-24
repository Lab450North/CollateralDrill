import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioTM:
    def __init__(self, portfolio, colNames):
        self.portfolio = portfolio
        self.colNames = colNames
        
    def plotStatusDist(self, col_name, continuous=True, excludeExtreme = {}):
        portfolioDisplay = self.portfolio.copy()

        if len(excludeExtreme) > 0:
            lowerBound = excludeExtreme.get('lowerBound', None)
            upperBound = excludeExtreme.get('upperBound', None)
            
            lowerBound = portfolioDisplay[col_name].quantile(lowerBound) if lowerBound else portfolioDisplay[col_name].min()
            upperBound = portfolioDisplay[col_name].quantile(upperBound) if upperBound else portfolioDisplay[col_name].max()
            
            portfolioDisplay = portfolioDisplay[(portfolioDisplay[col_name] >= lowerBound) & (portfolioDisplay[col_name] <= upperBound)]
        else:
            pass
            
        distinctStatus = list(sorted(portfolioDisplay[self.colNames['loanStatus']].unique()))
        countStatus = len(distinctStatus)
        if continuous:
            f, ax = plt.subplots(nrows=1, ncols = 3, figsize=(12, 3), dpi=90)
        else:
            f, ax = plt.subplots(
                nrows=1, ncols= 1 + countStatus, figsize=(42, 6), dpi=90
            )

        # Plot without loan status
        if continuous:
            sns.distplot(
                portfolioDisplay.loc[
                    portfolioDisplay[col_name].notnull(), col_name
                ],
                kde=False,
                ax=ax[0],
            )
        else:
            sns.countplot(
                x=portfolioDisplay[col_name],
                order=sorted(portfolioDisplay[col_name].unique()),
                color="#5975A4",
                saturation=1,
                ax=ax[0],
            )
            ax[0].tick_params(axis = "x", labelrotation=45)
            



        ax[0].set_xlabel(col_name)
        ax[0].set_ylabel("Count")
        ax[0].set_title(col_name)

        # Plot with loan status
        if continuous:
            sns.boxplot(
                x=col_name,
                y=self.colNames["loanStatus"],
                data=portfolioDisplay,
                ax=ax[1],
            )
            ax[1].set_ylabel("")
            ax[1].set_title(col_name + " by Loan Status")
            ax[1].set_xlabel(col_name)
            
            
            portfolioDisplay[self.colNames["loanStatus"]].value_counts(normalize=True).plot(kind='bar', ax=ax[2])
            ax[2].set_ylabel("")
            ax[2].set_title("Loan Status Distribution")
            ax[2].tick_params(axis = "x", labelrotation=45)
            
        else:
            transitionProbValue = (
                portfolioDisplay
                .groupby(col_name)[self.colNames["loanStatus"]]
                .value_counts(normalize=True)
            )
            transitionProbDf = pd.DataFrame(
                data=transitionProbValue.values,
                index=transitionProbValue.index,
                columns=["TransitionProb"],
            ).reset_index(drop=False)

            transitionProbDfPivot = (
                transitionProbDf.pivot(
                    index=col_name, columns=self.colNames["loanStatus"], values="TransitionProb"
                )
                .reset_index(drop=False)
                .fillna(0)
            )
            for toStatus, statusAx in zip(
                distinctStatus,
                ax[1:],
            ):
                sns.scatterplot(
                    x=col_name,
                    y=toStatus,
                    data=transitionProbDfPivot,
                    color="#5975A4",
                    ax=statusAx,
                    s=200,
                )
                statusAx.set_title("to " + toStatus + " Rate by " + col_name, fontsize=12)
                statusAx.set_ylabel("")
                statusAx.set_xlabel(col_name, fontsize=12)

                statusAx.set_yticklabels(["{:,.2%}".format(x) for x in statusAx.get_yticks()])

                statusAx.tick_params(axis="both", labelsize=9)
                statusAx.tick_params(axis = 'x', labelrotation=45)

        f.suptitle(" Transition Prob", fontsize=15)

        plt.tight_layout()
        plt.show()

