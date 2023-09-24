import numpy as np
import pandas as pd
from PortfolioEDA.PortfolioTrend import PortfolioTrend
from PortfolioEDA.Stratification import Stratification
from PortfolioEDA.PortfolioTM import PortfolioTM


class PortfolioEDA:
    def __init__(self, loanPortfolio,
                 colNames):
        self.loanPortfolio = loanPortfolio
        self.colNames = colNames
        self.tapeDateList = self.getTapeDateList()
        self.latestTape = self.getLatestTape()
        
        self.portfolioTrend = PortfolioTrend(self.loanPortfolio, 
                                                        colNames = self.colNames)

        self.portfolioStratification = Stratification(self.latestTape)
        
        self.portfolioTM = PortfolioTM(self.loanPortfolio, colNames = self.colNames)
    
    def getTapeDateList(self):
        return list(
            pd.to_datetime(
                self.loanPortfolio[self.colNames['date']].unique()
                )
            )

    def getLatestTape(self):
        return self.loanPortfolio[pd.to_datetime(self.loanPortfolio[self.colNames['date']]) == max(self.tapeDateList)]