import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

class PortfolioTrend:
    def __init__(self, portfolio, colNames):
        self.portfolio = portfolio
        self.transitionMatrixDict = {}
        self.colNames = colNames
        

    def getMeasureTrend(self, dateField, measure, calcOps, helper, label = None):        
        if calcOps.lower() == 'wt_avg':
            funcHandle = lambda x: 0 if self.portfolio.loc[x.index, helper].sum() == 0 else np.average(x, weights=self.portfolio.loc[x.index, helper])
        if calcOps.lower() == 'sum':
            funcHandle = lambda x : np.sum(x)
        res = self.portfolio.groupby(dateField).agg(res = (measure, funcHandle))

        if label is None:
            pass
        else:
           res = res.rename(columns = {'res': label})
        return res

    def calculateTransitionMatrix(self, startDate, endDate):
        loanTapeTraining = self.portfolio[(pd.to_datetime(self.portfolio[self.colNames['date']])>=startDate)
                                          &(pd.to_datetime(self.portfolio[self.colNames['date']])<=endDate)]
        transitionMatrixDollar = pd.pivot_table(loanTapeTraining, values=self.colNames['eopBal']+"_Lag1", index=self.colNames['loanStatus']+"_Lag1",
                        columns=self.colNames['loanStatus'], aggfunc=np.sum, fill_value=0)
        transitionMatrix = transitionMatrixDollar.div(transitionMatrixDollar.sum(axis=1), axis=0)
        return transitionMatrix

    def generateTransitionMatrix(self, WindowInMo = 6):
        self.transitionMatrixDict = {}
        endDate = pd.to_datetime(self.portfolio[self.colNames['date']]).max()
        startDate = pd.to_datetime(self.portfolio[self.colNames['date']]).min() + relativedelta(months=WindowInMo)

        monthRange = [pd.to_datetime(item) for item in list(self.portfolio['Snapshotdt'].unique()) if (pd.to_datetime(item) >=startDate and pd.to_datetime(item) <= endDate)]
        
        for trainingMonth in monthRange:
            self.transitionMatrixDict[trainingMonth] = self.calculateTransitionMatrix(trainingMonth - relativedelta(months=WindowInMo), trainingMonth)

        return self
    
    def getLatestTransitionMatrix(self):
        if len(self.transitionMatrixDict) == 0:
            return None
        return self.transitionMatrixDict[max(self.transitionMatrixDict.keys())]

    def portfolioCreditStats(self, creditStat, balanceCol):
        if creditStat.lower() not in ['cdr','cpr','dq']:
            print("creditStat must be one of 'CDR', 'CPR', 'DQ'")
            return None

        bal = None
        bal = balanceCol.get('defaultBal', None) if creditStat.lower() == 'cdr' else bal
        bal = balanceCol.get('prepayBal', None) if creditStat.lower() == 'cpr' else bal
        bal = balanceCol.get('dqBal', None) if creditStat.lower() == 'dq' else bal
        
        if bal is None:
            print("balanceCol must be a dictionary with keys 'defaultBal', 'prepayBal', 'dqBal'")
            return None    

        temp = (self.portfolio.groupby(self.colNames['date'])
        .agg(
            bal=(bal, "sum"),
            eopBal_Lag1=(self.colNames['eopBal'] + "_Lag1", "sum"),
            eopBal=(self.colNames['eopBal'], "sum"),
        )
        .reset_index()
        )

        if creditStat.lower() in ['cdr', 'cpr']:
            temp.loc[:, 'res'] = temp.loc[:, 'bal'] / temp.loc[:, 'eopBal_Lag1']
            temp.loc[:, 'res'] = temp.loc[:, 'res'].apply(lambda x: 1-(1-x)**(12))
        
        elif creditStat.lower() in ['dq']:
            temp.loc[:, 'res'] = temp.loc[:, 'bal'] / temp.loc[:, 'eopBal']
        
        return temp[['res']].rename(columns = {'res':creditStat})

loanPortfolio = pd.read_csv('./Data/loantape.csv')
portfolioTrendH = PortfolioTrend(loanPortfolio, colNames = {"date":'Snapshotdt', "loanStatus":'LoanStatus2', "eopBal":'UPB'})

# portfolioTrendH.generateTransitionMatrix()
# portfolioTrendH.getLatestTransitionMatrix()

# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'UPB', 'sum', None, 'eopBal'))
# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'HighFico', 'wt_avg', 'UPB', 'FICO'))
# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'DQBal', 'sum', None, 'DQBal'))
# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'OriginalTerm', 'wt_avg', 'UPB', 'OrigTerm'))
# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'RemainingTerm', 'wt_avg', 'UPB', 'RemTerm'))
# print(portfolioTrendH.getMeasureTrend('Snapshotdt', 'CurrentRate', 'wt_avg', 'UPB', 'IntRate'))

# print(portfolioTrendH.portfolioCreditStats('cdr', {'defaultBal':'DefaultBal'}))
# print(portfolioTrendH.portfolioCreditStats('cpr', {'prepayBal':'PrepayBal'}))
# print(portfolioTrendH.portfolioCreditStats('dq', {'dqBal':'DQBal'}))