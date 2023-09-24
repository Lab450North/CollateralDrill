import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import numpy_financial as npf
from datetime import date
import time
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

idx = pd.IndexSlice

class RollRateAmortization:
    def __init__(self, loanTape, TMModels):
        self.statusBucket  = ['Current','EarlyDQ','LateDQ','CO', 'Prepaid']
        self.loanColumns = ['BOPBal','TransitionBal','IntPayment','Default','PrepayPrin', 'ScheduledPrin', 'Recovery', 'Loss', 'PrinCF','TotalCF', 'DQBal', 'EOPBal', 'RemainingTerm']
        
        self.index = pd.MultiIndex.from_product([loanTape['ApplicationID'].unique(), self.statusBucket, self.loanColumns], names=['ApplicationID', 'StatusBucket', 'LoanColumns'])

        self.loanTape = loanTape
        self.extendedLoanTape = self.loanTape.copy()

        self.extendedLoanTapeEnrichment()
        self.loansBatch = []
        
        self.TMModels = TMModels

    def extendedLoanTapeEnrichment(self):
        bookTier = {}
        bookTier['Tier-1'] = ["Tier -", "Tier -1", "Tier 0", "Tier 1","Tier 2", "Tier 3"]
        bookTier['Tier-2'] = ["Tier 4", "Tier 5", "Tier 6", "Tier 7", "Tier 8" "Tier Thin"]
        bookTier['Tier-3'] = ["Tier 9", "Tier 10"]

        self.extendedLoanTape['BookTierPooling_Tier-2'] = self.extendedLoanTape['BookTier'].apply(lambda x: 1 if x in bookTier['Tier-2'] else 0)
        self.extendedLoanTape['BookTierPooling_Tier-3'] = self.extendedLoanTape['BookTier'].apply(lambda x: 1 if x in bookTier['Tier-3'] else 0)

        bins, labels = [0, 3, 6, 9, 12], [1, 2, 3, 4]
        self.extendedLoanTape['remitQtr'] = pd.cut(pd.to_datetime(self.extendedLoanTape['Snapshotdt']).dt.month, bins=bins, labels=labels, right=True)
        self.extendedLoanTape['remitQtrPooling_remit2H'] = self.extendedLoanTape['remitQtr'].apply(lambda x : 1 if x in [1,2] else 0)

        self.extendedLoanTape['BookNewUsed_Used'] = self.extendedLoanTape['BookNewUsed'].apply(lambda x: 1 if x == 'Used' else 0)

        self.extendedLoanTape['factor_lag1'] = self.extendedLoanTape['UPB'] / self.extendedLoanTape['OriginalAmtFinanced']

        self.extendedLoanTape.loc[:, 'Intercept'] = 1

    def updateExtendedLoanTape(self):
        self.extendedLoanTape = self.extendedLoanTape[self.extendedLoanTape['ApplicationID'].isin(list(self.loans.index.get_level_values('ApplicationID')))]
        self.extendedLoanTape = self.extendedLoanTape[
            list(set(list(self.TMModels['fromCModel'].pvalues.index)+
                     list(self.TMModels['fromEDModel'].pvalues.index)+
                     list(self.TMModels['fromLDModel'].pvalues.index)
                     )
                 ) + ['ApplicationID']]
        for field in ['RemainingTerm', 'factor_lag1']:
            if field == 'factor_lag1':
                fieldDf = self.workLoans.loc[idx[:,:,'EOPBal'], "value"].reset_index().groupby(['ApplicationID'])[['value']].agg(
                    value = ('value', lambda x: x.sum())
                ).rename(columns = {'value': 'EOPBal'}).reset_index()
                fieldDf = pd.merge(fieldDf, self.loanTape[['ApplicationID', 'OriginalAmtFinanced']], 
                                   how = 'left', 
                                   left_on = 'ApplicationID', 
                                   right_on = 'ApplicationID')
                fieldDf.loc[:, field] = fieldDf['EOPBal'] / fieldDf['OriginalAmtFinanced']
            else:
                fieldDf = self.workLoans.loc[idx[:,:,field], "value"].reset_index().groupby(['ApplicationID'])[['value']].agg(value = ('value', lambda x: x.iloc[0])).rename(columns = {'value': field}).reset_index()
            
            self.extendedLoanTape = self.extendedLoanTape.drop([field],axis = 1)
            self.extendedLoanTape = pd.merge(self.extendedLoanTape,
                                             fieldDf,
                                             how = 'left',
                                             left_on = "ApplicationID",
                                             right_on = "ApplicationID"
                                             )
        
    def runCashflow(self):
        self.setUpLoans()
        self.runLoans()
        self.aggregateLoans()


    # ** run each loan's transition matrix model        
    # this should be from separate class / procedure
    def runTransitionMatrixModel(self, loanIDList):
        data = [
            [0.9007, 0.0781, 0.0010, 0.0002, 0.02],
            [0.3256, 0.3716, 0.2912, 0.0004, 0.0112],
            [0.1164, 0.0810, 0.6980, 0.0971, 0.0075],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ]

        dummyTransitionMatrix = pd.concat([pd.DataFrame(data, index=self.statusBucket, columns=self.statusBucket)] * len(loanIDList), 
                                    keys=loanIDList, 
                                    names=['ApplicationID', 'StatusBucket'])
        

        predictionFromC = self.TMModels['fromCModel'].predict(self.extendedLoanTape[list(self.TMModels['fromCModel'].pvalues.index)])
        predictionFromC.index = loanIDList
        predictionFromC.columns = self.statusBucket
        predictionFromC.index.name = 'ApplicationID'
        predictionFromC.loc[:, 'StatusBucket'] = 'Current'
        predictionFromC = predictionFromC.set_index(['StatusBucket'], append=True, inplace=False)

        predictionFromED = self.TMModels['fromEDModel'].predict(self.extendedLoanTape[list(self.TMModels['fromEDModel'].pvalues.index)])
        predictionFromED.index = loanIDList
        predictionFromED.columns = self.statusBucket
        predictionFromED.index.name = 'ApplicationID'
        predictionFromED.loc[:, 'StatusBucket'] = 'EarlyDQ'
        predictionFromED = predictionFromED.set_index(['StatusBucket'], append=True, inplace=False)

        predictionFromLD = self.TMModels['fromLDModel'].predict(self.extendedLoanTape[list(self.TMModels['fromLDModel'].pvalues.index)])
        predictionFromLD.index = loanIDList
        predictionFromLD.columns = self.statusBucket
        predictionFromLD.index.name = 'ApplicationID'
        predictionFromLD.loc[:, 'StatusBucket'] = 'LateDQ'
        predictionFromLD = predictionFromLD.set_index(['StatusBucket'], append=True, inplace=False)

        
        predictionFromCO = pd.concat([pd.DataFrame([[0,0,0,1,0]], columns=self.statusBucket)] * len(loanIDList),
                                    keys = loanIDList)

                
        predictionFromCO.index = pd.MultiIndex.from_arrays(
            [predictionFromCO.index.get_level_values(0), ['CO'] * len(predictionFromCO)],
            names=['ApplicationID','StatusBucket'])


        predictionFromPrepaid = pd.concat([pd.DataFrame([[0,0,0,0,1]], columns=self.statusBucket)] * len(loanIDList), 
                                    keys=loanIDList)
        
        predictionFromPrepaid.index = pd.MultiIndex.from_arrays(
            [predictionFromPrepaid.index.get_level_values(0), ['Prepaid'] * len(predictionFromPrepaid)],
            names=['ApplicationID','StatusBucket'])
        

        
        transitionMatrix = pd.concat([predictionFromC,
                                      predictionFromED,
                                      predictionFromLD,
                                      predictionFromCO,
                                      predictionFromPrepaid
            ], ignore_index = False)

        return transitionMatrix

    def getPeriods(self):
        if len(self.loans.columns) == 1:
            self.prevPeriod, self.currPeriod = None, self.loans.columns[-1]
        else:
            self.prevPeriod, self.currPeriod = self.loans.columns[-2:]

    def getFieldFromLoanTapeToWorkloans(self, field):
        self.workLoans = self.workLoans.merge(self.loanTape[['ApplicationID', field]].set_index(['ApplicationID']),
                                              how = 'left', left_index=True, right_index=True)
        
        

    def unstackCurrentPeriod(self):
        self.workLoans = self.loans[self.currPeriod].unstack(level='LoanColumns')

    def setUpLoans(self):
        self.loans =  pd.DataFrame(index=self.index)

        # ** Assign to beinning period, (last period from loan tape)

        # create new column (first period / date)
        self.loans.loc[:, date(*map(int, self.loanTape['Snapshotdt'].unique()[0].split('-')))] = 0
        self.getPeriods()

        # assigne UPB basd off application id / status bucket to loan column EOPBal        
        self.mapFieldFromLoanTape(loanTapeField = 'UPB', loanColumn = 'EOPBal')

        # assigne RemainingTerm basd off application id  to loan column RemainingTerm, regardless of StatusBucket
        self.mapFieldFromLoanTape(loanTapeField = 'RemainingTerm', loanColumn = 'RemainingTerm', byStatusBucket = False)

        # map DQ balances
        self.unstackCurrentPeriod()
        def dqBalMap(x):
            if x.name[1] in ['EarlyDQ', 'LateDQ']:
                return x['EOPBal']
            else:
                return 0
        self.workLoans.loc[:, 'DQBal'] = self.workLoans.loc[:, ['EOPBal']].apply(lambda x : dqBalMap(x), axis = 1)
        self.updateLoansFromWorkloans()
           
        self.loans = self.loans.fillna(0)

    def runLoans(self):
        i = 0
        while self.loans.shape[0] > 0:
            # * run transition matrix modeling. for now, reduced to a dummy one and each loan has the same matrix
            loanTransitionMatrix = self.runTransitionMatrixModel(list(self.loans.index.get_level_values(0).unique()))

            # * new period / date
            self.loans.loc[:, self.loans.columns[-1] + relativedelta(months=1)] = 0
            self.getPeriods()

            # * bop bal
            self.loans.loc[idx[:,:, 'BOPBal'], self.currPeriod] = self.loans.loc[idx[:, :, 'EOPBal'], self.prevPeriod].values

            # * transition bal
            bopBal = self.loans.xs(key='BOPBal', level='LoanColumns', axis=0)[[self.currPeriod]]

            temp = bopBal.merge(loanTransitionMatrix,  how = 'left', left_index=True, right_index=True)
            result = temp[self.statusBucket].multiply(temp[self.currPeriod].fillna(0), axis = 0)
            summed_result = result.groupby(level='ApplicationID').sum().stack().reset_index()
            summed_result.columns = ['ApplicationID', 'StatusBucket', 'value']
            summed_result['LoanColumns'] = 'TransitionBal'
            summed_result.set_index(['ApplicationID', 'StatusBucket','LoanColumns'], inplace=True)

            self.loans = self.loans.merge(summed_result,  how = 'left', left_index=True, right_index=True)
            self.loans.loc[idx[:, :, 'TransitionBal'], self.currPeriod] = self.loans.loc[idx[:, :, 'TransitionBal'], 'value'].fillna(0).values
            self.loans = self.loans.drop(columns=['value'])
                            
            # * remaining term
            self.loans.loc[idx[:,:, 'RemainingTerm'], self.currPeriod] = self.loans.loc[idx[:,:, 'RemainingTerm'], self.prevPeriod].apply(lambda x: x - 1 if x > 0 else 0)

            # ! cash flow schedule
            self.unstackCurrentPeriod()
            fieldsFromLoanTape = ['CurrentRate']
            
            for field in fieldsFromLoanTape:
                self.getFieldFromLoanTapeToWorkloans(field)

            # * IntPayment
            def intPaymentMap(x):
                if x.name[1] in ['Current', 'Prepaid']:
                    return x['TransitionBal'] * x['CurrentRate'] / 12
                else:
                    return 0

            self.workLoans.loc[:, 'IntPayment'] = self.workLoans.loc[:, ['CurrentRate', 'TransitionBal']].apply(lambda x : intPaymentMap(x), axis = 1)

            # * ScheduledPrin
            def scheduledPrinMap(x):
                if x.name[1] in ['Current']:
                    if x['RemainingTerm'] > 0:
                        return npf.pmt(x['CurrentRate'] / 12, x['RemainingTerm'], -x['TransitionBal']) - x['IntPayment']
                    else:
                        return 0
                else:
                    return 0
                
            self.workLoans.loc[:, 'ScheduledPrin'] = self.workLoans.loc[:, ['CurrentRate', 'TransitionBal', 'RemainingTerm', 'IntPayment']].apply(lambda x : scheduledPrinMap(x), axis = 1)
            
            
            # * Default
            def defaultMap(x):
                if x.name[1] in ['CO']:
                    return x['TransitionBal']

                elif x.name[1] in ['EarlyDQ', 'LateDQ']:
                    # default all the delinquency loans for last period
                    if x['RemainingTerm'] == 1:
                        return x['TransitionBal']
                    else:
                        return 0
                else:
                    return 0
            self.workLoans.loc[:, 'Default'] = self.workLoans.loc[:, ['TransitionBal', 'RemainingTerm']].apply(lambda x : defaultMap(x), axis = 1)
            
            # * Recovery
            self.workLoans.loc[:, 'Recovery'] = self.workLoans.loc[:, 'Default'] * 0

            # * Loss
            self.workLoans.loc[:, 'Loss'] = self.workLoans.loc[:, 'Default'] - self.workLoans.loc[:, 'Recovery']

            # * PrepayPrin
            def prepayPrinMap(x):
                if x.name[1] in ['Prepaid']:
                    return x['TransitionBal']
                else:
                    return 0
            self.workLoans.loc[:, 'PrepayPrin'] = self.workLoans.loc[:, ['TransitionBal']].apply(lambda x : prepayPrinMap(x), axis = 1)
            
            # * PrinCF
            self.workLoans.loc[:, 'PrinCF'] = (self.workLoans.loc[:, 'ScheduledPrin'] + 
                                            self.workLoans.loc[:, 'Recovery'] + 
                                            self.workLoans.loc[:, 'PrepayPrin'])
            
            # * PrinCF
            self.workLoans.loc[:, 'TotalCF'] = self.workLoans.loc[:, 'IntPayment'] + self.workLoans.loc[:, 'PrinCF']

            # * EOPBal
            self.workLoans.loc[:, 'EOPBal'] = (self.workLoans.loc[:, 'TransitionBal'] - 
                                            self.workLoans.loc[:, 'Default'] -
                                            self.workLoans.loc[:, 'PrepayPrin'] - 
                                            self.workLoans.loc[:, 'ScheduledPrin']
                                            )
            self.workLoans.loc[:, 'EOPBal'] = self.workLoans.loc[:, 'EOPBal'].apply(lambda x : 0 if x < 0.1 else x)

            # * DQBal
            def dqBalMap(x):
                if x.name[1] in ['EarlyDQ', 'LateDQ']:
                    return x['EOPBal']
                else:
                    return 0
            self.workLoans.loc[:, 'DQBal'] = self.workLoans.loc[:, ['EOPBal']].apply(lambda x : dqBalMap(x), axis = 1)
        
            # * check remaining term 
            loansListRemainingZeroTerm = list(self.workLoans[self.workLoans['RemainingTerm'] == 0].index.get_level_values(0).unique())

            # * move remaning temr == 0 to loans batch; only keep loans with remaining term > 0
            self.loansBatch.append(self.loans.loc[idx[loansListRemainingZeroTerm, :, :], :])
            self.loans = self.loans[~self.loans.index.get_level_values('ApplicationID').isin(loansListRemainingZeroTerm)]
            self.updateLoansFromWorkloans()
            self.updateExtendedLoanTape()

        self.loans = pd.concat(self.loansBatch, axis = 0, ignore_index=False)

    def aggregateLoans(self):
        self.loansAgg = self.loans.groupby(level='LoanColumns').sum()
        self.loansAggTrans = self.loansAgg.transpose()

    def updateLoansFromWorkloans(self):
        self.workLoans = self.workLoans.stack().reset_index()
        self.workLoans.columns = ['ApplicationID', 'StatusBucket','LoanColumns', "value"]
        self.workLoans = self.workLoans.set_index(['ApplicationID', 'StatusBucket','LoanColumns'])
        self.loans.drop(columns=[self.currPeriod], inplace = True)
        self.loans = self.loans.merge(self.workLoans, how = 'left', left_index=True, right_index=True)
        self.loans = self.loans.rename(columns = {"value": self.currPeriod})
        
    def mapFieldFromLoanTape(self, loanTapeField, loanColumn, byStatusBucket = True):
        if byStatusBucket:
            self.loans = self.loans.merge(self.loanTape.assign(LoanColumns=loanColumn)
                                        .set_index(['ApplicationID', 'LoanStatus2','LoanColumns'])
                                        .rename_axis(index={'LoanStatus2': 'StatusBucket'})[[loanTapeField]].rename(columns = {loanTapeField: "value"}),
                    how = 'left', left_index=True, right_index=True
                    )
            self.loans.loc[idx[:, :, loanColumn], self.currPeriod] = self.loans.loc[idx[:, :, loanColumn], 'value'].values
            self.loans = self.loans.drop(columns=['value'], inplace = False)

        else:
            self.unstackCurrentPeriod()
            self.workLoans = self.workLoans.merge(self.loanTape[['ApplicationID',loanTapeField]]
                                 .rename(columns = {loanTapeField:"value"})
                                 .set_index(['ApplicationID']),
                                 how = 'left', left_index=True, right_index=True)

            self.workLoans.loc[:, loanColumn] = self.workLoans.loc[:, 'value']
            self.workLoans.drop(columns=['value'], inplace = True)
            self.updateLoansFromWorkloans()

            
            




