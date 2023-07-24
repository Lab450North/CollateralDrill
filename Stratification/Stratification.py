import numpy as np
import pandas as pd

class Measure:
    def __init__(self, label, column, calcMethod, calcHelper = None):
        self.label = label
        self.column = column
        self.calcMethod = calcMethod
        self.calcHelper = calcHelper

class Dimension:
    def __init__(self, label, column, binSize, sortBy, sortAscending = False, topN = None):
        self.label = label
        self.column = column
        self.binSize = binSize
        self.sortBy = sortBy
        self.sortAscending = sortAscending
        self.topN = topN
        
class Stratification:
    def __init__(self, loanTape):
        self.loans = loanTape
        self.measures = {}
        self.dimensions = {}
        self.stratsTable = {}
        self.func_dict = dict()
   
    def addSorting(self, byLabel, sortAscending = False):
        self.byLabel = byLabel
        self.sortAscending = sortAscending

    def addDimension(self, dimensionName,label, column, binSize, sortBy, sortAscending, topN):
        self.dimensions[dimensionName] = Dimension(label, column, binSize, sortBy, sortAscending, topN)

    def addMeasure(self, meausreName, label, column, calcMethod, calcHelper = None):
        self.measures[meausreName] = Measure(label, column, calcMethod, calcHelper)

    def func_define(self,calc_method, helper = None):
        if calc_method == 'Count':
            return lambda x: np.size(x)
        if calc_method == 'Count %':
            return lambda x: np.size(x)/(len(self.loans)*1.0)
        if calc_method == 'Sum':
            return lambda x: np.sum(x)
        if calc_method == 'Min':
            return lambda x: np.min(x)
        if calc_method == 'Sum %':
            return lambda x: np.sum(x)*1.0/self.loans[x.name].sum()
        if calc_method == 'Wt_Avg':
            return lambda x: 0 if self.loans.loc[x.index, helper].sum() == 0 else np.average(x, weights=self.loans.loc[x.index, helper])
        if calc_method == 'Avg':
            return lambda x: np.average(x)
        if calc_method == 'Divide':
            return lambda x: np.sum(x)*1.0/self.loans.loc[x.index, helper].sum()
        if calc_method == 'Countif_Over30':
            return lambda x: sum(self.loans.loc[x.index,x.name]>30)
        if calc_method == 'Countif_Over60':
            return lambda x: sum(self.loans.loc[x.index,x.name]>60)
        if calc_method == 'Countif_Over90':
            return lambda x: sum(self.loans.loc[x.index,x.name]>90)
        if calc_method == 'Countif_Over30_Ratio':
            return lambda x: sum(self.loans.loc[x.index,x.name]>30)*1.0/len(self.loans.loc[x.index, helper])
        if calc_method == 'Countif_Over60_Ratio':
            return lambda x: sum(self.loans.loc[x.index,x.name]>60)*1.0/len(self.loans.loc[x.index, helper])
        if calc_method == 'Countif_Over90_Ratio':
            return lambda x: sum(self.loans.loc[x.index,x.name]>90)*1.0/len(self.loans.loc[x.index, helper])

    def generateStrat(self):
        
        self.loans.loc[:, 'TOTALCOL'] = 'Total'

        for dimensionName, dimensionObj in self.dimensions.items():
            groupByCol = dimensionObj.column
            calcRes = pd.DataFrame()


            # -calc- ***************** Bucketing Numerical if Applicable *****************
            if dimensionObj.binSize is not None:
                if pd.api.types.is_numeric_dtype(self.loans[dimensionObj.column]):
                    binsCol = dimensionObj.column + "_bins"
                    groupByCol = binsCol
                    lower = np.percentile(self.loans[dimensionObj.column], 1)
                    upper = np.percentile(self.loans[dimensionObj.column], 99)
                    bins = np.arange(lower, upper + dimensionObj.binSize, dimensionObj.binSize)
                    self.loans.loc[:, binsCol] = pd.cut(self.loans[dimensionObj.column], bins = bins, include_lowest=True)

            # -calc- ***************** Calculate Each Measure *****************
            for measureName, measureObj in self.measures.items():
                temp = self.loans.groupby(groupByCol)[measureObj.column]\
                    .agg(temp = self.func_define(measureObj.calcMethod, measureObj.calcHelper))\
                        .rename(columns={'temp': measureObj.label})

                totalTemp = self.loans.groupby('TOTALCOL')[measureObj.column]\
                    .agg(temp = self.func_define(measureObj.calcMethod, measureObj.calcHelper))\
                        .rename(columns={'temp': measureObj.label})

                temp = pd.concat([temp, totalTemp], axis=0)

                calcRes = pd.concat([calcRes, temp], axis=1)

            
            # -calc- ***************** Sorting *****************
            if dimensionObj.sortBy is not None:
                calcRes = calcRes.sort_values(by = dimensionObj.sortBy, ascending = dimensionObj.sortAscending)
                calcRes = pd.concat(
                    [calcRes.drop('Total', axis=0), calcRes.loc[['Total'], :]], 
                    axis=0)

            # -calc- ****************** Top N ******************
            if dimensionObj.topN is not None:
                rowCount = calcRes.shape[0] - 1
                totalRow = calcRes.loc[['Total'], :]
                calcRes = calcRes.head(dimensionObj.topN)

                # -calc- ****************** Other Row ******************
                if rowCount > dimensionObj.topN:
                    otherRow = pd.DataFrame()
                    self.loans.loc[:, 'OTHERCOL'] = 'Other'
                    for measureName, measureObj in self.measures.items():
                        temp = self.loans[self.loans[groupByCol].isin(calcRes.index) == False]\
                            .groupby('OTHERCOL')[measureObj.column]\
                                .agg(temp = self.func_define(measureObj.calcMethod, measureObj.calcHelper))\
                                    .rename(columns={'temp': measureObj.label})
                        
                        otherRow = pd.concat([otherRow, temp], axis=1)
                    
                    self.loans = self.loans.drop('OTHERCOL', axis=1)
                    
                    calcRes = pd.concat([calcRes, otherRow], axis=0)

                calcRes = pd.concat([calcRes, totalRow], axis=0)


            self.stratsTable[dimensionName] = calcRes
        
        self.loans = self.loans.drop('TOTALCOL', axis=1)

# ********************************** Sample Data of Usage **********************************

# StratificationH = Stratification(pd.read_csv('./Data/loantape.20200131.csv'))

# # Measures
# StratificationH.addMeasure(meausreName = "LoanCount", label = "LoanCount", column = "ApplicationID", calcMethod = "Count", calcHelper = None)
# StratificationH.addMeasure(meausreName = "LoanCountPct", label = "LoanCount %", column = "ApplicationID", calcMethod = "Count %", calcHelper = None)
# StratificationH.addMeasure(meausreName = "OriginalAmtFinanced", label = "Financed", column = "OriginalAmtFinanced", calcMethod = "Sum", calcHelper = None)
# StratificationH.addMeasure(meausreName = "OriginalAmtFinancedPct", label = "Financed %", column = "OriginalAmtFinanced", calcMethod = "Sum %", calcHelper = None)
# StratificationH.addMeasure("LTVCore", "WAVGLTV", "LTVCore", "Wt_Avg", 'OriginalAmtFinanced')
# StratificationH.addMeasure("FICO", "WAVGFICO", "HighFico", "Wt_Avg", 'OriginalAmtFinanced')

# # Dimensions
# StratificationH.addDimension(dimensionName = "Make", label = "Make", column = "Make", binSize = None, sortBy = "Financed %", sortAscending = False, topN = 5)
# StratificationH.addDimension(dimensionName = "BookNewUsed", label = "BookNewUsed", column = "BookNewUsed", binSize = None, sortBy = None, sortAscending = False, topN = None)
# StratificationH.addDimension(dimensionName = "BookTier", label = "BookTier", column = "BookTier", binSize = None, sortBy = "Financed", sortAscending = False, topN = 5)
# StratificationH.addDimension(dimensionName = "OriginalRate", label = "OriginalRate", column = "OriginalRate", binSize = 0.02, sortBy = None, sortAscending = False, topN = 10)
# StratificationH.addDimension(dimensionName = "LTVCore", label = "LTVCore", column = "LTVCore", binSize = 0.1, sortBy = None, sortAscending = False, topN = None)
# StratificationH.addDimension(dimensionName = "HighFico", label = "HighFico", column = "HighFico", binSize = 30, sortBy = None, sortAscending = False, topN = None)



# StratificationH.generateStrat()
# print(StratificationH.stratsTable['HighFico'])
