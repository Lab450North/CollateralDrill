import pandas as pd
import pickle

with open('./TransitionMatrixModeling/SavedModels/fromCModel.pickle', 'rb') as pf:
    fromCModel = pickle.load(pf)
    
with open('./TransitionMatrixModeling/SavedModels/fromEDModel.pickle', 'rb') as pf:
    fromEDModel = pickle.load(pf)

with open('./TransitionMatrixModeling/SavedModels/fromLDModel.pickle', 'rb') as pf:
    fromLDModel = pickle.load(pf)

# data load and cleaning        
loanHist = pd.read_csv('./Data/loantape.csv')


statusMap = {'Current': 0, 'EarlyDQ': 1, 'LateDQ': 2, 'CO': 3, 'Prepaid': 4}
loanHist.loc[:, 'LoanStatus2_Ind'] = loanHist.loc[:, 'LoanStatus2'].replace(statusMap)

bins, labels = [0, 3, 6, 9, 12], [1, 2, 3, 4]
loanHist['remitQtr'] = pd.cut(pd.to_datetime(loanHist['Snapshotdt']).dt.month, bins=bins, labels=labels, right=True)

loanHist['UPB_lag1']= loanHist.groupby('ApplicationID')['UPB'].shift(1)

loanHist['factor'] = loanHist['UPB'] / loanHist['OriginalAmtFinanced']
loanHist['factor_lag1'] = loanHist.groupby('ApplicationID')['factor'].shift(1)

loanHist['LoanStatus2_lag1']= loanHist.groupby('ApplicationID')['LoanStatus2'].shift(1)
loanHist['LoanStatus2_Ind_lag1']= loanHist.groupby('ApplicationID')['LoanStatus2_Ind'].shift(1)


loanHist.loc[:, 'Intercept'] = 1

loanHist['BookNewUsed_Used'] = loanHist['BookNewUsed'].apply(lambda x: 1 if x == 'Used' else 0)

bookTier = {}
bookTier['Tier-1'] = ["Tier -", "Tier -1", "Tier 0", "Tier 1","Tier 2", "Tier 3"]
bookTier['Tier-2'] = ["Tier 4", "Tier 5", "Tier 6", "Tier 7", "Tier 8" "Tier Thin"]
bookTier['Tier-3'] = ["Tier 9", "Tier 10"]

loanHist['BookTierPooling_Tier-2'] = loanHist['BookTier'].apply(lambda x: 1 if x in bookTier['Tier-2'] else 0)
loanHist['BookTierPooling_Tier-3'] = loanHist['BookTier'].apply(lambda x: 1 if x in bookTier['Tier-3'] else 0)

loanHist['remitQtrPooling_remit2H'] = loanHist['remitQtr'].apply(lambda x : 1 if x in [1,2] else 0)


latestLoanTape = loanHist[loanHist['Snapshotdt'] == loanHist['Snapshotdt'].max()]
secondLargestDate = loanHist['Snapshotdt'].drop_duplicates().sort_values().tail(2).values[0]
latestLoanTape_1 = loanHist[loanHist['Snapshotdt'] == secondLargestDate]

temp = latestLoanTape_1[latestLoanTape_1.LoanStatus2 == 'Current']
predictionFromC = fromCModel['model'].predict(temp[list(fromCModel['model'].pvalues.index)])

temp = latestLoanTape_1[latestLoanTape_1.LoanStatus2 == 'EarlyDQ']
predictionFromED = fromEDModel['model'].predict(temp[list(fromEDModel['model'].pvalues.index)])

temp = latestLoanTape_1[latestLoanTape_1.LoanStatus2 == 'LateDQ']
predictionFromLD = fromLDModel['model'].predict(temp[list(fromLDModel['model'].pvalues.index)])


# ! ****************************************************************************************************


def fromCurrentPred(prevDate, nextDate):
    status = 'Current'

    prevDateLoanTape = loanHist[loanHist['Snapshotdt'] == prevDate]
    prevDateLoanTape = prevDateLoanTape[prevDateLoanTape['LoanStatus2'] == status]

    nextDateLoanTape = loanHist[loanHist['Snapshotdt'] == nextDate]
    realized = nextDateLoanTape[nextDateLoanTape['ApplicationID'].isin(list(prevDateLoanTape['ApplicationID']))]
    realizedProb = realized.groupby(['LoanStatus2_Ind'],group_keys=True)[['UPB_lag1']].agg(
        prob = ('UPB_lag1', lambda x: x.sum() / realized['UPB_lag1'].sum())
    ).transpose()
    realizedProb.columns = ['realized.'+ str(item) for item in realizedProb.columns]
    realizedProb.index = [nextDate]

    predictionFromC = fromCModel['model'].predict(prevDateLoanTape[list(fromCModel['model'].pvalues.index)])
    predictionProb = pd.merge(predictionFromC,prevDateLoanTape[['UPB']], how = 'left', left_index=True, right_index=True)
    predictionProb = pd.DataFrame(predictionProb[[0, 1, 2, 3, 4]].multiply(predictionProb["UPB"], axis="index").sum())
    predictionProb.loc[:, 'pred'] = predictionProb[0] /  predictionProb[0].sum()

    
    predictionProbOut = predictionProb[['pred']].transpose()
    predictionProbOut.columns = ['prediction.'+ item for item in ['0','1','2','3','4']]
    predictionProbOut.index = [nextDate]

    return pd.merge(realizedProb, predictionProbOut, how = 'outer', left_index=True, right_index=True)

import itertools
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

res = pd.DataFrame(columns = ['realized.0.0',
                              'realized.1.0',
                              'realized.2.0',
                              'realized.3.0',
                              'realized.4.0',
                              'prediction.0',
                              'prediction.1',
                              'prediction.2',
                              'prediction.3',
                              'prediction.4'])

for prevDate, nextDate in pairwise(loanHist['Snapshotdt'].drop_duplicates().sort_values()):
    res = pd.concat([res, fromCurrentPred(prevDate, nextDate)])