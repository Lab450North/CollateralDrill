import pandas as pd
import numpy as np
from RollRateAmortization import RollRateAmortization
import pickle
import os

loanTape = pd.read_csv('./Data/loantape.20200131.csv')
loanTape = loanTape

savePath = './TransitionMatrixModeling/SavedModels/'
for filename in os.listdir(savePath):
    fromCModel = pickle.load(open(os.path.join(savePath, 'fromCModel.pickle'), 'rb'))
    fromEDModel = pickle.load(open(os.path.join(savePath, 'fromEDModel.pickle'), 'rb'))
    fromLDModel = pickle.load(open(os.path.join(savePath, 'fromLDModel.pickle'), 'rb'))
    
TMModels = {'fromCModel': fromCModel['model'], 'fromEDModel': fromEDModel['model'], 'fromLDModel': fromLDModel['model']}


rollRateRun = RollRateAmortization(loanTape, TMModels)
rollRateRun.runCashflow()

rollRateRun.loansAggTrans.to_csv('./Forecasting/loansAggTransLoan.csv', index=False)
print('done')