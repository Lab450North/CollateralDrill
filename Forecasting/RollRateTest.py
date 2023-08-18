import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import numpy_financial as npf
from datetime import date
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

import RollRateModeling
loanTape = pd.read_csv('./Data/loantape.20200131.csv')
loanTape = loanTape[loanTape['UPB'] > 10]


replineLoanTape = loanTape.groupby(['LoanStatus2'])[['UPB','CurrentRate','RemainingTerm']].agg(
    UPB = ('UPB', lambda x : np.sum(x)),
    CurrentRate = ('CurrentRate', lambda x : np.average(x, weights=loanTape.loc[x.index, 'UPB'])),
    RemainingTerm = ('RemainingTerm', lambda x : np.around(np.average(x, weights=loanTape.loc[x.index, 'UPB'])))
).reset_index()

replineLoanTape.loc[:, 'Snapshotdt'] = '2020-12-31'
replineLoanTape.loc[:, 'ApplicationID'] = [1,2,3]

rollRateRun = RollRateModeling.RollRateAmortization(loanTape)
rollRateRun.runCashflow()

rollRateRunRepline = RollRateModeling.RollRateAmortization(replineLoanTape)
rollRateRunRepline.runCashflow()
rollRateRun.loansAggTrans.to_csv('loanlevel.csv')
rollRateRunRepline.loansAggTrans.to_csv('repline.csv')








