import tpqoa
import numpy as np
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

api = tpqoa.tpqoa('/opt/anaconda3/envs/Algea_3_7/oanda.cfg')
# print(api.get_instruments()[:15])

instrument = 'EUR_USD'
start = '2020-08-10'
end = '2020-08-15'
granularity = 'D'
price = 'M'

## HISTORICAL DATA
data = api.get_history(instrument, start, end, granularity, price)
data.info()
print(data.head())

data['returns'] = np.log(data['c'] / data['c'].shift(1))
cols = []
for momentum in [15, 30, 60, 120]:
    col = 'position_{}'.format(momentum)
    data[col] = np.sign(data['returns'].rolling(momentum).mean())
    cols.append(col)

strats = ['returns']

for col in cols:
    strat = 'strategy_{}'.format(col.split('_')[1])
    data[strat] = data[col].shift(1) * data['returns']
    strats.append(strat)

data[strats].dropna().cumsum().apply(lambda x: x * 20).apply(np.exp).plot(figsize=(10,6));
# plt.show()


