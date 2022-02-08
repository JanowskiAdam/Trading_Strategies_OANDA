import pandas as pd
import tpqoa
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from pylab import plt

import OANDA_Historical
import numpy as np

n_estimators = 15
random_state = 100
max_depth = 2
min_samples_leaf = 15
subsample = 0.33

dtc = DecisionTreeClassifier(random_state=random_state,
                             max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf)

model = AdaBoostClassifier(base_estimator=dtc,
                           n_estimators=n_estimators,
                           random_state=random_state)

example = OANDA_Historical.OANDA_Base(symbol='EUR_USD', start=OANDA_Historical.yearsAgo(1), end=OANDA_Historical.lastDay(), granularity='D', price='M', amount=10000, ftc=0.0, ptc=0.0008, verbos=True)
data = example.analize_data()

split = int(len(data) * 0.7)
train = data.iloc[:split].copy()
mu, std = train.mean(), train.std()
train_ = (train - mu) / std

model.fit(train_.iloc[:, 4:], train['direction'])
print('train accuracy')
print(accuracy_score(train['direction'], model.predict(train_.iloc[:, 4:])))

test = data.iloc[split:].copy()
test_ = (test - mu) / std
test['position'] = model.predict(test_.iloc[:, 4:])

print('test accuracy')
print(accuracy_score(test['direction'], test['position']))

test['strategy'] = test['position'] * test['return']

print('Number of trades: ')
print(sum(test['position'].diff() != 0))

test['strategy_tc'] = np.where(test['position'].diff() != 0,
                               test['strategy'] - example.ptc,
                               test['strategy'])

print(test[['return', 'strategy', 'strategy_tc']].sum().apply(np.exp))

#test[['return', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10,6))
#plt.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig, ax = plt.subplots(figsize=(10, 6))
plt.title('Signal: %s' % ('EUR/USD'))
ax2 = ax.twinx()
ax.plot(test.index, test[['return', 'strategy', 'strategy_tc']].cumsum())
ax2.plot(test.index, test[['position']], color='g')
ax.set_ylabel('Price', color='b')
ax2.set_ylabel('Position', color='g')
plt.tight_layout()
plt.legend(test[['price', 'strategy', 'strategy_tc']])
plt.show()




