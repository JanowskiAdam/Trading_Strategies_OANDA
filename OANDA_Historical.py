import pandas as pd
import tpqoa
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brute
import datetime
import math
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from pylab import plt
import pickle

api = tpqoa.tpqoa('/opt/anaconda3/envs/Algea_3_7/oanda.cfg')
#print(api.get_instruments())

equs = []

def kelly_strategy(f):
    global equs
    data = OANDA_Base.get_data()

    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = "capital_{:.2f}".format(f)
    data[equ] = 1
    data[cap] = data[equ] * f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        data.loc[t, cap] = data[cap].loc[t_1] * \
                            math.exp(data['return'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - \
                            data[cap].loc[t_1] + \
                            data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f

def lastDay():
    lastBusDay = datetime.datetime.today()
    if datetime.date.weekday(lastBusDay) == 5:  # if it's Saturday
        lastBusDay = lastBusDay - datetime.timedelta(days=1)  # then make it Friday
    elif datetime.date.weekday(lastBusDay) == 6:  # if it's Sunday
        lastBusDay = lastBusDay - datetime.timedelta(days=2)
    return lastBusDay.date().strftime("%Y-%m-%d")

def monthsAgo(month):
    monthsBack = datetime.datetime.today() - datetime.timedelta(30*month)
    if datetime.date.weekday(monthsBack) == 5:  # if it's Saturday
        lastBusDay = monthsBack - datetime.timedelta(days=1)  # then make it Friday
    elif datetime.date.weekday(monthsBack) == 6:  # if it's Sunday
        lastBusDay = monthsBack - datetime.timedelta(days=2)
    return monthsBack.date().strftime("%Y-%m-%d")

def yearsAgo(year):
    yearsBack = datetime.datetime.today() - datetime.timedelta(365*year)
    if datetime.date.weekday(yearsBack) == 5:  # if it's Saturday
        lastBusDay = yearsBack - datetime.timedelta(days=1)  # then make it Friday
    elif datetime.date.weekday(yearsBack) == 6:  # if it's Sunday
        lastBusDay = yearsBack - datetime.timedelta(days=2)
    return yearsBack.date().strftime("%Y-%m-%d")

class OANDA_Base(object):
    ''' Base class for OANDA backtesting of trading strategies.
    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to be used
    start: str
        start date for data selection
    end: str
        end date for data selection
    granularity: string
        a string like 'S5', 'M1' or 'D'
    price: string
        one of 'A' (ask), 'B' (bid) or 'M' (middle)
    amount: float
        amount to be invested either once or per trade
    ftc: float
        fixed transaction costs per trade (buy or sell)
    ptc: float
        proportional transaction costs per trade (buy or sell)

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    plot_data:
        plots the closing price for the symbol
    get_date_price:
        returns the date and price for the given bar
    print_balance:
        prints out the current (cash) balance
    print_net_wealth:
        prints out the current net wealth
    place_buy_order:
        places a buy order
    place_sell_order:
        places a sell order
    go_long:
        takes long position
    go_short
        takes short position
    optimize_Momentum
        using brute optimalization finds best momentum from range 1-100
    optimize_SMA
        using brute optimalization finds best 2 SMA from range 5-100
    plot_results
        plots results vs strategy showing position on chart
    close_out:
        closes out a long or short position

    '''

    def __init__(self, symbol, start, end, granularity, price, amount, ftc=0.0, ptc=0.0, verbos=True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.granularity = granularity
        self.price = price
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbos = verbos
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        data = api.get_history(self.symbol, self.start, self.end, self.granularity, self.price)
        data['return'] = np.log(data['c'] / data['c'].shift(1))
        self.nData = data.dropna()
        return self.nData

    def analize_data(self):
        ''' Retrieves and prepares the data for deep analization.
        '''
        window = 5
        BestSMA = OANDA_Base.optimize_single_SMA(self)
        BestMom = OANDA_Base.optimize_momentum(self)

        data = api.get_history(self.symbol, self.start, self.end, self.granularity, self.price)
        instrument = self.symbol
        data.rename(columns={'c': self.symbol}, inplace=True)
        data = pd.DataFrame(data, columns=[self.symbol, 'volume'])
        data['return'] = np.log(data[self.symbol] / data[self.symbol].shift(1))
        data['volatility'] = data['return'].rolling(window).std()
        data['momentum'] = np.sign(data['return'].rolling(BestMom).mean())
        data['sma'] = data[self.symbol].rolling(BestSMA).mean()
        data['min'] = data[self.symbol].rolling(window).min()
        data['max'] = data[self.symbol].rolling(window).max()

        data = data.dropna()
        # Create lagged columns
        lags = 5
        features = ['volume', 'return', 'volatility', 'momentum', 'sma', 'min', 'max']
        cols = []

        for f in features:
            for lag in range(1, lags + 1):
                col = f'{f}_lag_{lag}'
                data[col] = data[f].shift(lag)
                cols.append(col)

        aData = data.dropna()
        aData['direction'] = np.where(aData['return'] > 0, 1, -1)

        return aData, cols

    def plot_data(self, cols=None):
        ''' Plots the closing prices for symbol
        '''
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, subplot_titles=(self.symbol, 'Volume', 'Volatility'),
                            row_width=[0.2, 0.2, 0.7])
        # Plot OHLC on 1st row
        fig.add_trace(go.Candlestick(x=self.get_data().index,
                                     open=self.get_data()['o'],
                                     high=self.get_data()['h'],
                                     low=self.get_data()['l'],
                                     close=self.get_data()['c'],
                                     name="OHLC"),
                      row=1, col=1
                      )

        # Bar trace for Volume on 2nd row without legend
        fig.add_trace(go.Bar(x=self.get_data().index, y=self.get_data()['volume'], showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.get_data().index, y=self.get_data()['volume'].rolling(5).mean(), showlegend=False),row=2, col=1)

        # Bar trace for Volatility on 3nd row without legend
        fig.add_trace(go.Bar(x=self.get_data().index, y=self.get_data()['return'].abs(), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.get_data().index, y=self.get_data()['return'].abs().rolling(5).mean(), showlegend=False), row=3, col=1)

        # Do not show OHLC's rangeslider plot
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "sun"]),  # hide weekends
                # dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
            ]
        )
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()
        from plotly.subplots import make_subplots
        '''
        candlestick = go.Candlestick(
            x=self.get_data().index,
            open=self.get_data()['o'],
            high=self.get_data()['h'],
            low=self.get_data()['l'],
            close=self.get_data()['c']
        )

        fig = go.Figure(data=[candlestick])
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                #dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
            ]
        )

        fig.show()
        '''
    def get_date_price(self, bar):
        ''' Return date and price for bar
        '''
        date = str(self.nData.index[bar])[:10]
        price = self.nData.price.iloc[bar]
        return date, price

    def print_balance(self, bar):
        ''' Print out current cash balance info
        '''
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance {self.amount:.2f}')

    def print_net_wealth(self, bar):
        ''' Print out current net wealth info.
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date} | current net wealth {net_wealth:.2f}')

    def place_buy_order(self, bar, units=None, amount=None):
        ''' Place a buy order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbos:
            print(f'{date} | buying {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        ''' Place sell order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbos:
            print(f'{date} | selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.place_buy_order(bar, units=-self.units)
        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_buy_order(bar, amount=amount)

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)
        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_sell_order(bar, amount=amount)

    def optimize_single_SMA(self):
        def get_new_data():
            raw = self.get_data()
            raw.rename(columns={'c': 'price'}, inplace=True)
            raw['SMA'] = raw['price'].rolling(1).mean()
            raw = raw[['price', 'return', 'SMA']]
            self.nData = raw.dropna()
            return self.nData

        def set_parameter(SMA=None):
            if SMA is not None:
                self.SMA = SMA
                self.nData['SMA'] = self.nData['price'].rolling(self.SMA).mean()

        def update_and_run(SMA):
            set_parameter(int(SMA))
            return -run_strategy(SMA)[0]

        def run_brute(SMA_range):
            opt = brute(update_and_run, SMA_range, finish=None)
            return opt, -update_and_run(opt)

        def run_strategy(SMA):
            ''' Backtests the trading strategy.
            '''
            self.SMA = int(SMA)
            data = self.nData.copy().dropna()
            data['position'] = np.where(data['SMA'] > data['price'], -1, 1)
            data['strategy'] = data['position'].shift(1) * data['return']
            data.dropna(inplace=True)

            data['strategy_tc'] = np.where(data['position'].diff().fillna(0) != 0,
                                           data['strategy'] - self.ptc,
                                           data['strategy'])

            data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
            data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
            data['cstrategy_tc'] = self.amount * \
                                data['strategy_tc'].cumsum().apply(np.exp)
            self.results = data
            # absolute performance of the strategy
            aperf = self.results['cstrategy_tc'].iloc[-1]
            # out-/underperformance of strategy
            operf = aperf - self.results['creturns'].iloc[-1]
            return round(aperf, 2), round(operf, 2)

        get_new_data()
        param = run_brute(((1, 100, 1),))
        return int(param[0])

    def optimize_momentum(self):

        def set_parameter(momentum=None):
            if momentum is not None:
                self.momentum = momentum
                self.nData['position'] = self.nData['return'].rolling(self.momentum).mean()

        def update_and_run(momentum):
            set_parameter(int(momentum))
            return -run_strategy(momentum)[0]

        def run_brute(momentum_range):
            opt = brute(update_and_run, momentum_range, finish=None)
            return opt, -update_and_run(opt)

        def run_strategy(momentum):
            ''' Backtests the trading strategy.
            '''
            self.momentum = int(momentum)
            data = self.nData.copy().dropna()
            data['position'] = np.sign(data['return'].rolling(int(momentum)).mean())
            data['strategy'] = data['position'].shift(1) * data['return']
            # determine when a trade takes place
            data.dropna(inplace=True)

            data['strategy_tc'] = np.where(data['position'].diff().fillna(0) != 0,
                                           data['strategy'] - self.ptc,
                                           data['strategy'])

            data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
            data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
            data['cstrategy_tc'] = self.amount * \
                                data['strategy_tc'].cumsum().apply(np.exp)
            self.results = data
            # absolute performance of the strategy
            aperf = self.results['cstrategy'].iloc[-1]
            # out-/underperformance of strategy
            operf = aperf - self.results['creturns'].iloc[-1]
            return round(aperf, 2), round(operf, 2)

        OANDA_Base.get_data(self)
        param = run_brute(((1, 100, 1),))
        return int(param[0])

    def optimize_SMA(self):
        def get_new_data():
            raw = self.get_data()
            raw.rename(columns={'c': 'price'}, inplace=True)
            raw['SMA1'] = raw['price'].rolling(1).mean()
            raw['SMA2'] = raw['price'].rolling(2).mean()
            raw = raw[['price', 'return', 'SMA1', 'SMA2']]
            self.data = raw.dropna()
            return self.data

        def set_parameters(SMA1=None, SMA2=None):
            if SMA1 is not None:
                self.SMA1 = SMA1
                self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
            if SMA2 is not None:
                self.SMA2 = SMA2
                self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

        def update_and_run(SMA):
            set_parameters(int(SMA[0]), int(SMA[1]))
            return -run_strategy()[0]

        def run_brute( SMA1_range, SMA2_range):
            opt = brute(update_and_run, (SMA1_range, SMA2_range), finish=None)
            return opt, -update_and_run(opt)

        def run_strategy():
            data = self.data.copy().dropna()
            data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
            data['strategy'] = data['position'].shift(1) * data['return']


            data['strategy_tc'] = np.where(data['position'].diff().fillna(0) != 0,
                                           data['strategy'] - self.ptc,
                                           data['strategy'])

            data['creturns'] = data['return'].cumsum().apply(np.exp)
            data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
            data['cstrategy_tc'] = data['strategy_tc'].cumsum().apply(np.exp)
            self.results = data
            # absolute performance of the strategy
            aperf = data['cstrategy'].iloc[-1]
            # out-/underperformance of strategy
            operf = aperf - data['creturns'].iloc[-1]
            return round(aperf, 2), round(operf, 2)

        get_new_data()
        param = run_brute((5, 100, 1), (5, 100, 1))
        SMA1 = param[0][0].astype(np.int_)
        SMA2 = param[0][1].astype(np.int_)
        if SMA1>SMA2:
            SMA1 = param[0][1].astype(np.int_)
            SMA2 = param[0][0].astype(np.int_)
        return SMA1, SMA2

    def ML_strategy(self):

        BestSMA = OANDA_Base.optimize_single_SMA(self)
        BestMom = OANDA_Base.optimize_momentum(self)
        raw = self.get_data()
        spread = 0.00012
        mean = raw['c'].mean()
        ptc = spread / mean
        raw.rename(columns={'c': self.symbol}, inplace=True)
        data = pd.DataFrame(raw, columns=[self.symbol, 'volume', 'return'])
        #data = pd.DataFrame(raw, raw['c'])
        #data.columns = [self.symbol, ]
        window = 20
        #data['return'] = np.log(data[self.symbol] / data[self.symbol].shift(1))
        data['vol'] = data['return'].rolling(window).std()
        data['mom'] = np.sign(data['return'].rolling(BestMom).mean())
        data['sma'] = data[self.symbol].rolling(BestSMA).mean()
        data['min'] = data[self.symbol].rolling(window).min()
        data['max'] = data[self.symbol].rolling(window).max()

        data.dropna(inplace=True)

        lags = 6

        features = ['return', 'vol', 'mom', 'sma', 'min', 'max']

        cols = []
        for f in features:
            for lag in range(1, lags + 1):
                col = f'{f}_lag_{lag}'
                data[col] = data[f].shift(lag)
                cols.append(col)

        data.dropna(inplace=True)

        data['direction'] = np.where(data['return'] > 0, 1, -1)

        #data = self.analize_data()[0]
        #cols = self.analize_data()[1]
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

        split = int(len(data) * 0.7)

        train = data.iloc[:split].copy()
        mu, std = train.mean(), train.std()
        train_ = (train - mu) / std

        model.fit(train_[cols], train['direction'])
        print('train accuracy')
        print(accuracy_score(train['direction'], model.predict(train_[cols])))

        test = data.iloc[split:].copy()
        test_ = (test - mu) / std

        test['position'] = model.predict(test_[cols])

        print('test accuracy')
        print(accuracy_score(test['direction'], test['position']))

        test['strategy'] = test['position'] * test['return']

        print('Number of trades: ')
        print(sum(test['position'].diff() != 0))

        test['strategy_tc'] = np.where(test['position'].diff() != 0,
                                       test['strategy'] - ptc,
                                       test['strategy'])

        print(test[['return', 'strategy', 'strategy_tc']].sum().apply(np.exp))
        return model, mu, std
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Signal: %s' % (self.symbol))
        ax2 = ax.twinx()
        ax.plot(test.index, test[['return', 'strategy', 'strategy_tc']].cumsum())
        ax2.plot(test.index, test[['position']], color='g')
        ax.set_ylabel('Price', color='b')
        ax2.set_ylabel('Position', color='g')
        plt.tight_layout()
        plt.legend(test[['price', 'strategy', 'strategy_tc']])
        plt.show()
        '''

    def plot_results(self):
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        #fig, ax = plt.subplots()
        plt.style.use('ggplot')

        #title = 'Signal: %s' % (self.symbol)
        #self.results[['position']].plot(title=title, figsize=(10, 6))

        title = '%s' % (self.symbol)
        self.results[['creturns', 'cstrategy', 'position']].plot(title=title, figsize=(10, 6))
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Signal: %s' % (self.symbol))
        ax2 = ax.twinx()
        ax2.plot(self.results.index, self.results[['creturns', 'cstrategy', 'cstrategy_tc']])
        ax.plot(self.results.index,self.results[['position']], color='g')
        #ax.set_xlabel('x-axis', color='r')
        ax2.set_ylabel('Price', color='b')
        ax.set_ylabel('Position', color='g')
        plt.tight_layout()
        plt.legend(self.results[['creturns','cstrategy', 'cstrategy_tc']])
        plt.show()

    def close_out(self, bar):
        ''' Closing aout a long or short position.
        '''
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbos:
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print('=' * 55)
        print('Final balance [$] {:.2f}'.format(self.amount))
        perf = ((self.amount - self.initial_amount) /
                 self.initial_amount * 100)
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('=' * 55)

if __name__ == '__main__':
    #print(api.get_instruments())
    example = OANDA_Base(symbol='EUR_USD', start=monthsAgo(3), end=lastDay(), granularity='H1', price='M', amount=10000, ftc=0.000, ptc=0.0008, verbos=True)
    print(example.analize_data()[0].info())
    #print(example.data.info())
    #print(example.data.tail())

    #print(example.optimize_momentum())
    #example.plot_results()

    #print(example.optimize_single_SMA())
    #example.plot_results()

    #print(example.optimize_SMA())
    #example.plot_results()

    model = example.ML_strategy()[0]
    mu = example.ML_strategy()[1]
    std = example.ML_strategy()[2]
    algorithm = {'model': model, 'mu': mu, 'std': std}
    pickle.dump(algorithm, open('algorithm.pkl', 'wb'))

