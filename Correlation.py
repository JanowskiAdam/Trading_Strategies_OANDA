from OANDA_Historical import *
import statsmodels.formula.api as smf
#print(api.get_instruments())

SPX = OANDA_Base(symbol='SPX500_USD', start= monthsAgo(120), end= lastDay(), granularity='D', price='M', amount=10000, ftc=0.000, ptc=0.0008, verbos=True)
DAX = OANDA_Base(symbol='DE30_EUR', start= monthsAgo(120), end= lastDay(), granularity='D', price='M', amount=10000, ftc=0.000, ptc=0.0008, verbos=True)
df = pd.DataFrame()
df['SPX'] = SPX.get_data()['return']
df['DAX'] = DAX.get_data()['return']

''' plot

df.plot(subplots=True, grid=True, style='b',figsize=(8,6))
plt.show()
'''


result = smf.ols(formula="SPX ~ DAX", data=df).fit()
print(result.params)
print(result.summary())


plt.plot(df['SPX'], df['DAX'], 'r.')
ax = plt.axis()
x = np.linspace(ax[0], ax[1] +0.01)
plt.plot(x, result.params[0] + result.params[1] * x, 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.show()

print(df.corr())


df.rolling(50).corr(pairwise=True).plot(grid=True,style='b')
plt.show()