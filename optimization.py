import pandas as pd
from simulations import simulate_markets
import itertools
from tqdm import tqdm
from scipy.optimize import minimize
import quantstats as qs
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

prices = pd.read_csv('prices_etf.csv',index_col='Unnamed: 0')
prices = prices.dropna()
prices.index = pd.to_datetime(prices.index)

def calculate_negative_volatility(returns, periods):
        mean = returns.mean()
        N = len(returns[returns<0])
        negative_variance = ((returns[returns<0]-mean)**2).sum()/(N-1)
        negative_volatility = np.sqrt(negative_variance)
        return negative_volatility*np.sqrt(periods) 



results = []
for _ in tqdm(range(10000)):
    
    weights = np.random.rand(len(prices.columns))
    weights = weights/weights.sum()
    
    portfolio = (weights*(1+prices.pct_change().loc[:datetime(2010,1,1)]).cumprod().dropna()).sum(axis=1)
    
    results.append({'weights': weights,
                    'tot_return':portfolio.iloc[-1],
                    'cvar':qs.stats.cvar(portfolio.pct_change()),
                    'maxDD':qs.stats.max_drawdown(portfolio.pct_change()),
                    'neg_volatility':calculate_negative_volatility(portfolio.pct_change(),periods=252)})

    
results = pd.DataFrame(results)

plt.scatter(results['neg_volatility'],results['maxDD'],c=results['cvar'])
plt.colorbar()

weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))).idxmax(),'weights']
portfolio = (weights*(1+prices.pct_change().loc[:datetime(2010,1,1)]).cumprod().dropna()).sum(axis=1)
weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))*results['tot_return']**3).idxmax(),'weights']
portfolio2 = (weights*(1+prices.pct_change().loc[:datetime(2010,1,1)]).cumprod().dropna()).sum(axis=1)
weights = np.ones(len(prices.columns))
weights = weights/weights.sum()
benchmark = (weights*(1+prices.pct_change().loc[:datetime(2010,1,1)]).cumprod().dropna()).sum(axis=1)

portfolio.plot(label='1/(maxDD x neg_vol x cvar)')
portfolio2.plot(label='calmar x sortino x return/cvar')
benchmark.plot(label='benchmark')
plt.title('train sample fit with data')
plt.legend()



weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))).idxmax(),'weights']
portfolio = (weights*(1+prices.pct_change().loc[datetime(2010,1,1):]).cumprod().dropna()).sum(axis=1)
weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))*results['tot_return']**3).idxmax(),'weights']
portfolio2 = (weights*(1+prices.pct_change().loc[datetime(2010,1,1):]).cumprod().dropna()).sum(axis=1)
weights = np.ones(len(prices.columns))
weights = weights/weights.sum()
benchmark = (weights*(1+prices.pct_change().loc[datetime(2010,1,1):]).cumprod().dropna()).sum(axis=1)

portfolio.plot(label='1/(maxDD x neg_vol x cvar)')
portfolio2.plot(label='calmar x sortino x return/cvar')
benchmark.plot(label='benchmark')
plt.title('train sample fit with data')
plt.legend()

simulated_prices = simulate_markets(prices.loc[:datetime(2010,1,1)],n_simulations=20,simulation_length=1000)

results = []
for _ in tqdm(range(10000)):
    
    weights = np.random.uniform(low=-1.,high=1.,size=len(prices.columns))
    weights = np.round(weights/weights.sum(),2)
    tot_return_mean = 0
    mean_return_mean = 0
    cvar_mean = 0
    maxDD_mean = 0
    neg_volatility_mean = 0
    n = 1/len(simulated_prices)
    for sim_prices in simulated_prices:
        
        portfolio = (weights*(1+sim_prices.pct_change()).cumprod().dropna()).sum(axis=1)
        tot_return_mean += portfolio.iloc[-1]
        mean_return_mean += portfolio.mean()
        cvar_mean += qs.stats.cvar(portfolio.pct_change())
        maxDD_mean += qs.stats.max_drawdown(portfolio.pct_change())
        neg_volatility_mean += calculate_negative_volatility(portfolio.pct_change(),periods=252)
    
    results.append({'weights': weights,
                    'tot_return':tot_return_mean*n,
                    'mean_return':mean_return_mean*n,
                    'cvar':cvar_mean*n,
                    'maxDD':maxDD_mean*n,
                    'neg_volatility':neg_volatility_mean*n})

    
results = pd.DataFrame(results)

# plt.scatter(results['neg_volatility'],results['maxDD'],c=results['cvar'])
# plt.colorbar()

weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))).idxmax(),'weights']
portfolio = (1+(weights*prices.pct_change().loc[:datetime(2010,1,1)]).sum(axis=1)).cumprod()
weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))*results['tot_return']**3).idxmax(),'weights']
portfolio2 = (1+(weights*prices.pct_change().loc[:datetime(2010,1,1)]).sum(axis=1)).cumprod()
weights = np.ones(len(prices.columns))
weights = weights/weights.sum()
benchmark = (1+(weights*prices.pct_change().loc[:datetime(2010,1,1)]).sum(axis=1)).cumprod()

#(1+prices.pct_change().loc[datetime(2010,1,1):]).cumprod().plot()
portfolio.plot(c='blue',label='1/(maxDD x neg_vol x cvar)')
portfolio2.plot(c='orange',label='calmar x sortino x return/cvar')
benchmark.plot(c='k',label='benchmark')
plt.title('train sample fit by simulations')
plt.legend()
plt.show()

cum_ret = 1 
cum_ret2 = 1 
cum_b = 1 
for i in range(0,len(weights*prices.pct_change().loc[datetime(2010,1,1):]),64):
    weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))).idxmax(),'weights']
    portfolio = (1+(weights*prices.pct_change().loc[datetime(2010,1,1):]).iloc[i:i+64].sum(axis=1)).cumprod()
    weights = results.loc[((1/results['neg_volatility'])*(1/abs(results['maxDD']))*(1/abs(results['cvar']))*results['mean_return']**3).idxmax(),'weights']
    portfolio2 = (1+(weights*prices.pct_change().loc[datetime(2010,1,1):]).iloc[i:i+64].sum(axis=1)).cumprod()
    weights = np.ones(len(prices.columns))
    weights = weights/weights.sum()
    benchmark = (1+(weights*prices.pct_change().loc[datetime(2010,1,1):]).iloc[i:i+64].sum(axis=1)).cumprod()
    cum_ret *= portfolio.iloc[-1]
    cum_ret2 *= portfolio2.iloc[-1]
    cum_b *= benchmark.iloc[-1]
    
    #(1+prices.pct_change().loc[datetime(2010,1,1):]).cumprod().plot()
    portfolio.plot(c='blue',label='1/(maxDD x neg_vol x cvar)')
    #portfolio2.plot(c='orange',label='calmar x sortino x return/cvar')
    benchmark.plot(c='k',label='benchmark')
    plt.title('train sample fit by simulations')
    plt.legend()
    plt.show()

portfolio.index = [x.astimezone(None) for x in portfolio.index]
portfolio2.index = [x.astimezone(None) for x in portfolio2.index]
benchmark.index = [x.astimezone(None) for x in benchmark.index]
qs.reports.full(portfolio.pct_change(),benchmark=benchmark.pct_change())





def function_to_optimize(*weights):
    weights = np.array(weights)
    weights = weights/weights.sum()
    tot_return_mean = 0
    mean_return_mean = 0
    cvar_mean = 0
    maxDD_mean = 0
    neg_volatility_mean = 0
    for sim_prices in simulated_prices:
        
        portfolio = (weights*(1+sim_prices.pct_change()).cumprod().dropna()).sum(axis=1)
        tot_return_mean += portfolio.iloc[-1]
        mean_return_mean += portfolio.mean()
        cvar_mean += qs.stats.cvar(portfolio.pct_change())
        maxDD_mean += qs.stats.max_drawdown(portfolio.pct_change())
        neg_volatility_mean += calculate_negative_volatility(portfolio.pct_change(),periods=252)

    return ((tot_return_mean)**3)/(abs(cvar_mean)*abs(maxDD_mean)*neg_volatility_mean)

