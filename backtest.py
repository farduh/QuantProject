# -*- coding: utf-8 -*-
"""
BACKTEST
"""


import pandas as pd
import itertools
from tqdm import tqdm
from scipy.optimize import minimize
import quantstats as qs
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.stats import johnsonsu
import matplotlib.pyplot as plt


class BackTest():
    def __init__(self,
                 metrics_function,
                 bounds=None,
                 test_periods = 63,
                 train_periods = 0,
                 period_number_to_train_again = 10,
                 simulation_variable_mean=True,
                 simulation_length=1000,
                 n_simulations=100,
             ):
        """
        

        Parameters
        ----------
        metrics_function : TYPE
            Funcion a optimzar.
        bounds : TYPE, optional
            limites de cada peso posible. The default is None.
        test_periods : TYPE, optional
            periodo de rebalanceo. The default is 63.
        train_periods : TYPE, optional
            DESCRIPTION. The default is 0.
        period_number_to_train_again : TYPE, optional
            DESCRIPTION. The default is 10.
        simulation_variable_mean : TYPE, optional
            DESCRIPTION. The default is True.
        simulation_length : TYPE, optional
            DESCRIPTION. The default is 1000.
        n_simulations : TYPE, optional
            DESCRIPTION. The default is 100.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if not bounds:
            self.bounds = [(0.0,0.3) for _ in range(len(prices.columns))]
        self.metrics_function = metrics_function
        self.test_periods = test_periods
        self.train_periods = train_periods
        self.period_number_to_train_again = period_number_to_train_again
        self.simulation_length = simulation_length
        self.n_simulations = n_simulations
        
    def run(self,prices,start_date):
        returns = pd.Series()
        weights_list = []
        period_number = 0
        test_sample_length = len(prices.loc[start_date:]) 
        for i in range(0,test_sample_length,self.test_periods):
            
            start_test_period = prices.loc[start_date:].index[i]
            end_test_period = prices.loc[start_date:].index[min(i+self.test_periods,test_sample_length-1)]
            start_train_period = prices.loc[:start_test_period].index[-self.train_periods]
            #train
            if not period_number % self.period_number_to_train_again:
                weights = self.get_optimal_weights(prices.loc[start_train_period:start_test_period],self.test_periods)
            #test
            returns = returns.append(self.get_returns(weights,prices.loc[start_test_period:end_test_period]))
            weights_list.append({'start_date':start_test_period,'weights_list':weights})
            period_number +=1
        weights_df = pd.DataFrame(weights_list)
        weights_df = weights_df.set_index('start_date')
        return returns,weights_df
    
    def get_returns(self,weights,prices):
        portfolio_returns = (weights*(prices/prices.iloc[0])).sum(axis=1).pct_change().dropna()
        return portfolio_returns
    
    def get_optimal_weights(self,prices,test_periods):
        simulated_prices = self.simulate_markets(prices,n_simulations=100,simulation_length=1000,test_period=64)
        cons=({'type':'eq','fun':lambda x:x.sum()-1})
        weights0 = np.ones(len(prices.columns))/len(prices.columns)
        best_weights = np.zeros(len(prices.columns))
        number_of_simulations = len(simulated_prices)
        factor = 1/number_of_simulations
        for sim_prices in simulated_prices:
            
            
            opt_result = minimize(self.function_to_optimize,\
                    x0=weights0,\
                    bounds=self.bounds,\
                    method='SLSQP',\
                    constraints=cons,\
                    args=(sim_prices),options = {'eps':0.1})
            
            best_weights += factor * opt_result.x
        # best_metric = 999
        # for _ in tqdm(range(1000)):
        #     weights = np.random.uniform(low=0,high=1.,size=len(prices.columns))
        #     weights = np.round(weights/weights.sum(),2)
        #     metric = function_to_optimize(weights,simulated_prices)
        #     if metric < best_metric:
        #         best_weights = weights
        #         best_metric = metric
        return best_weights
        
    
    def function_to_optimize(self,weights,sim_prices):
        metric = 0
        portfolio = (weights*(sim_prices/sim_prices.iloc[0])).sum(axis=1)
        returns = portfolio.pct_change().dropna()
        metric = self.metrics_function(returns)
        return metric
    
    # def function_to_optimize(self,weights,simulations):
    #     metric = 0
    #     number_of_simulations = len(simulations)
    #     factor = 1/number_of_simulations
    #     for sim_prices in simulations:
    #         portfolio = (weights*(sim_prices/sim_prices.iloc[0])).sum(axis=1)
    #         returns = portfolio.pct_change().dropna()
    #         metric += self.metrics_function(returns)*factor
    #     return metric
    
    
    def simulate_means(self,returns,n_simulations=500,test_period=64):
        returns_means = returns.rolling(test_period).mean().dropna()
        pca = PCA()#inicializo pca
        pca_transform = pca.fit_transform(returns_means)#ajusto pca a los datos
        pca_transform = pd.DataFrame(pca_transform,index=returns_means.index)#lo transformo en un DataFrame
        functions_params = {}
        for cols in pca_transform.columns:
            params = johnsonsu.fit(pca_transform[cols],floc=0) #ajusto la funcion los componentes principales
            functions_params[cols] = params
            
        simulated_pca = pd.DataFrame()
        for cols in pca_transform.columns:
            simulated_pca[cols] = johnsonsu.rvs(*functions_params[cols],n_simulations)#simulo los datos para cada columna
        simulated_means = np.dot(simulated_pca,pca.components_)
        simulated_means = pd.DataFrame(simulated_means,columns=returns.columns)
        
        simulated_means = simulated_means + pca.mean_
        return simulated_means
    
    def simulate_markets(self,prices,n_simulations=500,simulation_length=1000,test_period=64):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with stock prices.
        n_simulations : int, optional
            number of simulations that will be generated. The default is 50.
    
        Returns
        -------
        simulations : list
            list with DataFrame of simulated prices.
    
        """
        returns = np.log(1+prices.pct_change().dropna())
        simulated_means = self.simulate_means(returns,n_simulations,test_period)
        
        returns = returns - returns.mean() 
        pca = PCA()#inicializo pca
        pca_transform = pca.fit_transform(returns)#ajusto pca a los datos
        pca_transform = pd.DataFrame(pca_transform,index=returns.index)#lo transformo en un DataFrame
        functions_params = {}
        for cols in pca_transform.columns:
            params = johnsonsu.fit(pca_transform[cols],floc=0) #ajusto la funcion los componentes principales
            functions_params[cols] = params
            
        simulations = []
        
        for i in tqdm(range(0,n_simulations)):
            simulated_pca = pd.DataFrame()
            for cols in pca_transform.columns:
                simulated_pca[cols] = johnsonsu.rvs(*functions_params[cols],size=simulation_length)#simulo los datos para cada columna
            simulated_returns = np.dot(simulated_pca,pca.components_)
            date_range_index = pd.date_range(periods=simulation_length,end=returns.iloc[-1].name,freq='d')
            simulated_returns = pd.DataFrame(simulated_returns,index=date_range_index,columns=returns.columns)
            
            simulated_returns = simulated_returns + simulated_means.loc[i]
            
            simulated_normalized_prices = (np.exp(simulated_returns)).cumprod()
            simulated_prices = simulated_normalized_prices.multiply(prices.iloc[0])
            
            simulations.append(simulated_prices)
        return simulations


prices = pd.read_csv('prices_etf.csv',index_col='Unnamed: 0')
prices = prices.dropna()
prices.index = pd.to_datetime(prices.index)
prices.index = [x.astimezone(None) for x in prices.index]


def calculate_negative_volatility(returns, periods=252):
        mean = returns.mean()
        N = len(returns[returns<0])
        negative_variance = ((returns[returns<0]-mean)**2).sum()/(N-1)
        negative_volatility = np.sqrt(negative_variance)
        return negative_volatility*np.sqrt(periods) 


def metric_to_minimize(returns):
    exp_return = (-1)*252*(np.exp(np.log(1+returns).skew())-1)
    
    metric = (-1)*qs.stats.cvar(returns)
    #metric *= (-1)*qs.stats.max_drawdown(returns)
    metric *= calculate_negative_volatility(returns)
    return metric/(exp_return**2)

bt = BackTest(metric_to_minimize)
returns,weights_df = bt.run(prices,datetime(2010,1,1))
 
weights = np.ones(len(prices.columns))/len(prices.columns)
benchmark = (weights*(prices.loc[datetime(2010,1,1):]/prices.loc[datetime(2010,1,1):].iloc[0])).sum(axis=1)

benchmark.plot(label='benchmark')
(1+returns).cumprod().plot(c='k',label='strategy')
plt.legend()
plt.show()

qs.reports.full(returns,benchmark)





w0=np.ones(14)/14
cons=({'type':'eq','fun':lambda x:x.sum()-1})

bounds = [(0.0,1) for _ in range(len(prices.columns))]
def function_to_optimize(w,prices):
    
    weighted_cov = np.dot(w,prices.pct_change().dropna().cov())
    metric = ((weighted_cov*w)-np.mean(weighted_cov*w))**2
    print(np.sum(abs(metric))*1e10)
    return np.sum(abs(metric))*1e10
    


opt_result = minimize(function_to_optimize,\
    x0=w0,\
    bounds=bounds,\
    method='SLSQP',\
    constraints=cons,\
    args=(prices),options = {'eps':0.1})

    
w = opt_result.x



sims = bt.simulate_markets(prices.loc[:datetime(2010,1,1)],n_simulations=3,simulation_length=64,test_period=64)



i=1
plt.figure(figsize=(20,16))
plt.subplot(2,2,i)
plt.title('real data')
for col in prices.columns:
    (prices[col].loc[datetime(2010,1,1):].iloc[:64]/prices[col].loc[datetime(2010,1,1):].iloc[0]).plot(label=col)
    plt.legend()
    
for sim in sims:
    i+=1
        plt.subplot(2,2,i)
    plt.title(f'simulation {i-1}')
    for col in prices.columns:
        (sim[col]/sim[col].iloc[0]).plot()
    

