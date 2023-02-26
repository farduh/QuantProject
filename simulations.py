import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import johnsonsu
from tqdm import tqdm


def simulate_means(returns,n_simulations=500,test_period=64):
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

def simulate_markets(prices,n_simulations=500,simulation_length=1000,test_period=64):
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
    simulated_means = simulate_means(returns,n_simulations,test_period)
    
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


        
        