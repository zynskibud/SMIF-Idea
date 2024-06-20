import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import math
import statistics
from scipy.stats import norm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab
import scipy

class HistoricalVaR:
    
    def __init__(self, tickers, list_weights, initial_portfolio, time):
        self.tickers = tickers
        self.list_weights = list_weights
        self.initial_investment = initial_portfolio
        self.time = time
        
        """chain of method calls"""
        dict_assets = self.data_grabber(tickers)
        assets, df_returns, mean_returns = self.returns(dict_assets)
        df_portfolio = self.portfolio(dict_assets)
        dict_assets, array_weights = self.weights(dict_assets, list_weights)
        df_portfolio, df_returns, series_portfolio_returns = self.portfolio_return(df_portfolio, df_returns, array_weights)
        df_cov = self.covariance_matrix(dict_assets, df_returns)
        portfolio_variance = self.portfolio_variance_calculator(df_cov, array_weights)
        std = self.portfolio_std(portfolio_variance, time)
        Pret = self.portfolio_mean_return(mean_returns, array_weights, time)
        df_returns = self.add_portfolio_returns_to_df_returns(df_returns, series_portfolio_returns)
        dict_VaR = self.historicalVaR(df_returns, initial_portfolio, time)
        dict_CVaR = self.historicalCVaR(df_returns, dict_VaR, initial_portfolio, time)
    
    def hour_wiper(df):
        
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.set_index('Date')
        
        return df

    def data_grabber(self, tickers):
        #Function designed to take any ticker and grab annual data
        dict_assets = {}
        
        for x in tickers:
            try: 
                
                ticker = yf.Ticker(x)
                ticker_data = ticker.history(period = "1y", interval = "1d")
                ticker_data = HistoricalVaR.hour_wiper(ticker_data)
                
                if not ticker_data.empty:
                    dict_assets[x] = ticker_data
                else:
                    print(f"The associated data for {x} is empty")
            except Exception as e:
                print(f"There was an error in fetching {x} data")
                
        return dict_assets

    def returns(self, dict_assets):
        #Calculates daily returns of each asset in the portfolio and puts them into a seperate dataframe
        for x in dict_assets:
            df_common = dict_assets[x]    
            df_common["Returns"] = df_common["Close"].pct_change()
            df_common = df_common.dropna()        
            dict_assets[x] = df_common
        
        dict_returns = {}    
        for x in dict_assets:
            df_common = dict_assets[x]
            series_returns = df_common["Returns"]
            dict_returns[x] = series_returns
            
        df_returns = pd.concat(dict_returns.values(), axis = 1, keys=dict_returns)
        mean_returns = df_returns.mean()
        
        return dict_assets, df_returns, mean_returns

    def weights(self, dict_assets, list_weights):
        #Takes weights and inserts them into dataframes of each asset as a column 
        
        list_counter = 0
        if isinstance(list_weights, list):
            for x in list_weights:
                x = float(x)
                x = x/100
                list_weights[list_counter] = x
                list_counter += 1
            array_weights = np.array(list_weights)
        else:
            raise TypeError("the expected data structure for the weights is a list")
        
        if np.sum(array_weights) == 1:
            pass
        else:
            print(f"The sum of your weights does not equal to 1")
        
        weights_counter = 0
        if  len(array_weights) == len(dict_assets):
            for key in dict_assets:
                df_common = dict_assets[key]
                df_common["Weights"] = list_weights[weights_counter]
                df_common = df_common.dropna()
                dict_assets[key] = df_common
                weights_counter += 1
        else:
            print(f"The number of assets and weights do not match up, please try again")
        
        return dict_assets, array_weights

    def portfolio(self, dict_assets):
        #Assembles the portfolio into a singular dataframe
        df_portfolio = pd.concat(dict_assets.values(), axis = 1, keys=dict_assets.keys())
        df_portfolio = df_portfolio.reset_index()

        return df_portfolio
        
    def portfolio_return(self, df_portfolio, df_returns, array_weights):
        #Calculates the daily returns of the portfolio given the weights
            
        series_portfolio_return = df_returns.dot(array_weights)
        df_portfolio["Portfolio Returns"] = series_portfolio_return
        
        return df_portfolio, df_returns, series_portfolio_return
        
    def covariance_matrix(self, dict_assets, df_returns):
        #Finds the covariance between the returns of each asset in the portfolio        
        df_cov = df_returns.cov()
        
        return df_cov
    
    def portfolio_variance_calculator(self, df_cov, array_weights):
        #Find portfolio variance given covariance matrix and weights of the portfolio
        array_portfolio_variance = np.dot(array_weights.T, np.dot(df_cov, array_weights))
        float_portfolio_variance = float(array_portfolio_variance)
        return float_portfolio_variance  
        
    def portfolio_std(self, float_portfolio_variance, time):
        #Multiply by the square root of time to annualize the volatility
        float_portfolio_std = (math.sqrt(float_portfolio_variance))*np.sqrt(time)
        
        return float_portfolio_std
        
    def portfolio_mean_return(self, mean_returns, array_weights, time):
        #Multiply by the number of trading days to annualize average portfolio return
        P_returns = np.sum(mean_returns*array_weights)*time

        
        return P_returns

    def add_portfolio_returns_to_df_returns(self, df_returns, series_portfolio_returns):
        df_returns["Portfolio Returns"] = series_portfolio_returns
        return df_returns

    def historicalVaR(self, returns, initial_portfolio, time):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the percentile of the distribution at the given alpha confidence level
        """
        list_alpha = [0.9, 0.95, 0.975, 0.99]
        dict_PreVaR = {}
        dict_PostVaR = {}
        
        if isinstance(returns, pd.Series):
            for x in list_alpha:
                VaR = np.percentile(returns, (1-x)*100)
                dict_PreVaR[x] = round((VaR * np.sqrt(time)) * initial_portfolio, 2) * -1
                dict_PostVaR[x] = VaR
        elif isinstance(returns, pd.DataFrame):
            for x in list_alpha:
                VaR = returns.apply(lambda col: np.percentile(col, (1-x)*100))
                dict_PreVaR[x] = round((VaR * np.sqrt(time)) * initial_portfolio, 2) * -1
                dict_PostVaR[x] = VaR
        
        for x in dict_PreVaR:
            dict_PreVaR[x] = (dict_PreVaR[x])["Portfolio Returns"]
            
        for x in dict_PostVaR:
            dict_PostVaR[x] = (dict_PostVaR[x])["Portfolio Returns"]
        
        display(f"The VaR with associated confidence levels for the portfolio is: \n {dict_PreVaR}")
        
        return dict_PostVaR

    def historicalCVaR(self, returns, dict_VaR, initial_portfolio, time):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the CVaR for dataframe / series
        """
        list_alpha = [0.9, 0.95, 0.975, 0.99]
        dict_CVaR = {}

        if isinstance(returns, pd.Series):
            for x in list_alpha:
                belowVaR = returns <= dict_VaR[x]
                CVaR = returns[belowVaR].mean()
                dict_CVaR[x] = round((CVaR * np.sqrt(time)) * initial_portfolio, 2) * -1
            
            for x in dict_CVaR:
                dict_CVaR[x] = (dict_CVaR[x])["Portfolio Returns"]
            display(f"The CVaR with associated confidence levels for the portfolio is: \n {dict_CVaR}")
           
            return dict_CVaR

        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            series_portfolio_return = returns["Portfolio Returns"]
            for x in list_alpha:
                belowVaR = series_portfolio_return <= dict_VaR[x]
                CVaR = returns[belowVaR].mean()
                dict_CVaR[x] = round((CVaR * np.sqrt(time)) * initial_portfolio, 2) * -1
            
            for x in dict_CVaR:
                dict_CVaR[x] = (dict_CVaR[x])["Portfolio Returns"]
            display(f"The CVaR with associated confidence levels for the portfolio is: \n {dict_CVaR}")
            
            return dict_CVaR

        else:
            raise TypeError("Expected returns to be dataframe or series") 

tickers = ["TSLA", "AAPL"]
list_weights = [50,50]
time = 252
initial_portfolio = 100000
portfolio1 = HistoricalVaR(tickers, list_weights, initial_portfolio, time)