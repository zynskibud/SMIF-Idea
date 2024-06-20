import pandas as pd
import numpy as np
import yfinance as yf
import math
import statistics
from scipy.stats import norm, t
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab
import scipy

class ParametricVaR:
    
    def __init__(self, tickers, list_weights, initial_portfolio, time):
        self.tickers = tickers
        self.list_weights = list_weights
        self.initial_portfolio = initial_portfolio
        self.time = time 
              
        """chain of method calls"""
        dict_assets = self.data_grabber(tickers)
        dict_assets, df_returns, mean_returns = self.returns(dict_assets)
        df_portfolio = self.portfolio(dict_assets)
        dict_assets, array_weights = self.weights(dict_assets, list_weights)
        df_portfolio, df_returns, series_portfolio_returns = self.portfolio_return(df_portfolio, df_returns, array_weights)
        df_cov = self.covariance_matrix(dict_assets, df_returns)
        portfolio_variance = self.portfolio_variance_calculator(df_cov, array_weights)
        std = self.portfolio_std(portfolio_variance, time)
        Pret = self.portfolio_mean_return(mean_returns, array_weights, time)
        VaR = self.Var_Parametric(Pret, std, "normal", initial_portfolio)
        CVaR = self.CVar_Parametric(Pret, std, "normal", initial_portfolio)
    
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
                ticker_data = ParametricVaR.hour_wiper(ticker_data)
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

    def Var_Parametric(self, portfolio_return, std, distribution, initial_portfolio, dof = 6):
        """Calculate the portfolio VaR Given a distribution with known parameters"""
        list_alpha = [0.9, 0.95, 0.975, 0.99]
        dict_VaR = {}
        
        for x in list_alpha:
            if distribution == "normal":
                VaR = round((norm.ppf(1-x) * std - portfolio_return)*initial_portfolio, 2)*-1
            elif distribution == "t-distribution":
                nu = dof
                VaR = round(((np.sqrt((nu-2)/nu) * t.ppf(1-x, nu)* std - portfolio_return) * initial_portfolio, 2))*-1
            else:
                raise TypeError("Expected distribution to be normal or t-distribution")
            dict_VaR[x] = VaR
        print(f"Annual Var with associated confidence levels: {dict_VaR}")    
            
        return dict_VaR

    def CVar_Parametric(self, portfolio_return, std, distribution, initial_portfolio, dof = 6):
        """Calculate the portfolio CVaR Given a distribution with known parameters"""
        list_alpha = [0.9, 0.95, 0.975, 0.99]
        dict_CVaR = {}
        
        for x in list_alpha:
            if distribution == "normal":
                CVaR = round((((norm.pdf(norm.ppf(x)) / (1 - x)) * std) - portfolio_return) * initial_portfolio, 2)
            elif distribution == "t-distribution":
                nu = dof
                x_anu = t.ppf(1-x, nu)
                CVaR = round((-1/(1-x) * (1-nu)**-1 * (nu-2+x_anu**2) * t.ppf(x_anu, nu)* std - portfolio_return)*initial_portfolio, 2)*-1
            else:
                raise TypeError("Expected distribution to be normal or t-distribution")
            dict_CVaR[x] = CVaR
        print(f"Annual CVar with associated confidence levels: {dict_CVaR}")    
            
        return dict_CVaR

tickers = ["HSBC", "TSLA", "LYG", "RBSPF"]
list_weights = ["30","20","40","10"]
initial_investment = 100000
time = 252

portfolio_1 = ParametricVaR(tickers, list_weights, initial_investment, time)