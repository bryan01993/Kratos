# regime_hmm_train.py
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime
import pickle
import click
import seaborn as sns
import qstrader
from qstrader import settings
import warnings
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from collections import deque
#from qstrader.event import (SignalEvent, EventType,OrderEvent)
#from qstrader.strategy.base import AbstractStrategy
#from qstrader.event import OrderEvent
#from qstrader.price_parser import PriceParser
#from qstrader.risk_manager.base import AbstractRiskManager
#from qstrader.compat import queue
#from qstrader.price_parser import PriceParser
#from qstrader.price_handler.yahoo_daily_csv_bar import YahooDailyCsvBarPriceHandler
#from qstrader.strategy import Strategies, DisplayStrategy
#from qstrader.position_sizer.naive import NaivePositionSizer
#from qstrader.risk_manager.example import ExampleRiskManager
#from qstrader.portfolio_handler import PortfolioHandler
#from qstrader.compliance.example import ExampleCompliance
#from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler
#from qstrader.statistics.tearsheet import TearsheetStatistics
#from qstrader.trading_session.backtest import Backtest
#from regime_hmm_strategy import MovingAverageCrossStrategy
#from regime_hmm_risk_manager import RegimeHMMRiskManager

def obtain_prices_df(csv_filepath, end_date):
    """
    Obtain the prices DataFrame from the CSV file,
    filter by the end date and calculate the
    percentage returns.
    """
    df = pd.read_csv(csv_filepath, header=0,names=["Date", "Open", "High", "Low","Close", "Volume", "Adj Close"],index_col="Date", parse_dates=True)
    df["Returns"] = df["Adj Close"].pct_change()
    df = df[:end_date.strftime("%Y-%m-%d")]
    df.dropna(inplace=True)
    return df

def plot_in_sample_hidden_states(hmm_model, df):
    """
    Plot the adjusted closing prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(hmm_model.n_components,sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(df.index[mask],df["Adj Close"][mask],".", linestyle='none',c=colour)
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

if __name__ == "__main__":
    # Hides deprecation warnings for sklearn
    warnings.filterwarnings("ignore")
    # Create the SPY dataframe from the Yahoo Finance CSV
    # and correctly format the returns for use in the HMM
    csv_filepath = "C:/Users/bryan/OneDrive/Desktop/Para Imprimir/^GSPC.csv"
    pickle_path = "C:/Users/bryan/OneDrive/Desktop/Para Imprimir/hmm_model_spy.pkl"
    end_date = datetime.datetime(2019, 12, 31)
    spy = obtain_prices_df(csv_filepath, end_date)
    rets = np.column_stack([spy["Returns"]])
    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10).fit(rets)
    print("Model Score:", hmm_model.score(rets))
    # Plot the in sample hidden states closing values
    plot_in_sample_hidden_states(hmm_model, spy)
    print("Pickling HMM model...")
    pickle.dump(hmm_model, open(pickle_path, "wb"))
    print("...HMM model pickled.")


