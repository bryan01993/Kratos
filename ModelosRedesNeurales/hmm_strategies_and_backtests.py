# regime_hmm_strategy.py
class MovingAverageCrossStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    short_window - Lookback period for short moving average
    long_window - Lookback period for long moving average
    """
    def __init__(self, tickers,events_queue, base_quantity,short_window=10, long_window=30):
        self.tickers = tickers
        self.events_queue = events_queue
        self.base_quantity = base_quantity
        self.short_window = short_window
        self.long_window = long_window
        self.bars = 0
        self.invested = False
        self.sw_bars = deque(maxlen=self.short_window)
        self.lw_bars = deque(maxlen=self.long_window)

    def calculate_signals(self, event):
        # Applies SMA to first ticker
        ticker = self.tickers[0]
        if event.type == EventType.BAR and event.ticker == ticker:
            # Add latest adjusted closing price to the
            # short and long window bars
            price = event.adj_close_price / float(PriceParser.PRICE_MULTIPLIER)
            self.lw_bars.append(price)
            if self.bars > self.long_window - self.short_window:
                self.sw_bars.append(price)
                # Enough bars are present for trading
            if self.bars > self.long_window:
                # Calculate the simple moving averages
                short_sma = np.mean(self.sw_bars)
                long_sma = np.mean(self.lw_bars)
                # Trading signals based on moving average cross
                if short_sma > long_sma and not self.invested:
                    print("LONG: %s" % event.time)
                    signal = SignalEvent(ticker, "BOT", self.base_quantity)
                    self.events_queue.put(signal)
                    self.invested = True
                elif short_sma < long_sma and self.invested:
                    print("SHORT: %s" % event.time)
                    signal = SignalEvent(ticker, "SLD", self.base_quantity)
                    self.events_queue.put(signal)
                    self.invested = False
            self.bars += 1

# regime_hmm_risk_manager.py

class RegimeHMMRiskManager(AbstractRiskManager):
    """
    Utilises a previously fitted Hidden Markov Model
    as a regime detection mechanism. The risk manager
    ignores orders that occur during a non-desirable
    regime.
    It also accounts for the fact that a trade may
    straddle two separate regimes. If a close order
    is received in the undesirable regime, and the
    order is open, it will be closed, but no new
    orders are generated until the desirable regime
    is achieved.
    """
    def __init__(self, hmm_model):
        self.hmm_model = hmm_model
        self.invested = False

    def determine_regime(self, price_handler, sized_order):
        """
        Determines the predicted regime by making a prediction
        on the adjusted closing returns from the price handler
        object and then taking the final entry integer as
        the "hidden regime state".
        """
        returns = np.column_stack([np.array(price_handler.adj_close_returns)])
        hidden_state = self.hmm_model.predict(returns)[-1]
        return hidden_state

    def refine_orders(self, portfolio, sized_order):
        """
        Uses the Hidden Markov Model with the percentage returns
        to predict the current regime, either 0 for desirable or
        1 for undesirable. Long entry trades will only be carried
        out in regime 0, but closing trades are allowed in regime 1.
        """
        # Determine the HMM predicted regime as an integer
        # equal to 0 (desirable) or 1 (undesirable)
        price_handler = portfolio.price_handler
        regime = self.determine_regime(price_handler, sized_order)
        action = sized_order.action
        # Create the order event, irrespective of the regime.
        # It will only be returned if the correct conditions
        # are met below.
        order_event = OrderEvent(sized_order.ticker,sized_order.action,sized_order.quantity)
        # If in the desirable regime, let buy and sell orders
        # work as normal for a long-only trend following strategy
        if regime == 0:
            if action == "BOT":
                self.invested = True
                return [order_event]
            elif action == "SLD":
                if self.invested == True:
                    self.invested = False
                    return [order_event]
                else:
                    return []
        # If in the undesirable regime, do not allow any buy orders
        # and only let sold/close orders through if the strategy
        # is already invested (from a previous desirable regime)
        elif regime == 1:
            if action == "BOT":
                self.invested = False
                return []
            elif action == "SLD":
                if self.invested == True:
                    self.invested = False
                return [order_event]
            else:
                return []


# regime_hmm_backtest.py

def run(config, testing, tickers, filename):
    # Set up variables needed for backtest
    pickle_path = "/path/to/your/model/hmm_model_spy.pkl"
    events_queue = queue.Queue()
    csv_dir = config.CSV_DATA_DIR
    initial_equity = PriceParser.parse(500000.00)
    # Use Yahoo Daily Price Handler
    start_date = datetime.datetime(2011, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    price_handler = YahooDailyCsvBarPriceHandler(csv_dir, events_queue, tickers,start_date=start_date, end_date=end_date,calc_adj_returns=True)
    # Use the Moving Average Crossover trading strategy
    base_quantity = 10000
    strategy = MovingAverageCrossStrategy(tickers, events_queue, base_quantity,short_window=10, long_window=30)
    strategy = Strategies(strategy, DisplayStrategy())
    # Use the Naive Position Sizer
    # where suggested quantities are followed
    position_sizer = NaivePositionSizer()
    # Use regime detection HMM risk manager
    hmm_model = pickle.load(open(pickle_path, "rb"))
    risk_manager = RegimeHMMRiskManager(hmm_model)
    # Use an example Risk Manager
    #risk_manager = ExampleRiskManager()
    # Use the default Portfolio Handler
    portfolio_handler = PortfolioHandler(initial_equity, events_queue, price_handler,position_sizer, risk_manager)
    # Use the ExampleCompliance component
    compliance = ExampleCompliance(config)
    # Use a simulated IB Execution Handler
    execution_handler = IBSimulatedExecutionHandler(events_queue, price_handler, compliance)
    # Use the Tearsheet Statistics
    title = ["Trend Following Regime Detection with HMM"]
    statistics = TearsheetStatistics(config, portfolio_handler, title,benchmark="SPY")
    # Set up the backtest
    backtest = Backtest(price_handler, strategy,portfolio_handler, execution_handler,position_sizer, risk_manager,statistics, initial_equity)
    results = backtest.simulate_trading(testing=testing)
    statistics.save(filename)
    return results

@click.command()
@click.option('--config',default=settings.DEFAULT_CONFIG_FILENAME,help='Config filename')
@click.option('--testing/--no-testing',default=False, help='Enable testing mode')
@click.option('--tickers', default='SPY',help='Tickers (use comma)')
@click.option('--filename', default='',help='Pickle (.pkl) statistics filename')

def main(config, testing, tickers, filename):
    tickers = tickers.split(",")
    config = settings.from_file(config, testing)
    run(config, testing, tickers, filename)

if __name__ == "__main__":
    main()