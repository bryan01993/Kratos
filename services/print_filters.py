class PrintFilters:
    def __init__(self, dto):
        self.dto = dto

    def print_filters(self):
        print('Filtered by:')
        print('Min. Net Profit:', self.dto.filter_net_profit_phase1)
        print('Min. Exp. Payoff:', self.dto.filter_expected_payoff_phase1)
        print('Min. Profit Factor:', self.dto.filter_profit_factor_phase1)
        print('Min. Custom:', self.dto.filter_custom_phase1)
        print('Max. Equity DD:', self.dto.filter_equity_dd_phase1)
        print('Min. Trades:', self.dto.filter_trades_phase1)

    def print_forward_filters(self):
        print('Filtered by:')
        print('Forward Min. Net Profit:', self.dto.forward_filter_net_profit_phase1)
        print('Forward Min. Exp. Payoff:', self.dto.forward_filter_expected_payoff_phase1)
        print('Forward Min. Profit Factor:', self.dto.forward_filter_profit_factor_phase1)
        print('Forward Min. Custom:', self.dto.forward_filter_custom_phase1)
        print('Forward Max. Equity DD:', self.dto.forward_filter_equity_dd_phase1)
        print('Forward Min. Trades:', self.dto.forward_filter_trades_phase1)
