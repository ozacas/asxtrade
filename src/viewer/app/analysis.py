"""
Responsible for performing computations to support a visualisation/analysis result
"""
from collections import defaultdict, OrderedDict
import secrets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pypfopt.expected_returns import mean_historical_return, returns_from_prices
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.plotting import plot_covariance
from pypfopt.hierarchical_portfolio import HRPOpt
from app.models import company_prices, day_low_high, stocks_by_sector, Timeframe, validate_date
from app.data import cache_plot
from app.messages import warning

def as_css_class(thirty_day_slope, three_hundred_day_slope):
    if thirty_day_slope > 0.0 and three_hundred_day_slope < 0.0:
        return "recent-upward-trend"
    elif thirty_day_slope < 0.0 and three_hundred_day_slope > 0.0:
        return "recent-downward-trend"
    else:
        return "none"

def calculate_trends(cumulative_change_df, watchlist_stocks):
    trends = {}  # stock -> (slope, nrmse) pairs
    for stock in watchlist_stocks:
        series = cumulative_change_df.loc[stock]
        n = len(series)
        assert n > 0
        series30 = series[-30:]
        coefficients, residuals, _, _, _ = np.polyfit(range(n), series, 1, full=True)
        coeff30, resid30, _, _, _ = np.polyfit(range(len(series30)), series30, 1, full=True)
        assert resid30 is not None
        mse = residuals[0] / n
        series_range = series.max() - series.min()
        if series_range == 0.0:
            continue
        nrmse = np.sqrt(mse) / series_range
        # ignore stocks which are barely moving either way
        if any([np.isnan(coefficients[0]), np.isnan(nrmse), abs(coefficients[0]) < 0.01]):
            pass
        else:
            trends[stock] = (coefficients[0],
                             nrmse,
                             '{:.2f}'.format(coeff30[0]) if not np.isnan(coeff30[0]) else '',
                             as_css_class(coeff30[0], coefficients[0]))
    # sort by ascending overall slope (regardless of NRMSE)
    return OrderedDict(sorted(trends.items(), key=lambda t: t[1][0]))

def rank_cumulative_change(df: pd.DataFrame, timeframe: Timeframe):
    cum_sum = defaultdict(float)
    #print(df)
    for date in filter(lambda k: k in df.columns, timeframe.all_dates()):
        for code, price_change in df[date].fillna(0.0).iteritems():
            cum_sum[code] += price_change
        rank = pd.Series(cum_sum).rank(method='first', ascending=False)
        df[date] = rank

    all_available_dates = df.columns
    avgs = df.mean(axis=1) # NB: do this BEFORE adding columns...
    assert len(avgs) == len(df)
    df['x'] = all_available_dates[-1]
    df['y'] = df[all_available_dates[-1]]

    bins = ['top', 'bin2', 'bin3', 'bin4', 'bin5', 'bottom']
    average_rank_binned = pd.cut(avgs, len(bins), bins)
    assert len(average_rank_binned) == len(df)
    df['bin'] = average_rank_binned
    df['asx_code'] = df.index
    stock_sector_df = stocks_by_sector() # make one DB call (cached) rather than lots of round-trips
    #print(stock_sector_df)
    stock_sector_df = stock_sector_df.set_index('asx_code')
    #print(df.index)
    df['sector'] = [stock_sector_df.loc[code].sector_name for code in df.index]
    df = pd.melt(df, id_vars=['asx_code', 'bin', 'sector', 'x', 'y'],
                 var_name='date',
                 value_name='rank',
                 value_vars=all_available_dates)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df['x'] = pd.to_datetime(df['x'], format="%Y-%m-%d")
    return df

def rule_move_up(state: dict):
    """
    Return 1 if the state indicates the stock moved up, 0 otherwise
    """
    assert state is not None

    move = state.get('stock_move')
    return 1 if move > 0.0 else 0

def rule_market_avg(state: dict):
    """
    If the magnitude of the stock move is greater than the magnitude of the
    market move, then award two points in the direction of the move. Otherwise 0
    """
    assert state is not None

    move = state.get('stock_move')
    market_avg = state.get('market_avg')
    ret = 0
    if abs(move) >= abs(market_avg):
        ret = np.sign(move) * 2
    return ret

def rule_sector_avg(state: dict):
    """
    Award 3 points for a stock beating the sector average on a given day or
    detract three points if it falls more than the magnitude of the sector
    """
    assert state is not None

    move = state.get('stock_move')
    sector_avg = state.get('sector_avg')

    if abs(move) >= abs(sector_avg):
        return np.sign(move) * 3
    return 0

def rule_signif_move(state: dict):
    """
    Only ~10% of stocks will move by more than 2% percent on a given day, so give a point for that...
    """
    assert state is not None

    move = state.get('stock_move') # percentage move
    if move >= 2.0:
        return 1
    elif move <= -2.0:
        return -1
    return 0

def rule_against_market(state: dict):
    """
    Award 1 point (in the direction of the move) if the stock moves against
    the overall market AND sector in defiance of the global sentiment (eg. up on a down day)
    """
    assert state is not None
    stock_move = state.get('stock_move')
    market_avg = state.get('market_avg')
    sector_avg = state.get('sector_avg')
    if stock_move > 0.0 and market_avg < 0.0 and sector_avg < 0.0:
        return 1
    elif stock_move < 0.0 and market_avg > 0.0 and sector_avg > 0.0:
        return -1
    return 0

def rule_at_end_of_daily_range(state: dict):
    """
    Award 1 point if the price at the end of the day is within 20% of the daily trading range (either end)
    Otherwise 0.
    """
    assert state is not None
    day_low_high_df = state.get('day_low_high_df')
    date = state.get('date')
    threshold = state.get('daily_range_threshold')
    if not 'bad_days' in state:
        state['bad_days'] = 0
    try:
        day_low = day_low_high_df.at[date, 'day_low_price']
        day_high = day_low_high_df.at[date, 'day_high_price']
        last_price = day_low_high_df.at[date, 'last_price']
        if np.isnan(day_low) and np.isnan(day_high):
            return 0
        day_range = (day_high - day_low) * threshold # 20% at either end of daily range
        if last_price >= day_high - day_range:
            return 1
        elif last_price <= day_low + day_range:
            return -1
        # else FALLTHRU...
    except KeyError:
        #stock = state.get('stock')
        state['bad_days'] += 1
    return 0

def default_point_score_rules():
    """
    Return a list of rules to apply as a default list during analyse_point_scores()
    """
    return [rule_move_up,
            rule_market_avg,
            rule_sector_avg,
            rule_signif_move,
            rule_against_market,
            rule_at_end_of_daily_range
            ]

def detect_outliers(stocks: list, all_stocks_cip: pd.DataFrame, rules=None):
    """
    Returns a dataframe describing those outliers present in stocks based on the provided rules.
    All_stocks_cip is the "change in percent" for at least the stocks present in the specified list
    """
    if rules is None:
        rules = default_point_score_rules()
    str_rules = { str(r):r for r in rules }
    rows = []
    stocks_by_sector_df = stocks_by_sector() # NB: ETFs in watchlist will have no sector
    stocks_by_sector_df.index = stocks_by_sector_df['asx_code']
    for stock in stocks:
        #print("Processing stock: ", stock)
        try:
            sector = stocks_by_sector_df.at[stock, 'sector_name']
            sector_companies = list(stocks_by_sector_df.loc[stocks_by_sector_df['sector_name'] == sector].asx_code)
            # day_low_high() may raise KeyError when data is currently being fetched, so it appears here...
            day_low_high_df = day_low_high(stock, all_stocks_cip.columns)
        except KeyError:
            warning(None, "Unable to locate watchlist entry: {} - continuing without it".format(stock))
            continue
        state = {
            'day_low_high_df': day_low_high_df,  # never changes each day, so we init it here
            'all_stocks_change_in_percent_df': all_stocks_cip,
            'stock': stock,
            'daily_range_threshold': 0.20, # 20% at either end of the daily range gets a point
        }
        points_by_rule = defaultdict(int)
        for date in all_stocks_cip.columns:
            market_avg = all_stocks_cip[date].mean()
            sector_avg = all_stocks_cip[date].filter(items=sector_companies).mean()
            stock_move = all_stocks_cip.at[stock, date]
            state.update({ 'market_avg': market_avg, 'sector_avg': sector_avg,
                           'stock_move': stock_move, 'date': date })
            for rule_name, rule in str_rules.items():
                try:
                    points_by_rule[rule_name] += rule(state)
                except TypeError: # handle nan's in dataset safely
                    pass
        d = { 'stock': stock }
        d.update(points_by_rule)
        rows.append(d)
    df = pd.DataFrame.from_records(rows)
    df = df.set_index('stock')
    #print(df)
    clf = IForest()
    clf.fit(df)
    scores = clf.predict(df)
    results = [row[0] for row, value in zip(df.iterrows(), scores) if value > 0]
    #print(results)
    print("Found {} outlier stocks".format(len(results)))
    return results

def clean_weights(weights: OrderedDict, portfolio, first_prices, latest_prices):
    """Remove low weights as not significant contributors to portfolio performance"""
    sum_of_weights = sum(map(lambda t: t[1], weights.items()))
    assert sum_of_weights - 1.0 < 1e-6
    cw = OrderedDict()
    total_weight = 0.0
    # some algo's can have lots of little stock weights, so we dont stop until we explain >80%
    for stock, weight in sorted(weights.items(), key=lambda t: t[1], reverse=True):
        total_weight += weight * 100.0
        if not stock in portfolio:
            continue
        n = portfolio[stock]
        after = latest_prices[stock]
        before = first_prices[stock]
        cw[stock] = (stock, weight, n, after, before, n * (after - before))
        if total_weight >= 80.5 and len(cw.keys()) > 30:
            break
    #print(clean_weights)
    return cw

def portfolio_performance(optimizer):
    assert optimizer is not None
    pt = optimizer.portfolio_performance()
    assert len(pt) == 3
    performance_dict = {'expected return': pt[0] * 100.0, 'volatility': pt[1] * 100.0, 'sharpe ratio': pt[2]}
    return performance_dict

def hrp_strategy(returns):
    assert returns is not None
    ef = HRPOpt(returns=returns)
    weights = ef.optimize()
    return weights, portfolio_performance(ef), ef

def ef_sharpe_strategy(returns=None, cov_matrix=None):
    assert returns is not None
    ef = EfficientFrontier(returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) # eliminate minor weights
    weights = ef.max_sharpe()
    return weights, portfolio_performance(ef), ef

def ef_risk_strategy(returns=None, cov_matrix=None, target_volatility=5.0):
    assert returns is not None
    assert cov_matrix is not None
    ef = EfficientFrontier(returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.efficient_risk(target_volatility=target_volatility)
    return weights, portfolio_performance(ef), ef

def ef_minvol_strategy(returns=None, cov_matrix=None):
    ef = EfficientFrontier(returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.min_volatility()
    return weights, portfolio_performance(ef), ef

def remove_bad_stocks(df: pd.DataFrame, date_to_check: str, min_price, warning_cb):
    """
    Remove stocks which have no data at start/end of the timeframe despite imputation that has been performed. 
    This ensures that profit/loss can be calculated without missing data and that optimisation is not biased towards ridiculous outcomes.
    If min_price is not None, exclude all stocks with a start/end price less than min_price
    """
    validate_date(date_to_check)
    missing_prices = list(df.columns[df.loc[date_to_check].isna()])
    if len(missing_prices) > 0:
        df = df.drop(columns=missing_prices)
        if warning_cb:
            warning_cb("Ignoring stocks with no data at {}: {}".format(date_to_check, missing_prices))
    if min_price is not None:
        bad_prices = list(df.columns[df.loc[date_to_check] <= min_price])
        if warning_cb:
            warning_cb(f"Ignoring stocks with price < {min_price} at {date_to_check}: {bad_prices}")
        df = df.drop(columns=bad_prices)
    return df
 
def setup_optimisation_matrices(stocks, timeframe: Timeframe, exclude_price, warning_cb):
     # ref: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html#processing-historical-prices
    
    stock_prices = company_prices(stocks, timeframe, fields='last_price', missing_cb=None)
    stock_prices = stock_prices.fillna(method='bfill', limit=10, axis=0)
    latest_date = stock_prices.index[-1]
    earliest_date = stock_prices.index[0]
    #print(stock_prices)

    stock_prices = remove_bad_stocks(stock_prices, earliest_date, exclude_price, warning_cb)
    stock_prices = remove_bad_stocks(stock_prices, latest_date, exclude_price, warning_cb)

    latest_prices = stock_prices.loc[latest_date]
    first_prices = stock_prices.loc[earliest_date]
    all_returns = returns_from_prices(stock_prices, log_returns=False).fillna(value=0.0)

    # check that the matrices are consistent to each other
    assert stock_prices.shape[1] == latest_prices.shape[0]
    assert stock_prices.shape[1] == all_returns.shape[1]
    assert all_returns.shape[0] == stock_prices.shape[0] - 1
    assert len(stock_prices.columns) > 0 # must have at least 1 stock
    assert len(stock_prices) > 7 # and at least one trading week of data

    #print(stock_prices.shape)
    #print(latest_prices)
    #print(all_returns.shape)

    return all_returns, stock_prices, latest_prices, first_prices

def assign_strategy(filtered_stocks: pd.DataFrame, algo: str, n_stocks: int) -> tuple:
    filtered_stocks = filtered_stocks.sample(n=n_stocks, axis=1)
    returns = returns_from_prices(filtered_stocks, log_returns=False)
    mu = mean_historical_return(filtered_stocks)
    assert len(mu) == n_stocks
    s = CovarianceShrinkage(filtered_stocks).ledoit_wolf()

    if algo == "hrp":
        return (hrp_strategy,
                "Hierarchical Risk Parity",
                {'returns': returns}, mu, s)
    elif algo == "ef-sharpe":
        return (ef_sharpe_strategy, 
                "Efficient Frontier - max. sharpe",
                {'returns': mu, 'cov_matrix': s}, mu, s)
    elif algo == "ef-risk":
        return (ef_risk_strategy,
                "Efficient Frontier - efficient risk",
                {'returns': mu, 'cov_matrix': s, 'target_volatility': 5.0}, mu, s)
    elif algo == "ef-minvol":
        return (ef_minvol_strategy,
                "Efficient Frontier - minimum volatility",
                {'returns': mu, 'cov_matrix': s}, mu, s)
    else:
        assert False

def select_suitable_stocks(all_returns, stock_prices, max_stocks, n_unique_min, var_min):
     # drop columns were there is no activity (ie. same value) in the observation period
    cols_to_drop = list(all_returns.columns[all_returns.nunique() < n_unique_min])

    print("Dropping due to inactivity: {}".format(cols_to_drop))
    # drop columns with very low variance
    v = all_returns.var()
    low_var = v[v < var_min]
    print("Dropping due to low variance: {}".format(low_var.index))
    cols_to_drop.extend(low_var.index)
    #print("Stocks ignored due to inactivity: {}".format(cols_to_drop))
    filtered_stocks = stock_prices.drop(columns=cols_to_drop)
    colnames = filtered_stocks.columns
    n_stocks = max_stocks if len(colnames) > max_stocks else len(colnames)
    return filtered_stocks, n_stocks

def run_iteration(title, strategy, first_prices, latest_prices, total_portfolio_value, n_stocks, mu, s, filtered_stocks, **kwargs):
    weights, performance_dict, _ = strategy(**kwargs)
    allocator = DiscreteAllocation(weights,
                                   first_prices,
                                   total_portfolio_value=total_portfolio_value)
    fig, ax = plt.subplots()
    portfolio, leftover_funds = allocator.greedy_portfolio()
    #print(portfolio)
    cleaned_weights = clean_weights(weights, portfolio, first_prices, latest_prices)
    
    # disabled due to TypeError during deepcopy of OSQP results object
    #if algo.startswith("ef"): # HRP doesnt support frontier plotting atm
    #    plot_efficient_frontier(ef, ax=ax, show_assets=False)
    volatility = performance_dict.get('volatility')
    expected_return = performance_dict.get('expected return')
    ax.scatter(volatility, expected_return, marker="*", s=100, c="r", label="Portfolio")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns (%)")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(n_stocks), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ s @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig = plt.gcf()

    # NB: we dont bother caching these plots since we must calculate so many other values but we need to serve them via cache_plot() anyway
    efficient_frontier_plot = cache_plot(secrets.token_urlsafe(32), lambda: fig)

    # only plot covmatrix/corr for significant holdings to ensure readability
    m = CovarianceShrinkage(filtered_stocks[list(cleaned_weights.keys())[:30]]).ledoit_wolf()
    #print(m)
    cor_plot = plot_covariance(m, plot_correlation=True)
    
    correlation_plot = cache_plot(secrets.token_urlsafe(32), lambda: cor_plot.figure)
    assert isinstance(cleaned_weights, OrderedDict)
    return cleaned_weights, performance_dict, \
            efficient_frontier_plot, correlation_plot, \
            title, total_portfolio_value, leftover_funds, len(latest_prices)
  

def optimise_portfolio(stocks, timeframe: Timeframe, algo="ef-minvol", max_stocks=80, total_portfolio_value=100*1000, exclude_price=None, warning_cb=None):
    assert len(stocks) >= 1
    assert timeframe is not None
    assert total_portfolio_value > 0
    assert max_stocks >= 5

    all_returns, stock_prices, latest_prices, first_prices = setup_optimisation_matrices(stocks, timeframe, exclude_price, warning_cb)
    for t in ((10, 0.0001), (20, 0.0005), (30, 0.001), (40, 0.005), (50, 0.01)):
        filtered_stocks, n_stocks = select_suitable_stocks(all_returns, stock_prices, max_stocks, *t)
        strategy, title, kwargs, mu, s = assign_strategy(filtered_stocks, algo, n_stocks)
        try:
            return run_iteration(title, strategy, first_prices, latest_prices, total_portfolio_value, n_stocks, mu, s, filtered_stocks, **kwargs)
        except ValueError as ve:
            if warning_cb:
                warning_cb("Unable to optimise stocks with min_unique={} and var_min={}: n_stocks={} - {}".format(t[0], t[1], n_stocks, str(ve)))
            # try next iteration
        
    print("*** WARNING: unable to optimise portolio!")
    return (None, None, None, None, title, total_portfolio_value, 0.0, len(latest_prices))

