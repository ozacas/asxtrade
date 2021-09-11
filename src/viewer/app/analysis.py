"""
Responsible for performing computations to support a visualisation/analysis result
"""
from collections import defaultdict, OrderedDict
import secrets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lazydict import LazyDictionary
from pyod.models.iforest import IForest
from pypfopt.expected_returns import mean_historical_return, returns_from_prices
from pypfopt.black_litterman import (
    BlackLittermanModel,
    market_implied_risk_aversion,
    market_implied_prior_returns,
)
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.plotting import plot_covariance
from pypfopt.hierarchical_portfolio import HRPOpt
from app.models import (
    company_prices,
    day_low_high,
    stocks_by_sector,
    Timeframe,
    validate_date,
    valid_quotes_only,
)
from app.data import cache_plot
from app.messages import warning


def as_css_class(thirty_day_slope: float, three_hundred_day_slope: float) -> str:
    if thirty_day_slope > 0.0 and three_hundred_day_slope < 0.0:
        return "recent-upward-trend"
    elif thirty_day_slope < 0.0 and three_hundred_day_slope > 0.0:
        return "recent-downward-trend"
    else:
        return "none"


def calculate_trend_error(
    x_data: np.ndarray, y_data: np.ndarray, fitted_parameters
) -> tuple:
    modelPredictions = np.polyval(fitted_parameters, x_data)
    absolute_errors = modelPredictions - y_data

    squared_errors = np.square(absolute_errors)
    mean_squared_errors = np.mean(squared_errors)  # mean squared errors
    root_mean_squared_error = np.sqrt(
        mean_squared_errors
    )  # Root Mean Squared Error, RMSE
    y_var = np.var(y_data)

    if y_var > 0.0:
        r_squared = 1.0 - (np.var(absolute_errors) / y_var)
    else:
        r_squared = float(np.nan)
    series_range = y_data.max() - y_data.min()
    nrmse = root_mean_squared_error / series_range if abs(series_range) > 0.0 else 0.0

    return nrmse, r_squared


def calculate_trends(
    df: pd.DataFrame,
    polynomial_degree: int = 1,
    nrmse_cutoff: float = 0.2,  # exclude poor fits
    r_squared_cutoff: float = 0.05,
) -> OrderedDict:
    """
    Use numpy polyfit to calculate the trends associated with all stocks (rows) in the specified dataframe. Each row is considered to be a
    uniform timeseries eg. daily prices. An ordereddict of (stock, tuple) is returned describing the trend calculation.
    """
    assert polynomial_degree >= 1
    assert nrmse_cutoff > 0.0
    assert r_squared_cutoff >= 0.0
    assert len(df) > 0
    trends = {}  # stock -> (slope, nrmse) pairs
    for index_name, series in df.iterrows():
        assert isinstance(
            index_name, str
        )  # maybe a stock, but not guaranteed as calculate_trends() is also used by financial metrics
        assert isinstance(series, pd.Series)
        series = series.dropna()  # NB: np.polyfit doesnt work with NA so...
        n = len(series)
        if n == 0 or n < max(
            4, 3 + polynomial_degree
        ):  # too few data points for a trend? if so, ignore the series
            continue

        # timeseries have more than 30 days? if so, compute a short-term trend for the user
        # print(series)
        x_data = range(n)
        fitted_parameters = np.polyfit(x_data, series, polynomial_degree, full=True)
        nrmse, r_squared = calculate_trend_error(x_data, series, fitted_parameters[0])

        if n > 30:
            series30 = series[-30:]
            x_30 = range(len(series30))
            fp_30 = np.polyfit(x_30, series30, polynomial_degree, full=True)
            nrmse_30, r_squared_30 = calculate_trend_error(x_30, series30, fp_30[0])
        else:
            nrmse_30 = 0.0
            r_squared_30 = 0.0
            fp_30 = ((np.nan, np.nan), 1.0)

        # reasonable trend identified?
        # print(f"{index_name} {r_squared} {nrmse} {r_squared_30} {nrmse_30}")
        if (abs(r_squared) > r_squared_cutoff and nrmse < nrmse_cutoff) or (
            abs(r_squared_30) > r_squared_cutoff and nrmse_30 < nrmse_cutoff
        ):
            trends[index_name] = (
                r_squared,
                nrmse,
                r_squared_30,
                nrmse_30,
                as_css_class(fp_30[0][0], fitted_parameters[0][0]),
            )
    # sort by ascending overall slope (regardless of NRMSE)
    return OrderedDict(sorted(trends.items(), key=lambda t: t[0]))


def rank_cumulative_change(df: pd.DataFrame, timeframe: Timeframe):
    cum_sum = defaultdict(float)
    # print(df)
    for date in filter(lambda k: k in df.columns, timeframe.all_dates()):
        for code, price_change in df[date].fillna(0.0).iteritems():
            cum_sum[code] += price_change
        rank = pd.Series(cum_sum).rank(method="first", ascending=False)
        df[date] = rank

    all_available_dates = df.columns
    avgs = df.mean(axis=1)  # NB: do this BEFORE adding columns...
    assert len(avgs) == len(df)
    df["x"] = all_available_dates[-1]
    df["y"] = df[all_available_dates[-1]]

    bins = ["top", "bin2", "bin3", "bin4", "bin5", "bottom"]
    average_rank_binned = pd.cut(avgs, len(bins), bins)
    assert len(average_rank_binned) == len(df)
    df["bin"] = average_rank_binned
    df["asx_code"] = df.index
    stock_sector_df = (
        stocks_by_sector()
    )  # make one DB call (cached) rather than lots of round-trips
    # print(stock_sector_df)
    stock_sector_df = stock_sector_df.set_index("asx_code")
    # print(df.index)
    df = df.merge(
        stock_sector_df, left_index=True, right_on="asx_code"
    )  # NB: this merge will lose rows: those that dont have a sector eg. ETF's
    df = pd.melt(
        df,
        id_vars=["asx_code", "bin", "sector_name", "x", "y"],
        var_name="date",
        value_name="rank",
        value_vars=all_available_dates,
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["x"] = pd.to_datetime(df["x"], format="%Y-%m-%d")
    return df


def rule_move_up(state: dict):
    """
    Return 1 if the state indicates the stock moved up, 0 otherwise
    """
    assert state is not None

    move = state.get("stock_move")
    return 1 if move > 0.0 else 0


def rule_market_avg(state: dict):
    """
    If the magnitude of the stock move is greater than the magnitude of the
    market move, then award two points in the direction of the move. Otherwise 0
    """
    assert state is not None

    move = state.get("stock_move")
    market_avg = state.get("market_avg")
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

    move = state.get("stock_move")
    sector_avg = state.get("sector_avg")

    if abs(move) >= abs(sector_avg):
        return np.sign(move) * 3
    return 0


def rule_signif_move(state: dict):
    """
    Only ~10% of stocks will move by more than 2% percent on a given day, so give a point for that...
    """
    assert state is not None

    move = state.get("stock_move")  # percentage move
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
    stock_move = state.get("stock_move")
    market_avg = state.get("market_avg")
    sector_avg = state.get("sector_avg")
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
    day_low_high_df = state.get("day_low_high_df")
    date = state.get("date")
    threshold = state.get("daily_range_threshold")
    if not "bad_days" in state:
        state["bad_days"] = 0
    try:
        day_low = day_low_high_df.at[date, "day_low_price"]
        day_high = day_low_high_df.at[date, "day_high_price"]
        last_price = day_low_high_df.at[date, "last_price"]
        if np.isnan(day_low) and np.isnan(day_high):
            return 0
        day_range = (day_high - day_low) * threshold  # 20% at either end of daily range
        if last_price >= day_high - day_range:
            return 1
        elif last_price <= day_low + day_range:
            return -1
        # else FALLTHRU...
    except KeyError:
        # stock = state.get('stock')
        state["bad_days"] += 1
    return 0


def default_point_score_rules():
    """
    Return a list of rules to apply as a default list during analyse_point_scores()
    """
    return [
        rule_move_up,
        rule_market_avg,
        rule_sector_avg,
        rule_signif_move,
        rule_against_market,
        rule_at_end_of_daily_range,
    ]


def detect_outliers(stocks: list, all_stocks_cip: pd.DataFrame, rules=None):
    """
    Returns a dataframe describing those outliers present in stocks based on the provided rules.
    All_stocks_cip is the "change in percent" for at least the stocks present in the specified list
    """
    if rules is None:
        rules = default_point_score_rules()
    str_rules = {str(r): r for r in rules}
    rows = []
    stocks_by_sector_df = (
        stocks_by_sector()
    )  # NB: ETFs in watchlist will have no sector
    stocks_by_sector_df.index = stocks_by_sector_df["asx_code"]
    for stock in stocks:
        # print("Processing stock: ", stock)
        try:
            sector = stocks_by_sector_df.at[stock, "sector_name"]
            sector_companies = list(
                stocks_by_sector_df.loc[
                    stocks_by_sector_df["sector_name"] == sector
                ].asx_code
            )
            # day_low_high() may raise KeyError when data is currently being fetched, so it appears here...
            day_low_high_df = day_low_high(stock, all_stocks_cip.columns)
        except KeyError:
            warning(
                None,
                "Unable to locate watchlist entry: {} - continuing without it".format(
                    stock
                ),
            )
            continue
        state = {
            "day_low_high_df": day_low_high_df,  # never changes each day, so we init it here
            "all_stocks_change_in_percent_df": all_stocks_cip,
            "stock": stock,
            "daily_range_threshold": 0.20,  # 20% at either end of the daily range gets a point
        }
        points_by_rule = defaultdict(int)
        for date in all_stocks_cip.columns:
            market_avg = all_stocks_cip[date].mean()
            sector_avg = all_stocks_cip[date].filter(items=sector_companies).mean()
            stock_move = all_stocks_cip.at[stock, date]
            state.update(
                {
                    "market_avg": market_avg,
                    "sector_avg": sector_avg,
                    "stock_move": stock_move,
                    "date": date,
                }
            )
            for rule_name, rule in str_rules.items():
                try:
                    points_by_rule[rule_name] += rule(state)
                except TypeError:  # handle nan's in dataset safely
                    pass
        d = {"stock": stock}
        d.update(points_by_rule)
        rows.append(d)
    df = pd.DataFrame.from_records(rows)
    df = df.set_index("stock")
    # print(df)
    clf = IForest()
    clf.fit(df)
    scores = clf.predict(df)
    results = [row[0] for row, value in zip(df.iterrows(), scores) if value > 0]
    # print(results)
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
    # print(clean_weights)
    return cw


def portfolio_performance(ld: LazyDictionary) -> dict:
    weights = ld[
        "raw_weights"
    ]  # force optimization to be done, so the weights exists and thus the performance...
    pt = ld["optimizer"].portfolio_performance()
    assert len(pt) == 3
    performance_dict = {
        "expected return": pt[0] * 100.0,
        "volatility": pt[1] * 100.0,
        "sharpe ratio": pt[2],
    }
    return performance_dict


def hrp_strategy(ld: LazyDictionary, **kwargs) -> None:
    ef = HRPOpt(returns=kwargs.get("returns"))
    ld["optimizer"] = ef
    ld["raw_weights"] = lambda ld: ld["optimizer"].optimize()


def ef_sharpe_strategy(ld: LazyDictionary, **kwargs) -> None:
    ef = EfficientFrontier(
        expected_returns=kwargs.get("returns"),
        cov_matrix=kwargs.get("cov_matrix", None),
    )
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # eliminate minor weights
    ld["optimizer"] = ef
    ld["raw_weights"] = lambda ld: ld["optimizer"].max_sharpe()


def ef_risk_strategy(
    ld: LazyDictionary, returns=None, cov_matrix=None, target_volatility=5.0
) -> None:
    assert returns is not None
    assert cov_matrix is not None
    ef = EfficientFrontier(returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    ld["optimizer"] = ef
    ld["raw_weights"] = lambda ld: ld["optimizer"].efficient_risk(
        target_volatility=target_volatility
    )


def ef_minvol_strategy(ld: LazyDictionary, returns=None, cov_matrix=None):
    ef = EfficientFrontier(returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.min_volatility()
    ld["optimizer"] = ef
    ld["raw_weights"] = lambda ld: ld["optimizer"].min_volatility()


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
            warning_cb(
                "Ignoring stocks with no data at {}: {}".format(
                    date_to_check, missing_prices
                )
            )
    if min_price is not None:
        bad_prices = list(df.columns[df.loc[date_to_check] <= min_price])
        if warning_cb:
            warning_cb(
                f"Ignoring stocks with price < {min_price} at {date_to_check}: {bad_prices}"
            )
        df = df.drop(columns=bad_prices)
    return df


def setup_optimisation_matrices(
    stocks, timeframe: Timeframe, exclude_price, warning_cb
):
    # ref: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html#processing-historical-prices

    stock_prices = company_prices(
        stocks, timeframe, fields="last_price", missing_cb=None
    )
    stock_prices = stock_prices.fillna(method="bfill", limit=10, axis=0)
    latest_date = stock_prices.index[-1]
    earliest_date = stock_prices.index[0]
    # print(stock_prices)

    stock_prices = remove_bad_stocks(
        stock_prices, earliest_date, exclude_price, warning_cb
    )
    stock_prices = remove_bad_stocks(
        stock_prices, latest_date, exclude_price, warning_cb
    )

    latest_prices = stock_prices.loc[latest_date]
    first_prices = stock_prices.loc[earliest_date]
    all_returns = returns_from_prices(stock_prices, log_returns=False).fillna(value=0.0)

    # check that the matrices are consistent to each other
    assert stock_prices.shape[1] == latest_prices.shape[0]
    assert stock_prices.shape[1] == all_returns.shape[1]
    assert all_returns.shape[0] == stock_prices.shape[0] - 1
    assert len(stock_prices.columns) > 0  # must have at least 1 stock
    assert len(stock_prices) > 7  # and at least one trading week of data

    # print(stock_prices.shape)
    # print(latest_prices)
    # print(all_returns.shape)

    return all_returns, stock_prices, latest_prices, first_prices


def assign_strategy(ld: LazyDictionary, algo: str) -> tuple:
    assert ld is not None
    # use of black-litterman is based on https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb
    # print(market_prices)
    ld["s"] = CovarianceShrinkage(ld["filtered_stocks"]).ledoit_wolf()
    ld["delta"] = market_implied_risk_aversion(ld["market_prices"])
    # use BlackLitterman model to compute returns - hopefully better estimate of returns than extrapolation of historical prices
    # market_prior = market_implied_prior_returns(ld["market_caps"], delta, ld["s"])
    ld["bl"] = lambda ld: BlackLittermanModel(
        ld["s"],
        pi="market",
        market_caps=ld["market_caps"],
        risk_aversion=ld["delta"],
        absolute_views={},
    )
    ld["posterior_total_returns"] = lambda ld: ld["bl"].bl_returns()
    ld["posterior_s"] = lambda ld: ld["bl"].bl_cov()
    ld["mu"] = lambda ld: mean_historical_return(ld["filtered_stocks"])
    ld["returns_from_prices"] = lambda ld: returns_from_prices(ld["filtered_stocks"])

    use_bl = ld["returns_by"] != "by_prices"
    kwargs = (
        {"returns": ld["mu"]} if use_bl else {"returns": ld["posterior_total_returns"]}
    )
    if algo != "hrp":
        kwargs["cov_matrix"] = ld["s"] if not use_bl else ld["posterior_s"]
    else:
        # algo is HRP
        kwargs = {"returns": ld["returns_from_prices"]}

    if algo == "hrp":
        ld["title"] = "Hierarchical Risk Parity"
        return (hrp_strategy, kwargs)
    elif algo == "ef-sharpe":
        ld["title"] = "Efficient Frontier - max. sharpe"
        return (ef_sharpe_strategy, kwargs)
    elif algo == "ef-risk":
        ld["title"] = "Efficient Frontier - efficient risk"
        kwargs["target_volatility"] = 5.0
        return (ef_risk_strategy, kwargs)
    elif algo == "ef-minvol":
        ld["title"] = "Efficient Frontier - minimum volatility"
        return (ef_minvol_strategy, kwargs)
    else:
        assert False


def select_suitable_stocks(
    all_returns, stock_prices, max_stocks, n_unique_min, var_min
):
    # drop columns were there is no activity (ie. same value) in the observation period
    cols_to_drop = list(all_returns.columns[all_returns.nunique() < n_unique_min])
    print("Dropping due to inactivity: {}".format(cols_to_drop))
    # drop columns with very low variance
    v = all_returns.var()
    low_var = v[v < var_min]
    print("Dropping due to low variance: {}".format(low_var.index))
    # cols_to_drop.extend(low_var.index)
    # print("Stocks ignored due to inactivity: {}".format(cols_to_drop))
    filtered_stocks = stock_prices.drop(columns=cols_to_drop)
    colnames = filtered_stocks.columns
    n_stocks = max_stocks if len(colnames) > max_stocks else len(colnames)
    return filtered_stocks, n_stocks


def run_iteration(
    ld: LazyDictionary,
    strategy,
    first_prices,
    latest_prices,
    filtered_stocks,
    **kwargs,
):
    assert ld is not None
    strategy(ld, **kwargs)

    ld["optimizer_performance"] = lambda ld: portfolio_performance(ld)
    ld["allocator"] = lambda ld: DiscreteAllocation(
        ld["raw_weights"],
        first_prices,
        total_portfolio_value=ld["total_portfolio_value"],
    )
    ld["portfolio"] = lambda ld: ld[
        "allocator"
    ].greedy_portfolio()  # greedy_portfolio returns (dict, float) tuple
    ld["cleaned_weights"] = lambda ld: clean_weights(
        ld["raw_weights"], ld["portfolio"][0], first_prices, latest_prices
    )
    ld["latest_prices"] = latest_prices
    ld["len_latest_prices"] = lambda ld: len(ld["latest_prices"])

    # only plot covmatrix/corr for significant holdings to ensure readability
    ld["m"] = lambda ld: CovarianceShrinkage(
        filtered_stocks[list(ld["cleaned_weights"].keys())[:30]]
    ).ledoit_wolf()


def plot_random_portfolios(ld: LazyDictionary):
    assert ld is not None
    fig, ax = plt.subplots()
    # disabled due to TypeError during deepcopy of OSQP results object
    # if algo.startswith("ef"): # HRP doesnt support frontier plotting atm
    #    plot_efficient_frontier(ef, ax=ax, show_assets=False)
    performance_dict = ld["optimizer_performance"]
    volatility = performance_dict.get("volatility")
    expected_return = performance_dict.get("expected return")
    ax.scatter(volatility, expected_return, marker="*", s=100, c="r", label="Portfolio")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns (%)")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ld["n_stocks"]), n_samples)
    rets = w.dot(ld["posterior_total_returns"])
    stds = np.sqrt(np.diag(w @ ld["s"] @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title(ld["title"])
    ax.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def optimise_portfolio(
    stocks,
    timeframe: Timeframe,
    algo="ef-minvol",
    max_stocks=80,
    total_portfolio_value=100 * 1000,
    exclude_price=None,
    warning_cb=None,
    **kwargs,
):
    assert len(stocks) >= 1
    assert timeframe is not None
    assert total_portfolio_value > 0
    assert max_stocks >= 5

    (
        all_returns,
        stock_prices,
        latest_prices,
        first_prices,
    ) = setup_optimisation_matrices(stocks, timeframe, exclude_price, warning_cb)

    market_prices = company_prices(
        ("A200",), Timeframe(past_n_days=180), missing_cb=None
    )
    market_prices.index = pd.to_datetime(market_prices.index, format="%Y-%m-%d")
    market_prices = pd.Series(market_prices["A200"])
    quotes, ymd = valid_quotes_only("latest", ensure_date_has_data=True)

    for t in ((10, 0.0001), (20, 0.0005), (30, 0.001), (40, 0.005), (50, 0.01)):
        filtered_stocks, n_stocks = select_suitable_stocks(
            all_returns, stock_prices, max_stocks, *t
        )
        # since the sample of stocks might be different, we must recompute each iteration...
        filtered_stocks = filtered_stocks.sample(n=n_stocks, axis=1)
        # print(len(filtered_stocks.columns))
        market_caps = {
            q.asx_code: q.market_cap
            for q in quotes
            if q.asx_code in filtered_stocks.columns
        }

        ld = (
            LazyDictionary()
        )  # must start a new dict since each key is immutable after use
        ld["n_stocks"] = n_stocks
        ld["filtered_stocks"] = filtered_stocks
        ld["market_prices"] = market_prices
        ld["market_caps"] = market_caps
        ld["total_portfolio_value"] = total_portfolio_value
        ld["returns_by"] = kwargs.get("returns_by", "by_prices")

        strategy, kwargs = assign_strategy(ld, algo)
        try:
            run_iteration(
                ld,
                strategy,
                first_prices,
                latest_prices,
                filtered_stocks,
                **kwargs,
            )

            # NB: we dont bother caching these plots since we must calculate so many other values but we need to serve them via cache_plot() anyway
            ld["efficient_frontier_plot"] = cache_plot(
                secrets.token_urlsafe(32), plot_random_portfolios, datasets=ld
            )
            ld["correlation_plot"] = lambda ld: cache_plot(
                secrets.token_urlsafe(32),
                lambda ld: plot_covariance(ld["m"], plot_correlation=True).figure,
                datasets=ld,
            )
            return ld
        except ValueError as ve:
            if warning_cb:
                warning_cb(
                    "Unable to optimise stocks with min_unique={} and var_min={}: n_stocks={} - {}".format(
                        t[0], t[1], n_stocks, str(ve)
                    )
                )
            # try next iteration
            raise ve

    print("*** WARNING: unable to optimise portolio!")
    return LazyDictionary()