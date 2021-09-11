from app.data import make_portfolio_performance_dataframe
from django.db import models
from django.http import Http404
import investpy as ip
from datetime import datetime
from app.models import Timeframe
import pandas as pd
import re


def get_cryptocurrencies(as_choices=False):
    try:
        cryptos_df = ip.crypto.get_cryptos()
        if as_choices:
            return sorted(
                [
                    (l["symbol"] + ":" + l["name"], l["name"])
                    for i, l in cryptos_df.iterrows()
                ],
                key=lambda t: t[1],
            )
        else:
            return cryptos_df
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch crypto list: {str(e)}")


def investpy_date(ymd: str) -> str:
    d = datetime.strptime(ymd, "%Y-%m-%d")
    return d.strftime("%d/%m/%Y")


def grok_crypto_symbol(cs: str) -> tuple:
    assert len(cs) >= 3 and ":" in cs
    m = re.match("^(\w+):(.*)$", cs)
    assert m is not None
    return (m.group(1), m.group(2))


def get_crypto_prices(
    crypto_symbol: str, timeframe: Timeframe, interval="Daily"
) -> pd.DataFrame:
    try:
        symbol, name = grok_crypto_symbol(crypto_symbol)
        df = ip.crypto.get_crypto_historical_data(
            name,
            investpy_date(timeframe.earliest_date),
            investpy_date(timeframe.most_recent_date),
            interval=interval,
        )
        return df
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch prices for {crypto_symbol}: {str(e)}")


def get_commodities(as_choices=False):
    try:
        commodities = ip.commodities.get_commodities()
        if as_choices:
            return [
                (l["name"], l["full_name"])
                for i, l in sorted(commodities.iterrows(), key=lambda t: t[1]["name"])
            ]
        else:
            return commodities
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch commodity list: {str(e)}")


def get_commodity_prices(
    commodity_str: str, timeframe: Timeframe, interval="Daily"
) -> pd.DataFrame:
    try:
        prices_df = ip.commodities.get_commodity_historical_data(
            commodity_str,
            investpy_date(timeframe.earliest_date),
            investpy_date(timeframe.most_recent_date),
            interval=interval,
        )
        return prices_df
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch commodity prices: {str(e)}")


def get_bond_countries(as_choices=False) -> list:
    try:
        countries = ip.bonds.get_bond_countries()
        if as_choices:
            return [(i, i) for i in countries]
        else:
            return countries
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch list of country bonds: {str(e)}")


def get_bonds_for_country(country_name: str) -> pd.DataFrame:
    try:
        available_bonds = ip.bonds.get_bonds(country=country_name)
        return available_bonds
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch bonds for {country_name}: {str(e)}")


def get_bond_prices(bond_name: str, timeframe: Timeframe) -> pd.DataFrame:
    try:
        df = ip.bonds.get_bond_historical_data(
            bond_name,
            investpy_date(timeframe.earliest_date),
            investpy_date(timeframe.most_recent_date),
        )
        return df
    except (ValueError, ConnectionError, IOError) as e:
        raise Http404(f"Unable to fetch bond prices for {bond_name}: {str(e)}")