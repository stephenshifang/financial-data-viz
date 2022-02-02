"""Financial data visualization utils"""

import functools
import re
import math
from typing import List, Optional
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset, BDay, BMonthEnd, BYearEnd, BQuarterEnd
import pandas_datareader.data as web
import plotly.express as px

DT_FMT = '%m/%d/%Y'

# Vanguard sector ETFs
TICKER_X_SECTOR = {
    'VOX': 'Communication Services',
    'VCR': 'Consumer Discretionary',
    'VDC': 'Consumer Staples',
    'VDE': 'Energy',
    'VFH': 'Financials',
    'VHT': 'Health Care',
    'VIS': 'Industrials',
    'VGT': 'Information Technology',
    'VAW': 'Materials',
    'VNQ': 'Real Estate',
    'VPU': 'Utilities'
}

# Vanguard factor ETFs
TICKER_X_FACTOR = {
    'VFVA': 'Value',
    'VFMO': 'Momentum',
    'VFQY': 'Quality',
    'VFLQ': 'Liquidity',
}

# Combine
TICKER_X_NAME = {**TICKER_X_SECTOR, **TICKER_X_FACTOR}


@functools.lru_cache()
def _query_data(ticker, source, cached_sample='2y'):
    """Pull data for a given ticker and cache it"""
    start_dt, end_dt = _find_sample(None, None, cached_sample)

    # Query data
    try:
        df = web.DataReader(ticker, source, start_dt, end_dt)
        return df
    except Exception as e:
        # Return empty dataframe if data pull error
        print(f"Data query failed for ticker {ticker} source {source} from "
              f"{start_dt.strftime(DT_FMT)} to {end_dt.strftime(DT_FMT)}")
        print(e)
        return pd.DataFrame()


def _find_sample(start_dt=None, end_dt=None, sample_period='2y'):
    """Helper function to find sample start date and end date"""
    # Find start/end dates
    if end_dt is None:
        # Default end date is the most recent weekday up to today
        end_dt = pd.Timestamp.today().floor('D')
    else:
        end_dt = pd.to_datetime(end_dt, errors='raise')
    end_dt = BDay().rollback(end_dt)

    if start_dt is not None:
        start_dt = BDay().rollback(pd.to_datetime(start_dt, errors='raise'))
        return start_dt, end_dt

    assert sample_period, f"invalid sample_period; received {sample_period}"

    # Infer start date
    sample_period = sample_period.lower()
    if sample_period == 'ytd':
        start_dt = end_dt - BYearEnd(1)

    elif sample_period == 'qtd':
        start_dt = end_dt - BQuarterEnd(1)

    elif sample_period == 'mtd':
        start_dt = end_dt - BMonthEnd(1)

    else:
        # accept sample_period as #y/m/w/d
        assert re.search(r"^\d+[ymwd]$", sample_period, re.I), \
            f"invalid sample_period {sample_period}"

        if sample_period.endswith('y'):
            num_years = int(sample_period[:-1])
            start_dt = end_dt - DateOffset(years=num_years)

        elif sample_period.endswith('m'):
            num_months = int(sample_period[:-1])
            start_dt = end_dt - DateOffset(months=num_months)

        elif sample_period.endswith('w'):
            num_weeks = int(sample_period[:-1])
            start_dt = end_dt - DateOffset(weeks=num_weeks)

        elif sample_period.endswith('d'):
            num_days = int(sample_period[:-1])
            start_dt = end_dt - BDay(num_days)

    start_dt = BDay().rollback(start_dt)
    return start_dt, end_dt


def query_data(tickers: List[str],
               source: str,
               start_dt: Optional[pd.Timestamp] = None,
               end_dt: Optional[pd.Timestamp] = None,
               sample_period: str = 'YTD',
               freq: str = 'D',
               cached_sample: str = '2Y') -> pd.DataFrame:
    """Queries financial time-series data from online sources, e.g. Yahoo Finance.

    Args:
        tickers: List of tickers.
        source: Data source.
        start_dt: Start date of time-series.
        end_dt: End date of time-series.
        sample_period: Sample period, e.g. 'YTD' or '1Y'.
        freq: Frequency of data, e.g. 'D' or 'W'.
        cached_sample: Sample period for data cache, e.g. '2Y'.

    Returns:
        A DataFrame.
    """
    tickers = np.atleast_1d(tickers).tolist()
    tickers = [v for i, v in enumerate(tickers) if v not in tickers[:i]]

    # Find start/end dates
    start_dt, end_dt = _find_sample(start_dt, end_dt, sample_period)

    # Find cache dates
    cache_start_dt, _ = _find_sample(None, None, cached_sample)

    # Renew cache if necessary
    if cache_start_dt > start_dt:
        num_years = math.ceil((pd.Timestamp.today() - start_dt).days / 365.25)
        cached_sample = f"{num_years}y"

    # Get data
    dfs = []
    for ticker in tickers:
        df_one = _query_data(ticker, source, cached_sample)
        df_one = df_one.loc[slice(start_dt, end_dt)]
        if not df_one.empty:
            # Reset to multi-index for yahoo
            if source.lower() == 'yahoo':
                df_one.columns = pd.MultiIndex.from_product([df_one.columns, (ticker,)])
            dfs.append(df_one)

    df = pd.concat(dfs, axis=1, join='outer')
    df = df.dropna(how='all').sort_index()

    freq = freq.upper()
    if freq == 'D':
        # Note this will preserve native frequency of the raw data
        df = df.ffill()
    elif freq == 'W':
        df = df.resample('W-FRI').ffill()
    elif freq == 'M':
        df = df.resample('BM').ffill()
    elif freq == 'Y':
        df = df.resample('BY').ffill()
    else:
        raise NotImplementedError(f"freq {freq} not implemented")

    return df


def _rename_tickers(df, ticker_name_map):
    """Rename ticker to names"""
    if df.columns.intersection(ticker_name_map.keys()).size > 0:
        df = df.rename(ticker_name_map, axis=1)
    return df


def compute_cumulative_returns(tickers: List[str],
                               start_dt: Optional[pd.Timestamp] = None,
                               end_dt: Optional[pd.Timestamp] = None,
                               sample_period: str = 'YTD',
                               freq: str = 'D') -> pd.DataFrame:
    """ Computes cumulative returns for a list of tickers over a given sample

    Args:
        tickers: List of tickers.
        start_dt: Start date of cumulative returns.
        end_dt: End date of cumulative returns.
        sample_period: Sample period.
        freq: Frequency of cumulative returns.

    Returns:
        A DataFrame containing the cumulative returns
    """
    # Query data
    source = 'yahoo'
    df_raw = query_data(tickers, source, start_dt, end_dt, sample_period, freq)
    df_close = df_raw['Adj Close']

    # Compute cumulative return
    begin_price = df_close.apply(lambda x: x.loc[x.first_valid_index()], axis=0)
    df_cumret = (df_close.div(begin_price, axis=1) - 1) * 100
    df_cumret = df_cumret.fillna(0)

    return df_cumret


def plot_cumulative_returns(tickers: List[str],
                            start_dt: Optional[pd.Timestamp] = None,
                            end_dt: Optional[pd.Timestamp] = None,
                            sample_period: str = 'YTD',
                            freq: str = 'D',
                            **plot_kws):
    """Compute & plot cumulative total returns as of end date

    Args:
        tickers: List of tickers.
        start_dt: Start date of cumulative returns.
        end_dt: End date of cumulative returns.
        sample_period: Sample period.
        freq: Frequency of cumulative returns.
        **plot_kws:

    Returns:

    """
    df_cumret = compute_cumulative_returns(tickers, start_dt, end_dt, sample_period, freq)

    # Sort columns by end-of-period return
    ret = df_cumret.iloc[-1].sort_values(ascending=False)
    df_cumret = df_cumret[ret.index]

    # Rename columns if necessary
    df_cumret = _rename_tickers(df_cumret, TICKER_X_NAME)
    df_cumret.columns.name = 'Symbol'

    nobs = df_cumret.shape[0]
    if not sample_period:
        start_dt = df_cumret.index[1]
    end_dt = df_cumret.index[-1]

    if nobs > 2:
        # Time series
        if sample_period:
            title = f"{sample_period.upper()} Cumulative Return"
        else:
            title = f"Cumulative Return ({start_dt:%m/%d/%Y} - {end_dt:%m/%d/%Y})"

        # Reshape to long format
        df_plot = df_cumret.stack().reset_index()
        df_plot.columns = ['Date', 'Symbol', 'Return']
        fig = px.line(df_plot, x='Date', y='Return', color='Symbol', markers=nobs < 25, title=title, **plot_kws)
        fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2f}')

        if nobs < 25:
            # Hide weekends & major holidays
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    else:
        # Bar chart
        df_plot = df_cumret.iloc[-1].reset_index()
        df_plot.columns = ['Symbol', 'Return']
        title = f"Return as of {end_dt.strftime(DT_FMT)}"
        fig = px.bar(df_plot, x='Return', y='Symbol', orientation='h', title=title, **plot_kws)
        fig.update_traces(hovertemplate='Return: %{x:.2f}')

    fig.show()
    return df_cumret


def compute_returns(tickers: List[str],
                    start_dt: Optional[pd.Timestamp] = None,
                    end_dt: Optional[pd.Timestamp] = None,
                    sample_period: str = 'YTD',
                    freq: str = 'D') -> pd.DataFrame:
    """Compute returns for a list of tickers

    Args:
        tickers: List of tickers.
        start_dt: Start date of returns.
        end_dt: End date of returns.
        sample_period: Sample period.
        freq: Frequency of returns.

    Returns:
        A DataFrame containing returns
    """
    # Compute returns
    source = 'yahoo'
    df_raw = query_data(tickers, source, start_dt, end_dt, sample_period, freq)
    df_ret = (df_raw['Adj Close'] / df_raw['Adj Close'].shift(1) - 1) * 100

    # Drop if all missing
    df_ret = df_ret.dropna(how='all')

    return df_ret


def plot_ts(series: str,
            source: str = 'yahoo',
            start_dt: Optional[pd.Timestamp] = None,
            end_dt: Optional[pd.Timestamp] = None,
            sample_period: str = 'YTD',
            freq: str = 'D', **plot_kws):
    """Plot time-series data

    Args:
        series: Name of time-series.
        source: Data Source.
        start_dt: Start date of time-series.
        end_dt: End date of time-series.
        sample_period: Sample period of time-series.
        freq: Frequency of time-series.
        **plot_kws: Pass-through params.

    Returns:
        A DataFrame containing the time-series data.
    """
    # Query data
    df_ts = query_data(series, source, start_dt, end_dt, sample_period, freq)

    # Special case data from yahoo
    if source.lower() == 'yahoo':
        df_ts = df_ts['Adj Close']

    # Rename columns if necessary
    df_ts = _rename_tickers(df_ts, TICKER_X_NAME)
    df_ts.columns.name = 'Series'

    # Sort columns by end-of-period values
    vals = df_ts.iloc[-1].sort_values(ascending=False)
    df_ts = df_ts[vals.index]

    # Plot
    nobs = df_ts.shape[0]
    if (sample_period and sample_period.lower() != '1d') or (nobs > 2):
        # Reshape to long format
        df_plot = df_ts.stack().reset_index()
        df_plot.columns = ['Date', 'Series', 'Value']
        fig = px.line(df_plot, x='Date', y='Value', color='Series', markers=nobs < 25, **plot_kws)
        fig.update_traces(hovertemplate='Date: %{x} <br>Value: %{y:.2f}')
        # Hide weekends
        if nobs < 25:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    else:
        df_plot = df_ts.iloc[-1].reset_index()
        df_plot.columns = ['Series', 'Value']
        fig = px.bar(df_plot, x='Value', y='Series', orientation='h', **plot_kws)
        fig.update_traces(hovertemplate='Value: %{x:.2f}')

    fig.show()
    return df_ts


if __name__ == 'main':
    # query time-series data
    series = '^VIX'
    sample_period = 'ytd'
    start_dt = end_dt = None
    freq = 'd'
    df_ts = plot_ts(series, source='yahoo', sample_period=sample_period, freq='D')

