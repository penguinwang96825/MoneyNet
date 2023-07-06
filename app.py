import os
import datetime
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deta import Deta
from hashlib import sha256
from tqdm.auto import tqdm
from stqdm import stqdm
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from streamlit_extras.stateful_button import button as extra_button

from src.utils import hide_header_and_footer, read_css_file, make_clickable
from src.tickers import (
    get_nasdaq_tickers, 
    get_sp500_tickers, 
    get_top_crypto_tickers, 
    get_taiwan_sem_tickers, 
    get_taiwan_otc_tickers
)
from src.crawlers import nypost
from src.visualization import plot_candlestick

st.set_page_config(layout='wide')


@st.cache_data(persist=True)
def get_all_tickers():
    nasdaq_tickers = get_nasdaq_tickers()
    sp500_tickers = get_sp500_tickers()
    crypto_tickers = get_top_crypto_tickers()
    taiwan_sem_stickers = get_taiwan_sem_tickers()
    taiwan_otc_stickers = get_taiwan_otc_tickers()
    tickers = [nasdaq_tickers, sp500_tickers, crypto_tickers, taiwan_sem_stickers, taiwan_otc_stickers]
    tickers = list(itertools.chain(*tickers))
    tickers = sorted(list(set(tickers)))
    return tickers


def page_for_candlestick():
    tickers = get_all_tickers()
    num_tickers = len(tickers)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        symbol = st.selectbox(
            'Symbol', tickers, 
            index=tickers.index('AAPL'), 
            help=f'There are {num_tickers} tickers in total. Currently support NASDAQ, S&P500, Crypto, and Taiwan OTC.', 
            key=1
        )
    with col2:
        since = st.date_input(
            "Since", datetime.date(2022, 1, 1), key=2
        )
    with col3:
        until = st.date_input(
            # "Until", datetime.date(2023, 1, 1), key=3
            "Until", datetime.date.today(), key=3
        )
    with col4:
        interval = st.selectbox(
            'Interval', 
            ('1d', '5d', '1wk', '1mo', '3mo'), 
            index=0, 
            key=4
        )
    button = st.button('Run', key=5)

    if button:
        df = yf.download(symbol, start=since, end=until, interval=interval)
        df = df.reset_index(drop=False)
        df.columns = df.columns.str.lower()
        fig = plot_candlestick(df)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def page_for_strategies():
    tickers = get_all_tickers()
    num_tickers = len(tickers)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        symbol = st.selectbox(
            'Symbol', tickers, 
            index=tickers.index('AAPL'), 
            help=f'There are {num_tickers} tickers in total. Currently support NASDAQ, S&P500, Crypto, and Taiwan OTC.', 
            key=1
        )
    with col2:
        since = st.date_input(
            "Since", datetime.date(2022, 1, 1), key=2
        )
    with col3:
        until = st.date_input(
            # "Until", datetime.date(2023, 1, 1), key=3
            "Until", datetime.date.today(), key=3
        )
    with col4:
        interval = st.selectbox(
            'Interval', 
            ('1d', '5d', '1wk', '1mo', '3mo'), 
            index=0, 
            key=4
        )
    strategy = st.selectbox(
        'Strategy', 
        options=['SuperTrend'], 
        key=5
    )

    if strategy.lower() == 'supertrend':
        col1, col2 = st.columns([1, 1])
        with col1:
            length = st.number_input('Length', min_value=1, value=10)
        with col2:
            multiplier = st.number_input('Multiplier', min_value=1, value=3)

    button = st.button('Run', key=6)

    if button:
        df = yf.download(symbol, start=since, end=until, interval=interval)
        df = df.reset_index(drop=False)
        df.columns = df.columns.str.lower()

        if strategy.lower() == 'supertrend':
            supertrend_df = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
            supertrend_df.columns = ['trend', 'direction', 'long', 'short']
            supertrend_df = pd.concat([df, supertrend_df], axis=1)
            supertrend_df['sign'] = np.sign(supertrend_df['direction']).diff().ne(0)

            fig = make_subplots()
            price_scatter = go.Scatter(
                x=supertrend_df['date'],
                y=supertrend_df['close'],
                line=dict(color='white', width=1),
                name="Close Price", 
                opacity=.5
            )
            supertrend_upper_scatter = go.Scatter(
                x=supertrend_df['date'], 
                y=np.where(
                    supertrend_df.index.isin(supertrend_df.query('direction==1').index), 
                    supertrend_df['long'], 
                    np.nan
                ), 
                line=dict(color='green'), 
                name="SuperTrend Upper", 
                opacity=.9
            )
            supertrend_lower_scatter = go.Scatter(
                x=supertrend_df['date'], 
                y=np.where(
                    supertrend_df.index.isin(supertrend_df.query('direction==-1').index), 
                    supertrend_df['short'], 
                    np.nan
                ), 
                line=dict(color='red'), 
                name="SuperTrend Lower", 
                opacity=.9
            )
            fig.add_trace(
                price_scatter,
                secondary_y=False
            )
            fig.add_trace(
                supertrend_upper_scatter,
                secondary_y=False
            )
            fig.add_trace(
                supertrend_lower_scatter,
                secondary_y=False
            )
            for idx, row in supertrend_df.query('sign==1 & direction==1').iterrows():
                if idx == 0:
                    continue
                x, y = row['date'], row['trend']
                fig.add_annotation(
                    x=x,
                    y=y-5,
                    xref="x",
                    yref="y",
                    text="BUY", 
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=10,
                        color="#ffffff"
                    ),
                    align="center", 
                    bordercolor="#007500",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#007500",
                    opacity=0.8
                )
            for idx, row in supertrend_df.query('sign==1 & direction==-1').iterrows():
                if idx == 0:
                    continue
                x, y = row['date'], row['trend']
                fig.add_annotation(
                    x=x,
                    y=y+5,
                    xref="x",
                    yref="y",
                    text="SELL", 
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=10,
                        color="#ffffff"
                    ),
                    align="center", 
                    bordercolor="#DC143C",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#DC143C",
                    opacity=0.8
                )
            fig.update_traces(connectgaps=False)
            fig.update_layout(
                title='SuperTrend',
                xaxis_title='Date', 
                yaxis_title='Price', 
                xaxis_rangeslider_visible=False, 
                legend=dict(
                    title='', 
                    orientation="h", 
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ), 
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def page_for_news():
    fetch_or_update = st.sidebar.selectbox('Database', options=['Fetch', 'Update'])

    DETA_KEY = os.getenv('DETA_KEY') or st.secrets['NEWS_DATABASE']['DETA_KEY']
    deta = Deta(DETA_KEY)
    db = deta.Base("News")

    if fetch_or_update.lower() == 'fetch':
        sources = st.multiselect('Sources', options=['NYPOST'])
        sources = [source.lower() for source in sources]
        button = extra_button('Run', key='fetch_button')

        if button:
            res = db.fetch()
            df = pd.DataFrame(res.items)
            df = df.query('source in @sources')
            df = df[['title', 'date', 'time', 'link', 'source']]
            st.dataframe(filter_dataframe(df))
            
    elif fetch_or_update.lower() == 'update':
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            sources = st.multiselect('Sources', options=['NYPOST'])
            sources = [source.lower() for source in sources]
        with col2:
            pages = st.number_input('Pages', min_value=1, value=5)
        if 'nypost' in sources:
            sections = st.multiselect(
                'NYPOST Sections', 
                options=['business', 'tech', 'metro', 'sports', 'entertainment', 'opinion']
            )
        button = st.button('Run', key='update_button')

        if button:
            if 'nypost' in sources:
                data_dict = []
                pbar = stqdm(enumerate(sections), total=len(sections))
                for idx, section in pbar:
                    pbar.set_description(f'Download {section}')
                    base_url = nypost.URLS[section]
                    data_dict.extend(nypost.get_multi_pages(pages, base_url))
                df = pd.DataFrame(data_dict)
                # df['link'] = df['link'].apply(make_clickable)
                # st.write(df.to_html(escape=False), unsafe_allow_html=True)

                for data in stqdm(data_dict, desc='Store to DB'):
                    key = sha256(data['title'].encode('utf-8')).hexdigest()
                    db.put(data, key=key)

            st.success('Store to Database!')
        

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # modify = st.checkbox("Add filters")

    # if not modify:
    #     return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    # for col in df.columns:
    #     if is_object_dtype(df[col]):
    #         try:
    #             df[col] = pd.to_datetime(df[col])
    #         except Exception:
    #             pass

    #     if is_datetime64_any_dtype(df[col]):
    #         df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input.lower(), case=False)]

    return df


def main():
    page = st.sidebar.selectbox(
        label='MENU', 
        options=['Candlestick', 'Strategy', 'News']
    )
    if page.lower() == 'candlestick':
        page_for_candlestick()
    elif page.lower() == 'strategy':
        page_for_strategies()
    elif page.lower() == 'news':
        page_for_news()


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    hide_header_and_footer()
    css_file = read_css_file('static/css/main.css')
    st.markdown(css_file, unsafe_allow_html=True)

    main()