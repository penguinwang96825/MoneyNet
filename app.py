import datetime
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import hide_header_and_footer, read_css_file
from src.tickers import (
    get_nasdaq_tickers, 
    get_sp500_tickers, 
    get_top_crypto_tickers, 
    get_taiwan_sem_tickers, 
    get_taiwan_otc_tickers
)
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


def main():
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
            "Until", datetime.date(2023, 1, 1), key=3
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

        supertrend_df = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        supertrend_df.columns = ['trend', 'direction', 'long', 'short']
        supertrend_df = pd.concat([df, supertrend_df], axis=1)
        supertrend_df['sign'] = np.sign(supertrend_df['direction']).diff().ne(0)
        print(supertrend_df.query('sign==1 & direction==1'))
        print(supertrend_df.query('sign==1 & direction==-1'))

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


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    hide_header_and_footer()
    css_file = read_css_file('static/css/main.css')
    st.markdown(css_file, unsafe_allow_html=True)

    main()