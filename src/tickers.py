import requests
import pandas as pd
from requests_html import HTMLSession


def get_taiwan_sem_tickers():
    """
    Taiwan stock exchange market
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    taiwan_securities_df = pd.read_html('https://www.twse.com.tw/rwd/en/afterTrading/BWIBBU_d?response=html')[0]
    taiwan_securities_df.columns = taiwan_securities_df.columns.get_level_values(1)
    taiwan_tickers = taiwan_securities_df['Security Code'].tolist()
    taiwan_tickers = [f'{ticker}.TW' for ticker in taiwan_tickers]
    taiwan_tickers = sorted(list(set(taiwan_tickers)))
    return taiwan_tickers


def get_taiwan_otc_tickers():
    """
    Taiwang over-the-counter market
    """
    res = requests.get('https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw')
    taiwan_tickers = [f'{row[0]}.TWO' for row in res.json()['aaData']]
    taiwan_tickers = sorted(list(set(taiwan_tickers)))
    return taiwan_tickers


def get_nasdaq_tickers():
    import io
    import ftplib

    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")
    reader = io.BytesIO()
    ftp.retrbinary('RETR nasdaqlisted.txt', reader.write)
    
    info = reader.getvalue().decode()
    splits = info.split("|")
    tickers = [x for x in splits if "N\r\n" in x]
    tickers = [x.replace("N\r\n", "") for x in tickers if 'File' not in x]
    tickers = sorted(list(set(tickers)))
    ftp.close()

    return tickers


def get_sp500_tickers():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sorted(sp500.Symbol.tolist())
    return tickers


def get_dow_tickers():
    site = r"https://finance.yahoo.com/quote/%5EDJI/components?p=%5EDJI"
    table = pd.read_html(site)[0]
    tickers = sorted(table['Symbol'].tolist())
    return tickers


def get_top_crypto_tickers():
    session = HTMLSession()
    res = session.get("https://finance.yahoo.com/cryptocurrencies?offset=0&count=100")
    tables = pd.read_html(res.html.raw_html)               
    df = tables[0].copy()

    def force_float(val):
        try:
            return float(val)
        except:
            return val
    
    
    df["% Change"] = df["% Change"].map(lambda x: float(x.strip("%").strip("+").replace(",", "")))
    del df["52 Week Range"]
    del df["Day Chart"]
    
    fields_to_change = [
        x for x in df.columns.tolist() if "Volume" in x or x == "Market Cap" or x == "Circulating Supply"
    ]
    
    for field in fields_to_change:
        if isinstance(df[field][0], str):
            df[field] = df[field].str.strip("B").map(force_float)
            df[field] = df[field].map(lambda x: x if type(x) == str else x * 1000000000)
            df[field] = df[field].map(lambda x: x if type(x) == float else force_float(x.strip("M")) * 1000000)
            
    session.close()
    tickers = sorted(df['Symbol'].tolist())
    return tickers