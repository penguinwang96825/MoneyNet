import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_candlestick(df):
    candlestick = go.Candlestick(
        x=df['date'],
        open=df['open'], 
        high=df['high'],
        low=df['low'], 
        close=df['close'], 
        name='Candle'
    )
    scatter = go.Scatter(
        x=df['date'],
        y=df['close'],
        line=dict(color='white', width=1),
        name="Close Price", 
        opacity=.5
    )
    bar = go.Bar(
        x=df['date'], 
        y=df['volume'], 
        opacity=0.1, 
        marker_color='white', 
        name="Volume"
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        candlestick,
        secondary_y=False
    )
    fig.add_trace(
        scatter,
        secondary_y=False
    )
    fig.add_trace(
        bar,
        secondary_y=True
    )
    fig.layout.yaxis2.showgrid = False
    fig.update_layout(
        title='Candlestick',
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

    return fig