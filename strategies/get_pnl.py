import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_pnl_df(history):
    df = pd.DataFrame({
        'time': history['time'],
        'coin_position': history['coin_position'],
        'money_position': history['money_position'],
        'mid_price': history['mid_price'],
    })
    df['pnl'] = df.money_position + df.coin_position * df.mid_price
    return df


def visualize_pnl(df, freq=50, save_path=None):
    fig, _ = plt.subplots(1, 1, figsize=(15, 5))
    sns.lineplot(x=df.index.values[::freq], y=df.pnl.values[::freq])
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for label in plt.gca().get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.title('PnL')
    plt.xlabel('Time')
    plt.ylabel('USDT')

    plt.show()
    if save_path:
        fig.savefig(save_path)
