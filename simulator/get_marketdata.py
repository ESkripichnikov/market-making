import re

import pandas as pd
from tqdm import tqdm

from simulator.data_structures import OrderbookSnapshotUpdate, MdUpdate, AnonTrade


def load_data(path: str):
    lobs_df = pd.read_csv(path + '/lobs.csv')
    trades_df = pd.read_csv(path + '/trades.csv')

    lobs_df.rename(columns=lambda x: re.sub('.*LinearPerpetual_', '', x).strip(), inplace=True)

    for column in ['receive_ts', 'exchange_ts']:
        lobs_df[column] = pd.to_datetime(lobs_df[column])
        trades_df[column] = pd.to_datetime(trades_df[column])

    return lobs_df, trades_df


def get_marketdata(row, lobs: bool):
    if lobs:
        orderbook = OrderbookSnapshotUpdate(
            exchange_ts=row['exchange_ts'],
            receive_ts=row['receive_ts'],
            asks=row['asks'],
            bids=row['bids']
        )

        return MdUpdate(orderbook, None)
    else:
        trade = AnonTrade(
            exchange_ts=row['exchange_ts'],
            receive_ts=row['receive_ts'],
            side=row['aggro_side'],
            size=row['size'],
            price=row['price']
        )

        return MdUpdate(None, trade)


def load_md_from_file(path: str) -> list[MdUpdate]:
    lobs_df, trades_df = load_data(path)
    print('Data loaded successfully')

    prices_and_volumes = ['ask_price', 'ask_vol', 'bid_price', 'bid_vol']
    for filt in prices_and_volumes:
        lobs_df[filt] = lobs_df.filter(regex=filt).values.tolist()

    tqdm.pandas(desc="Lobs asks aggregating")
    lobs_df['asks'] = lobs_df[['ask_price', 'ask_vol']].progress_apply(lambda x: list(zip(x[0], x[1])), axis=1)
    tqdm.pandas(desc="Lobs bids aggregating")
    lobs_df['bids'] = lobs_df[['bid_price', 'bid_vol']].progress_apply(lambda x: list(zip(x[0], x[1])), axis=1)

    tqdm.pandas(desc="Lobs market data generating")
    md_lobs = lobs_df.progress_apply(get_marketdata, axis=1, lobs=True).values.tolist()
    tqdm.pandas(desc="Trades market data generating")
    md_trades = trades_df.progress_apply(get_marketdata, axis=1, lobs=False).values.tolist()
    md_queue = md_lobs + md_trades
    print('Sorting')
    md_queue = sorted(
        md_queue, key=lambda md_update: md_update.orderbook.receive_ts
        if md_update.orderbook else md_update.trade.receive_ts
    )

    return md_queue
