from typing import List

import pandas as pd

from simulator.simulator import AnonTrade, MdUpdate, OrderbookSnapshotUpdate


def load_trades(path, nrows=10000) -> List[AnonTrade]:
    '''
        This function downloads trades data

        Args:
            path(str): path to file
            nrows(int): number of rows to read

        Return:
            trades(List[AnonTrade]): list of trades 
    '''
    trades = pd.read_csv(path + 'trades.csv', nrows=nrows)

    # переставляю колонки, чтобы удобнее подавать их в конструктор AnonTrade
    trades = trades[['exchange_ts', 'receive_ts', 'aggro_side', 'size', 'price']].sort_values(
        ["exchange_ts", 'receive_ts'])
    receive_ts = trades.receive_ts.values
    exchange_ts = trades.exchange_ts.values
    trades = [AnonTrade(*args) for args in trades.values]
    return trades


def load_books(path, nrows=10000, btc=True) -> List[OrderbookSnapshotUpdate]:
    '''
        This function downloads orderbook market data

        Args:
            path(str): path to file
            nrows(int): number of rows to read

        Return:
            books(List[OrderbookSnapshotUpdate]): list of orderbooks snapshots 
    '''
    lobs = pd.read_csv(path + 'lobs.csv', nrows=nrows)

    # rename columns
    names = lobs.columns.values
    if btc:
        ln = len('btcusdt:Binance:LinearPerpetual_')
    else:
        ln = len('ethusdt:Binance:Spot_')
    renamer = {name: name[ln:] for name in names[2:]}
    renamer[' exchange_ts'] = 'exchange_ts'
    lobs.rename(renamer, axis=1, inplace=True)

    # timestamps
    receive_ts = lobs.receive_ts.values
    exchange_ts = lobs.exchange_ts.values
    # список ask_price, ask_vol для разных уровней стакана
    # размеры: len(asks) = 10, len(asks[0]) = len(lobs)
    asks = [list(zip(lobs[f"ask_price_{i}"], lobs[f"ask_vol_{i}"])) for i in range(10)]
    # транспонируем список
    asks = [[asks[i][j] for i in range(len(asks))] for j in range(len(asks[0]))]
    # тоже самое с бидами
    bids = [list(zip(lobs[f"bid_price_{i}"], lobs[f"bid_vol_{i}"])) for i in range(10)]
    bids = [[bids[i][j] for i in range(len(bids))] for j in range(len(bids[0]))]

    books = list(OrderbookSnapshotUpdate(*args) for args in zip(exchange_ts, receive_ts, asks, bids))
    return books


def merge_books_and_trades(books: List[OrderbookSnapshotUpdate], trades: List[AnonTrade]) -> List[MdUpdate]:
    '''
        This function merges lists of orderbook snapshots and trades 
    '''
    trades_dict = {(trade.exchange_ts, trade.receive_ts): trade for trade in trades}
    books_dict = {(book.exchange_ts, book.receive_ts): book for book in books}

    ts = sorted(trades_dict.keys() | books_dict.keys())

    md = [MdUpdate(*key, books_dict.get(key, None), trades_dict.get(key, None)) for key in ts]
    return md


def load_md_from_file(path: str, nrows=10000, btc=True) -> List[MdUpdate]:
    '''
        This function downloads orderbooks ans trades and merges them
    '''
    books = load_books(path, nrows, btc)
    trades = load_trades(path, nrows)
    return merge_books_and_trades(books, trades)
