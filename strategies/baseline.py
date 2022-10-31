import random
from datetime import timedelta

import pandas as pd

from simulator.simulator import MdUpdate, OwnTrade
from simulator.simulator import Simulator


class BaselineStrategy:
    def __init__(self, max_position: float, maker_fee: float = 0, time_to_cancel=0, trade_size=0.01) -> None:
        self.coin_position = 0
        self.money_position = 0
        self.max_position = max_position
        self.maker_fee = maker_fee
        self.trade_size = trade_size  # fixed trade size
        self.time_to_cancel = timedelta(milliseconds=time_to_cancel)  # waiting time before order cancellation

        self.best_bid = None
        self.best_ask = None
        self.current_time = pd.Timestamp(1655942402249000000)
        self.history = {'time': [], 'coin_position': [], 'money_position': [], 'mid_price': []}
        self.active_orders = []
        self.completed_trades = []
        self.prev_visualize_time = pd.Timestamp(1655942402249000000)  # the first timestamp in md

    def update_prices(self, md: MdUpdate) -> None:
        if md.orderbook:
            self.best_ask = md.orderbook.asks[0][0]
            self.best_bid = md.orderbook.bids[0][0]
            self.current_time = md.orderbook.receive_ts
        else:
            side = md.trade.side
            price = md.trade.price
            if side == 'BID':
                self.best_ask = price
            else:
                self.best_bid = price
            self.current_time = md.trade.receive_ts

    def run(self, sim: Simulator):
        while True:
            try:
                _, sim_update = sim.tick()
                if isinstance(sim_update, OwnTrade):
                    self.completed_trades.append(sim_update)
                    for order in self.active_orders:
                        if order.order_id == sim_update.order_id:
                            self.active_orders.remove(order)

                    if sim_update.side == 'ASK':
                        self.coin_position -= sim_update.size
                        self.money_position += sim_update.size * sim_update.price
                    else:
                        self.coin_position += sim_update.size
                        self.money_position -= sim_update.size * sim_update.price

                    mid_price = (self.best_bid + self.best_ask) / 2
                    self.history['time'].append(self.current_time)
                    self.history['coin_position'].append(self.coin_position)
                    self.history['money_position'].append(self.money_position)
                    self.history['mid_price'].append(mid_price)

                elif isinstance(sim_update, MdUpdate):
                    self.update_prices(sim_update)

                    for order in self.active_orders:
                        if self.current_time - order.timestamp >= self.time_to_cancel:
                            sim.cancel_order(order.order_id)
                            self.active_orders.remove(order)

                    if abs(self.coin_position) >= self.max_position and self.coin_position > 0:
                        side = 'ASK'
                    elif abs(self.coin_position) >= self.max_position and self.coin_position < 0:
                        side = 'BID'
                    else:
                        side = random.choice(['BID', 'ASK'])

                    price = self.best_ask if side == 'ASK' else self.best_bid
                    new_order = sim.place_order(side, self.trade_size, price)
                    self.active_orders.append(new_order)

                    if self.current_time - self.prev_visualize_time > timedelta(minutes=10):
                        print(f'Current time: {self.current_time}')
                        print(f'Coin Position: {self.coin_position}')
                        print(f'Money Position: {self.money_position}')
                        print()
                        self.prev_visualize_time = self.current_time

            except (StopIteration, IndexError):
                break

        return self.history, self.completed_trades
