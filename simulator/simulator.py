from collections import deque
from datetime import timedelta
from queue import PriorityQueue

import pandas as pd

from simulator.data_structures import MdUpdate, Order, OwnTrade, CancelOrder
from simulator.get_marketdata import load_md_from_file


class Simulator:
    def __init__(self, path: str, execution_latency: float, md_latency: float, ready_md=None) -> None:
        self.execution_latency = timedelta(milliseconds=execution_latency)
        self.md_latency = timedelta(milliseconds=md_latency)
        self.md_queue = deque(ready_md) if ready_md else deque(load_md_from_file(path))
        self.actions_queue = deque()
        self.strategy_updates_queue = PriorityQueue()
        self.strategy_updates_queue.put((self.get_event_time('md'), self.md_queue[0]))
        self.best_bid = 0
        self.best_ask = 0

        self.current_time = self.strategy_updates_queue.queue[0][0]
        self.active_orders = []
        self.order_id = 1
        self.trade_id = 1

    def get_event_time(self, style):
        if style == 'md' and self.md_queue:
            event_time = (
                self.md_queue[0].orderbook.exchange_ts if self.md_queue[0].orderbook else
                self.md_queue[0].trade.exchange_ts
            )
        elif style == 'strategy' and not self.strategy_updates_queue.empty():
            event_time, _ = self.strategy_updates_queue.queue[0]

        elif style == 'actions' and self.actions_queue:
            event_time = self.actions_queue[0].timestamp
        else:
            event_time = pd.Timestamp.max

        return event_time

    def apply_md_update(self, md: MdUpdate):
        if md.orderbook:
            self.best_ask = md.orderbook.asks[0][0]
            self.best_bid = md.orderbook.bids[0][0]
            receive_ts = md.orderbook.receive_ts
        else:
            side = md.trade.side
            price = md.trade.price
            if side == 'BID':
                self.best_ask = price
            else:
                self.best_bid = price
            receive_ts = md.trade.receive_ts

        self.strategy_updates_queue.put((receive_ts, md))

    def tick(self):
        strategy_time = self.get_event_time('strategy')
        md_time = self.get_event_time('md')
        actions_time = self.get_event_time('actions')

        while md_time <= strategy_time or actions_time <= strategy_time:
            if md_time < actions_time:
                self.apply_md_update(self.md_queue.popleft())
                self.execute_orders()

                self.current_time = md_time
                md_time = self.get_event_time('md')
                strategy_time = self.get_event_time('strategy')
            else:
                self.prepare_orders(self.actions_queue.popleft())
                self.execute_orders()

                self.current_time = actions_time
                actions_time = self.get_event_time('actions')
                strategy_time = self.get_event_time('strategy')

        return self.strategy_updates_queue.get_nowait()

    def check_order(self, order: Order) -> bool:
        if order.side == 'BID' and order.price >= self.best_ask:
            return 1

        if order.side == 'ASK' and order.price <= self.best_bid:
            return 1

        return 0

    def prepare_orders(self, action_order) -> None:
        if isinstance(action_order, Order):
            self.active_orders.append(action_order)
        else:
            for order in self.active_orders:
                if order.order_id == action_order.order_id:
                    self.active_orders.remove(order)

    def execute_orders(self) -> list[Order]:
        for order in self.active_orders:
            if self.check_order(order):
                self.strategy_updates_queue.put(
                    (self.current_time + self.md_latency,
                     OwnTrade(
                         exchange_ts=self.current_time,
                         receive_ts=self.current_time + self.md_latency,
                         trade_id=self.trade_id,
                         order_id=order.order_id,
                         side=order.side,
                         size=order.size,
                         price=order.price
                     )
                     )
                )

                self.trade_id += 1
                self.active_orders.remove(order)

        return self.active_orders

    def place_order(self, side: str, size: float, price: float) -> Order:
        new_order = Order(
            order_id=self.order_id,
            side=side,
            size=size,
            price=price,
            timestamp=self.current_time + self.execution_latency
        )
        self.actions_queue.append(new_order)
        self.order_id += 1
        return new_order

    def cancel_order(self, order_id: int) -> None:
        new_cancel_order = CancelOrder(
            order_id=order_id,
            timestamp=self.current_time + self.execution_latency
        )
        self.actions_queue.append(new_cancel_order)
