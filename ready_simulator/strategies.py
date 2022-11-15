from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(self, delay: float, hold_time: Optional[float] = None) -> None:
        """
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        """
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Order]): list of all placed orders
        """

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', best_bid)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', best_ask)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders


class StoikovStrategy:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            trade_size: float,
            delay: float,
            terminal_date,
            k=1.5,
            hold_time: Optional[float] = None,
            risk_preference: Optional[float] = 0.01,
            initial_vol: Optional[float] = 2.19346e-08,
            vol_freq: Optional[float] = 1,
            lamb: Optional[float] = 0.95,

    ) -> None:
        """
            Args:
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
                vol_freq:: volatility frequency in seconds
                risk_preference: >0 for risk aversion, ~0 for risk-neutrality, or <0 for risk-seeking
                initial_vol: initial volatility estimated on history
                vol_freq: volatility frequency in seconds
                lamb: lambda in EWMA for updating volatility
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.risk_preference = risk_preference
        self.current_time = None
        self.coin_position = 0
        self.prev_midprice = None
        self.current_midprice = None
        self.terminal_date = terminal_date

        self.volatility = initial_vol
        self.lamb = lamb
        self.vol_freq = pd.Timedelta(vol_freq, 's').delta
        self.reservation_price = None
        self.spread = None
        self.k = k

        self.quotes_history = []

    def update_volatility(self) -> None:
        ret = (self.current_midprice - self.prev_midprice) / self.prev_midprice
        self.volatility = self.lamb * self.volatility + (1 - self.lamb) * ret ** 2

    def update_reservation_price(self) -> None:
        time_to_terminal = (self.terminal_date - pd.to_datetime(self.current_time)).delta / self.vol_freq

        self.reservation_price = (
                self.current_midprice - (self.coin_position / self.trade_size)
                * self.risk_preference * self.volatility * time_to_terminal
        )

    def update_spread(self) -> None:
        time_to_terminal = (self.terminal_date - pd.to_datetime(self.current_time)).delta / self.vol_freq

        self.spread = (
                self.risk_preference * self.volatility * time_to_terminal +
                (2 / self.risk_preference) * np.log(1 + self.risk_preference / self.k)
        )

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                quotes_history: list of tuples(time, coin_pos, bid, mid, reservation, ask)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        vol_prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.current_midprice = (best_bid + best_ask) / 2

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - vol_prev_time >= self.vol_freq:
                if self.prev_midprice:
                    self.update_volatility()

                self.prev_midprice = self.current_midprice
                vol_prev_time = self.current_time

            if self.current_time - prev_time >= self.delay:
                # place order
                self.update_reservation_price()
                self.update_spread()

                bid_price = round(self.reservation_price - self.spread / 2, 1)  # increment
                ask_price = round(self.reservation_price + self.spread / 2, 1)
                bid_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                ask_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                prev_time = self.current_time
                self.quotes_history.append((self.current_time, self.coin_position,
                                            bid_price, self.current_midprice, self.reservation_price, ask_price))

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.quotes_history

    @staticmethod
    def visualize_bids(quotes_history, freq=10):
        time, pos, bid, mid, reservation, ask = list(map(list, zip(*quotes_history[::freq])))

        df = pd.DataFrame([time, pos, bid, mid, reservation, ask]).T
        df.columns = ['time', 'pos', 'bid', 'mid', 'reservation', 'ask']
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df['bid'], name="bid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['mid'], name="mid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['reservation'], name="reservation"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['ask'], name="ask"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['pos'], name="pos"),
            secondary_y=True)

        fig.update_layout(
            title_text="The mid-price and the optimal bid and ask quotes"
        )

        fig.update_xaxes(title_text="Time")

        fig.update_yaxes(title_text="<b>Prices</b>: USDT", secondary_y=False)
        fig.update_yaxes(title_text="<b>Coin Position</b>: BTC", secondary_y=True)
        fig.show()
        return fig, df


class LimitMarketStrategy:
    """
        This strategy places limit or market orders every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            line_coefficients: Tuple[float, float],
            parabola_coefficients: Tuple[float, float, float],
            trade_size: Optional[float] = 0.001,
            price_tick: Optional[float] = 0.1,
            delay: Optional[int] = 1e8,
            hold_time: Optional[int] = 1e10
    ) -> None:
        """
            Args:
                line_coefficients: line coefficients [k, b] y = kx + b
                parabola_coefficients: parabola coefficients [a, b, c] y = ax^2 + bx + c
                trade_size: volume of each trade
                price_tick: a value by which we increase a bid (reduce an ask) limit order
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.current_time = None
        self.coin_position = 0
        self.prev_midprice = None
        self.current_midprice = None
        self.current_spread = None
        self.price_tick = price_tick

        self.line_k, self.line_b = line_coefficients
        self.parabola_a, self.parabola_b, self.parabola_c = parabola_coefficients

        self.actions_history = []

    def get_normalized_data(self) -> Tuple[float, float]:
        # implement normalization
        return self.coin_position, self.current_spread

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                actions_history: list of tuples(time, coin_pos, spread, action)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.md_list.append(update)
                    self.current_spread = best_ask - best_bid

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - prev_time >= self.delay:
                # place order
                inventory, spread = self.get_normalized_data()

                if (self.parabola_a * inventory ** 2 + self.parabola_b * inventory + self.parabola_c) > spread:
                    bid_market_order = sim.place_order(self.current_time, self.trade_size, 'BID', best_ask)
                    ongoing_orders[bid_market_order.order_id] = bid_market_order
                    action = 'market buy'
                elif (self.parabola_a * inventory ** 2 + self.parabola_b * (-inventory) + self.parabola_c) > spread:
                    ask_market_order = sim.place_order(self.current_time, self.trade_size, 'ASK', best_bid)
                    ongoing_orders[ask_market_order.order_id] = ask_market_order
                    action = 'market sell'
                else:
                    above_line1 = (self.line_k * inventory + self.line_b) < spread
                    above_line2 = (self.line_k * (-inventory) + self.line_b) < spread

                    bid_price = best_bid + self.price_tick * above_line1
                    ask_price = best_ask - self.price_tick * above_line2

                    bid_limit_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                    ask_limit_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                    ongoing_orders[bid_limit_order.order_id] = bid_limit_order
                    ongoing_orders[ask_limit_order.order_id] = ask_limit_order
                    action = 'limit order'

                prev_time = self.current_time
                self.actions_history.append((self.current_time, self.coin_position,
                                             self.current_spread, action))

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.actions_history
