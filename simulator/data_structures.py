from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:  # Our own placed order
    order_id: int
    side: str
    size: float
    price: float
    timestamp: float


@dataclass
class CancelOrder:  # Our own cancel order
    order_id: int
    timestamp: float


@dataclass
class AnonTrade:  # Market trade
    exchange_ts: float
    receive_ts: float
    side: str
    size: float
    price: str


@dataclass(order=True)
class OwnTrade:  # Execution of own placed order
    exchange_ts: float
    receive_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    exchange_ts: float
    receive_ts: float
    asks: list[tuple[float, float]]  # tuple[price, size]
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None
