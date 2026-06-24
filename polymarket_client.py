"""
Polymarket client (minimal scaffolding)
- REST and WebSocket examples
- Supports dry_run mode (no network side-effects)
- Placeholder for signing requests if wallet-based auth is required by Polymarket

Dependencies: requests, websockets, asyncio, python-dotenv (optional), eth-account (optional for signing)
"""
import os
import time
import json
import logging
import asyncio
import requests

try:
    import websockets
except Exception:
    websockets = None

LOG = logging.getLogger("polymarket_client")
LOG.setLevel(logging.INFO)


class PolymarketClient:
    def __init__(
        self,
        api_key=None,
        api_secret=None,
        wallet_private_key=None,
        base_url="https://api.polymarket.com",  # replace with official base if different
        ws_url="wss://ws.polymarket.com/stream",  # replace with official ws if different
        dry_run=True,
        timeout=10,
    ):
        self.api_key = api_key or os.environ.get("POLY_API_KEY")
        self.api_secret = api_secret or os.environ.get("POLY_API_SECRET")
        self.wallet_private_key = wallet_private_key or os.environ.get("POLY_WALLET_PRIVATE_KEY")
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.dry_run = dry_run
        self.timeout = timeout
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_market(self, market_id):
        url = f"{self.base_url}/markets/{market_id}"
        LOG.info("GET %s", url)
        if self.dry_run:
            LOG.info("dry_run: returning fake market for %s", market_id)
            return {"id": market_id, "status": "open", "outcomes": ["Yes", "No"]}
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def place_order(self, market_id, outcome, size, price=None, order_type="limit", metadata=None):
        """
        Place an order on a market.
        - market_id: str
        - outcome: str or index
        - size: float (money to invest)
        - price: float (for limit orders) or None
        - order_type: 'limit' or 'market' (behavior depends on Polymarket API)
        """
        payload = {
            "market_id": market_id,
            "outcome": outcome,
            "size": float(size),
            "order_type": order_type,
            "price": None if price is None else float(price),
            "metadata": metadata or {},
            "timestamp": int(time.time() * 1000),
        }

        LOG.info("Placing order (dry_run=%s): %s", self.dry_run, payload)
        if self.dry_run:
            # Return a fake response that looks like an order accepted response
            fake_order = {
                "order_id": f"dry-{int(time.time()*1000)}",
                "status": "accepted",
                "placed_at": time.time(),
                "market_id": market_id,
                "outcome": outcome,
                "size": size,
                "price": price,
            }
            return fake_order

        # Real request flow (may need signing depending on Polymarket)
        url = f"{self.base_url}/orders"
        headers = self._headers()
        # TODO: If Polymarket requires signed payloads with wallet, sign here per their docs.
        r = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def cancel_order(self, order_id):
        LOG.info("Cancel order %s (dry_run=%s)", order_id, self.dry_run)
        if self.dry_run:
            return {"order_id": order_id, "status": "cancelled", "dry_run": True}
        url = f"{self.base_url}/orders/{order_id}"
        r = self.session.delete(url, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    async def connect_ws(self, on_message, subscribe_payload=None, reconnect_delay=5):
        """
        Lightweight websocket loop. on_message is a sync or async callable taking (message_dict).
        subscribe_payload: optional JSON to send on connect to subscribe to markets.
        """
        if websockets is None:
            raise RuntimeError("websockets library not available; install 'websockets'")

        while True:
            try:
                LOG.info("Connecting to WS %s", self.ws_url)
                async with websockets.connect(self.ws_url) as ws:
                    if subscribe_payload:
                        LOG.info("Subscribing with payload: %s", subscribe_payload)
                        await ws.send(json.dumps(subscribe_payload))
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            msg = {"raw": raw}
                        if asyncio.iscoroutinefunction(on_message):
                            await on_message(msg)
                        else:
                            on_message(msg)
            except Exception as e:
                LOG.exception("WebSocket connection error: %s — reconnecting in %s seconds", e, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
