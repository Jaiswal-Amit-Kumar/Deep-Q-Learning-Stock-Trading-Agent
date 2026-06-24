"""
Polymarket client (minimal scaffolding) with request signing helpers
- Adds EIP-712 helper to sign ClobAuth messages (L1) using eth_account
- Adds HMAC-SHA256 request signing for API requests (L2)

Dependencies: requests, websockets, asyncio, python-dotenv (optional), eth-account
"""
import os
import time
import json
import logging
import asyncio
import requests
import hmac
import hashlib

from eth_account import Account
from eth_account.messages import encode_structured_data

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
        base_url="https://api.polymarket.com",
        ws_url="wss://ws.polymarket.com/stream",
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
            self.session.headers.update({"POLY_API_KEY": self.api_key})

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["POLY_API_KEY"] = self.api_key
        return headers

    # --- EIP-712 signing helper (L1) ---
    def eip712_sign_clob_auth(self, address=None, nonce=0, message_text=None, chain_id=137):
        """
        Produce an EIP-712 signature for the ClobAuth domain used by Polymarket to create API keys.
        Returns: dict with POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_NONCE
        """
        if not self.wallet_private_key:
            raise RuntimeError("wallet_private_key required for EIP-712 signing")

        signing_address = (address or Account.from_key(self.wallet_private_key).address).lower()
        ts = str(int(time.time() * 1000))
        nonce_val = int(nonce)
        message_text = message_text or "This message attests that I control the given wallet"

        domain = {
            "name": "ClobAuthDomain",
            "version": "1",
            "chainId": chain_id,
        }

        types = {
            "ClobAuth": [
                {"name": "address", "type": "address"},
                {"name": "timestamp", "type": "string"},
                {"name": "nonce", "type": "uint256"},
                {"name": "message", "type": "string"},
            ]
        }

        value = {
            "address": signing_address,
            "timestamp": ts,
            "nonce": nonce_val,
            "message": message_text,
        }

        structured = {"types": {"EIP712Domain": [], **types}, "domain": domain, "primaryType": "ClobAuth", "message": value}

        # eth_account expects the structured data to be encoded via encode_structured_data
        encoded = encode_structured_data(structured)
        signed = Account.sign_message(encoded, private_key=self.wallet_private_key)

        headers = {
            "POLY_ADDRESS": signing_address,
            "POLY_SIGNATURE": signed.signature.hex(),
            "POLY_TIMESTAMP": ts,
            "POLY_NONCE": str(nonce_val),
        }
        return headers

    # --- HMAC-SHA256 request signing (L2) ---
    def sign_request_headers(self, method, path, body=""):
        """
        Sign an API request using HMAC-SHA256.
        Returns a dict of headers to include: POLY_API_KEY, POLY_PASSPHRASE (if present), POLY_SIGNATURE, POLY_TIMESTAMP

        payload string format: timestamp + method + path + body
        signature: hex HMAC-SHA256(secret, payload)
        """
        if not self.api_key or not self.api_secret:
            raise RuntimeError("api_key and api_secret required for HMAC signing")

        timestamp = str(int(time.time()))
        method_up = (method or "GET").upper()
        path_only = path if path.startswith("/") else "/" + path
        body_str = body if isinstance(body, str) else (json.dumps(body) if body else "")

        payload = timestamp + method_up + path_only + body_str
        sig = hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        headers = {
            "POLY_TIMESTAMP": timestamp,
            "POLY_SIGNATURE": sig,
            "POLY_API_KEY": self.api_key,
        }
        # Optionally include passphrase if provided in env
        passphrase = os.environ.get("POLY_PASSPHRASE")
        if passphrase:
            headers["POLY_PASSPHRASE"] = passphrase
        return headers

    # --- API wrappers using signing ---
    def get_market(self, market_id):
        url = f"{self.base_url}/markets/{market_id}"
        path = f"/markets/{market_id}"
        LOG.info("GET %s", url)
        if self.dry_run:
            LOG.info("dry_run: returning fake market for %s", market_id)
            return {"id": market_id, "status": "open", "outcomes": ["Yes", "No"]}

        headers = self._headers()
        headers.update(self.sign_request_headers("GET", path, ""))
        r = self.session.get(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def place_order(self, market_id, outcome, size, price=None, order_type="limit", metadata=None):
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

        url = f"{self.base_url}/orders"
        path = "/orders"
        headers = self._headers()
        headers.update(self.sign_request_headers("POST", path, payload))
        r = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def cancel_order(self, order_id):
        LOG.info("Cancel order %s (dry_run=%s)", order_id, self.dry_run)
        if self.dry_run:
            return {"order_id": order_id, "status": "cancelled", "dry_run": True}
        url = f"{self.base_url}/orders/{order_id}"
        path = f"/orders/{order_id}"
        headers = self._headers()
        headers.update(self.sign_request_headers("DELETE", path, ""))
        r = self.session.delete(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    async def connect_ws(self, on_message, subscribe_payload=None, reconnect_delay=5):
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
