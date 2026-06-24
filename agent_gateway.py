"""
AgentGateway: safe wrapper between your agent and PolymarketClient (or simulator).
- Risk checks: max_trade_size, allowed_markets, cooldown
- Dry-run toggle: if dry_run True, no real orders are sent
- Maps high-level agent actions into order API calls
"""
import time
import logging

LOG = logging.getLogger("agent_gateway")
LOG.setLevel(logging.INFO)


class AgentGateway:
    def __init__(self, client, simulator=None, max_trade_size=100.0, allowed_markets=None, min_cooldown=1.0):
        """
        client: PolymarketClient instance
        simulator: optional SimpleOrderSimulator — if provided and client.dry_run True, will use it for fills
        """
        self.client = client
        self.simulator = simulator
        self.max_trade_size = max_trade_size
        self.allowed_markets = set(allowed_markets or [])
        self.min_cooldown = min_cooldown
        self._last_trade_ts = 0

    def _check_risk(self, market_id, size):
        if self.allowed_markets and market_id not in self.allowed_markets:
            raise ValueError(f"market {market_id} not allowed by gateway policy")
        if size <= 0:
            raise ValueError("size must be > 0")
        if size > self.max_trade_size:
            raise ValueError(f"size {size} exceeds max_trade_size {self.max_trade_size}")
        if time.time() - self._last_trade_ts < self.min_cooldown:
            raise RuntimeError("cooldown: trades are too frequent")

    def execute_action(self, action):
        """
        action: dict expected fields:
          - 'market_id' (str)
          - 'side' ('buy' or 'sell') or 'outcome'
          - 'outcome' (optional)
          - 'size' (float)
          - 'price' (optional)
          - 'order_type' (optional)
        Returns: order response (real or simulated)
        """
        market_id = action.get("market_id")
        size = float(action.get("size", 0))
        outcome = action.get("outcome") or action.get("side")
        price = action.get("price")
        order_type = action.get("order_type", "limit")

        self._check_risk(market_id, size)

        payload = {
            "market_id": market_id,
            "outcome": outcome,
            "size": size,
            "price": price,
            "order_type": order_type,
            "agent_tag": action.get("agent_tag", "dqn-agent"),
        }

        LOG.info("Gateway executing action: %s", payload)

        if self.client.dry_run and self.simulator:
            order = self.simulator.place_order(payload)
        else:
            order = self.client.place_order(
                market_id=market_id, outcome=outcome, size=size, price=price, order_type=order_type, metadata={"agent": "dqn-agent"}
            )

        self._last_trade_ts = time.time()
        return order
