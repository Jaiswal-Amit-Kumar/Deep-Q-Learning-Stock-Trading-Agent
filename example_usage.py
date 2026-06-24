"""
Example usage that wires everything together in dry-run/simulated mode.
Adapt this to call your DQN agent's policy to get actions and pass them to the gateway.
"""
import os
import time
import logging
from polymarket_client import PolymarketClient
from simulator import SimpleOrderSimulator
from agent_gateway import AgentGateway

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("example_usage")


def main():
    # Configure via env vars or directly here
    # IMPORTANT: do NOT hardcode private keys in source.
    client = PolymarketClient(
        api_key=os.environ.get("POLY_API_KEY"),
        base_url="https://api.polymarket.com",
        ws_url="wss://ws.polymarket.com/stream",
        dry_run=True,  # keep True while testing
    )

    sim = SimpleOrderSimulator(fill_delay=2.0)
    gateway = AgentGateway(client, simulator=sim, max_trade_size=50.0, allowed_markets=None, min_cooldown=0.5)

    # Example agent action (replace with your DQN's action output)
    action = {
        "market_id": "example-market-123",
        "outcome": "Yes",
        "size": 10.0,
        "price": 0.6,
        "order_type": "limit",
    }

    LOG.info("Sending action to gateway: %s", action)
    order = gateway.execute_action(action)
    LOG.info("Order response: %s", order)

    # Wait a bit and examine simulator state
    time.sleep(3)
    if client.dry_run:
        LOG.info("Simulated order status: %s", sim.get_order(order['order_id']))


if __name__ == "__main__":
    main()
