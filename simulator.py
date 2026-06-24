"""
Simple simulator used for local testing:
- Keeps a list of simulated orders
- Simulates fills after a configurable delay
This is useful for validating your agent integration without Polymarket sandbox access.
"""
import time
import threading
import logging

LOG = logging.getLogger("simulator")
LOG.setLevel(logging.INFO)


class SimpleOrderSimulator:
    def __init__(self, fill_delay=1.0):
        self.fill_delay = fill_delay
        self.orders = {}
        self._lock = threading.Lock()

    def place_order(self, order_payload):
        order_id = f"sim-{int(time.time()*1000)}"
        order = {
            "order_id": order_id,
            "payload": order_payload,
            "status": "pending",
            "placed_at": time.time(),
        }
        with self._lock:
            self.orders[order_id] = order
        # spawn fill thread
        t = threading.Thread(target=self._fill_order, args=(order_id,), daemon=True)
        t.start()
        LOG.info("Simulator placed order %s", order_id)
        return order

    def _fill_order(self, order_id):
        time.sleep(self.fill_delay)
        with self._lock:
            if order_id in self.orders:
                self.orders[order_id]["status"] = "filled"
                self.orders[order_id]["filled_at"] = time.time()
                LOG.info("Simulator filled order %s", order_id)

    def cancel_order(self, order_id):
        with self._lock:
            if order_id in self.orders and self.orders[order_id]["status"] == "pending":
                self.orders[order_id]["status"] = "cancelled"
                LOG.info("Simulator cancelled order %s", order_id)
                return True
        return False

    def get_order(self, order_id):
        with self._lock:
            return self.orders.get(order_id)
