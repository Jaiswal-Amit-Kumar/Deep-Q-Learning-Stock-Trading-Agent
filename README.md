# Polymarket Python Integration (starter)

What this provides
- Minimal PolymarketClient with REST + WebSocket scaffolding
- SimpleOrderSimulator for dry-run testing
- AgentGateway to enforce basic risk rules before placing orders
- example_usage.py to show wiring

Safety first — read this
- This starter code uses a `dry_run` flag by default so nothing is sent to real endpoints.
- Never paste or share private keys in chat. Use environment variables, an OS-level secret store, or a vault.
- Start in Polymarket sandbox/test environment and verify behavior thoroughly before any production trades.
- Confirm automated trading is permitted for your account and that you comply with Polymarket Terms of Service and local regulations.

Setup
1. Create a Python virtualenv and install deps:
   pip install requests websockets python-dotenv

   Optional (if you implement wallet signing): pip install eth-account

2. Configure env vars (example):
   export POLY_API_KEY="your_api_key"
   export POLY_API_SECRET="your_api_secret"      # if applicable
   export POLY_WALLET_PRIVATE_KEY="..."         # only if using wallet-based signing

3. Run the example in dry-run:
   python example_usage.py

Notes / Next steps
- The Polymarket API may require signed requests or specific authentication flows (wallet signature). This starter leaves request signing as a TODO and uses an API key Bearer header if present.
- Replace base_url and ws_url with the official Polymarket sandbox or production endpoints (see https://docs.polymarket.com/).
- If you have sandbox credentials from Polymarket, switch `PolymarketClient(dry_run=False)` and test with small orders.
- Add robust error handling, rate-limiting, logging/metrics, persistence of orders/positions, and automated stop-loss logic before any production run.
- I can:
  - implement signing per Polymarket spec if you paste the relevant snippet from their docs (or I can fetch it),
  - add an adapter that converts your DQN action-space to market/order parameters,
  - open a PR in the repo you linked with these files (tell me repo owner/name and confirm).
