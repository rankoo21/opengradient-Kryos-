import asyncio
import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
import requests
from typing import List, Dict, Any

import opengradient as og
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="OG DeFi Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenGradient LLM ──────────────────────────────────────────────────────────
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        pk = os.environ.get("OG_PRIVATE_KEY")
        if not pk:
            raise ValueError("OG_PRIVATE_KEY not set in .env")
        _llm = og.LLM(private_key=pk)
        # Try to refresh approval; if wallet balance is too low, rely on existing on-chain approval.
        try:
            _llm.ensure_opg_approval(10.0)
        except Exception as e:
            print(f"[OG] ensure_opg_approval skipped: {e}")
    return _llm

# ── Price Alerts (in-memory) ──────────────────────────────────────────────────
PRICE_ALERTS: List[dict] = []

# ── News RSS feeds ────────────────────────────────────────────────────────────
NEWS_FEEDS = {
    "coindesk":     "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt":      "https://decrypt.co/feed",
}

def fetch_rss_news(source: str = "cointelegraph", limit: int = 6) -> list:
    url = NEWS_FEEDS.get(source, NEWS_FEEDS["cointelegraph"])
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OGAdvisor/1.0)"}
    r = requests.get(url, headers=headers, timeout=8)
    root = ET.fromstring(r.content)
    articles = []
    for item in root.findall(".//item")[:limit]:
        title   = item.find("title")
        link    = item.find("link")
        pubdate = item.find("pubDate")
        desc    = item.find("description")
        if title is None:
            continue
        # Strip CDATA / HTML from description
        raw_desc = (desc.text or "") if desc is not None else ""
        clean_desc = raw_desc.replace("<![CDATA[", "").replace("]]>", "").strip()[:140]
        articles.append({
            "title":  title.text or "",
            "link":   link.text or "#",
            "date":   pubdate.text or "",
            "source": source,
            "summary": clean_desc
        })
    return articles

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are **Kryos** — an elite AI DeFi and crypto intelligence analyst, powered by OpenGradient's TEE-verified decentralized AI infrastructure.

Your expertise:
• Real-time crypto market analysis with live price data and charts
• DeFi protocol yields, risks, and strategy (Aave, Compound, Uniswap, etc.)
• Portfolio risk management and asset allocation
• Multi-chain EVM wallet portfolio analysis (ETH/Base/Arbitrum/Optimism/Polygon) with per-token risk & upside
• Gas fee optimization across chains
• Market sentiment analysis (Fear & Greed)
• Price alert management

You have real-time tools — use them proactively when the user asks about specific coins, protocols, wallets, or market conditions.
Be concise, data-driven, and sharp. Use **bold** for key numbers. Keep responses well-structured. Always prioritize actionable insights.

**Portfolio analysis rules** (when `get_wallet_portfolio` is used):
- Start with: total portfolio value in USD + which chains have assets (e.g. "Active on: Ethereum, Base, Arbitrum").
- Show native balances per chain (ETH, POL) if any.
- For EACH significant token (>1% of portfolio), give: **RISK** (Low/Medium/High/Very High) + **POTENTIAL** (Low/Medium/High) + a 1-line reason. Mention the chain it's on.
- Risk factors: tiny market cap (<$10M = Very High), unknown/meme tokens, concentration on one chain, stablecoin de-peg risk.
- Potential factors: narrative strength, market cap room to grow, recent momentum, L2 adoption.
- End with 2-3 actionable recommendations (rebalance, bridge, take profits, diversify chains, etc.)."""

# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": "Get real-time price, 24h change, and market cap of a cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "CoinGecko coin ID e.g. 'bitcoin', 'ethereum', 'solana'"}
                },
                "required": ["coin"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_coins",
            "description": "Get market overview of top N cryptocurrencies by market cap",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of top coins (max 10)"}
                },
                "required": ["limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_defi_yields",
            "description": "Get DeFi yield farming and lending opportunities for a protocol",
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string", "description": "Protocol: 'aave', 'compound', or 'uniswap'"}
                },
                "required": ["protocol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_chart",
            "description": "Get historical price chart data for a coin to visualize price trends",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "CoinGecko coin ID e.g. 'bitcoin', 'ethereum'"},
                    "days": {"type": "integer", "description": "Number of days of history: 1, 7, 14, or 30"}
                },
                "required": ["coin", "days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_gas_fees",
            "description": "Get current gas fees across multiple blockchain networks (ETH, Base, Polygon, Arbitrum)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_wallet",
            "description": "Check ETH balance and USD value of an Ethereum wallet address",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Ethereum wallet address starting with 0x"},
                    "chain": {"type": "string", "description": "Chain: 'ethereum', 'base', or 'polygon'"}
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Get crypto market sentiment including Fear & Greed index and Bitcoin dominance",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_news",
            "description": "Get the latest cryptocurrency news headlines from top crypto news sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "News source: 'cointelegraph', 'coindesk', or 'decrypt'"},
                    "limit": {"type": "integer", "description": "Number of articles to return (max 6)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_wallet_portfolio",
            "description": "Analyze a wallet's full portfolio across 5 EVM chains (Ethereum, Base, Arbitrum, Optimism, Polygon). Returns native balances (ETH/POL) + all ERC-20 tokens with price, market cap. Use when user asks to analyze, review, or audit a wallet's portfolio / holdings / what they hold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Ethereum wallet address starting with 0x"}
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_price_alert",
            "description": "Create a price alert that will notify when a coin reaches a target price",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "CoinGecko coin ID e.g. 'bitcoin'"},
                    "condition": {"type": "string", "description": "Condition: '>' (above) or '<' (below)"},
                    "target_price": {"type": "number", "description": "Target price in USD"}
                },
                "required": ["coin", "condition", "target_price"]
            }
        }
    }
]

# ── RPC helpers ───────────────────────────────────────────────────────────────
RPCS = {
    "ethereum": [
        "https://eth.llamarpc.com",
        "https://ethereum-rpc.publicnode.com",
        "https://cloudflare-eth.com",
        "https://rpc.payload.de",
    ],
    "base": [
        "https://mainnet.base.org",
        "https://base.llamarpc.com",
        "https://base-rpc.publicnode.com",
    ],
    "polygon": [
        "https://polygon-rpc.com",
        "https://polygon-bor-rpc.publicnode.com",
        "https://polygon.llamarpc.com",
    ],
    "arbitrum": [
        "https://arb1.arbitrum.io/rpc",
        "https://arbitrum.llamarpc.com",
        "https://arbitrum-one-rpc.publicnode.com",
    ],
}

def rpc_call(chain: str, method: str, params: list) -> dict:
    """Try multiple RPC endpoints until one succeeds (or returns a valid JSON-RPC result)."""
    urls = RPCS.get(chain, RPCS["ethereum"])
    last_err = None
    for url in urls:
        try:
            r = requests.post(url,
                              json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                              timeout=6)
            data = r.json()
            if "result" in data:
                return data
            last_err = data.get("error", {"message": f"HTTP {r.status_code}"})
        except Exception as e:
            last_err = {"message": str(e)}
            continue
    return {"error": last_err or "All RPC endpoints failed"}

# ── Multi-chain EVM portfolio ────────────────────────────────────────────────
CHAINS = {
    "ethereum": {"blockscout": "https://eth.blockscout.com",      "native": "ETH",   "native_cg": "ethereum"},
    "base":     {"blockscout": "https://base.blockscout.com",     "native": "ETH",   "native_cg": "ethereum"},
    "arbitrum": {"blockscout": "https://arbitrum.blockscout.com", "native": "ETH",   "native_cg": "ethereum"},
    "optimism": {"blockscout": "https://optimism.blockscout.com", "native": "ETH",   "native_cg": "ethereum"},
    "polygon":  {"blockscout": "https://polygon.blockscout.com",  "native": "POL",   "native_cg": "matic-network"},
}

def _fetch_chain(address: str, chain: str, cfg: dict, native_prices: dict) -> dict:
    """Fetch native + ERC-20 holdings for one chain via Blockscout."""
    result = {"chain": chain, "native_balance": 0, "native_usd": 0, "tokens": [], "error": None}
    base_url = cfg["blockscout"]

    # Native balance
    try:
        r = requests.get(f"{base_url}/api/v2/addresses/{address}", timeout=8)
        if r.status_code == 200:
            d = r.json()
            wei = int(d.get("coin_balance") or 0)
            bal = wei / 1e18
            price = native_prices.get(cfg["native_cg"], 0)
            result["native_balance"] = round(bal, 6)
            result["native_usd"] = round(bal * price, 2)
            result["native_symbol"] = cfg["native"]
    except Exception as e:
        result["error"] = f"native: {e}"

    # ERC-20 tokens
    try:
        r = requests.get(f"{base_url}/api/v2/addresses/{address}/tokens",
                         params={"type": "ERC-20"}, timeout=10)
        if r.status_code != 200:
            return result
        for item in (r.json().get("items") or []):
            tok = item.get("token", {}) or {}
            decimals = int(tok.get("decimals") or 18)
            try:
                bal = float(item.get("value", 0)) / (10 ** decimals)
            except Exception:
                bal = 0
            try:
                price = float(tok.get("exchange_rate") or 0)
            except Exception:
                price = 0
            usd_value = bal * price

            try:
                mcap = float(tok.get("circulating_market_cap") or 0)
            except Exception:
                mcap = 0

            # Spam filter (airdrop scam detection):
            # - Must have real price AND real market cap
            # - Min market cap $5M (legitimate tokens)
            # - Position must be < 1% of circulating mcap (else = mint-spam)
            # - Balance sanity: > 1T units of a token = airdrop dust
            # - Symbol sanity: very long or unusual symbols = spam
            if price <= 0 or mcap <= 0:
                continue
            if mcap < 5_000_000:
                continue
            if usd_value > 50 and (usd_value / mcap) > 0.01:
                continue
            if bal > 1e12:
                continue
            sym = (tok.get("symbol") or "").strip()
            if len(sym) > 10 or len(sym) < 2:
                continue
            if usd_value < 5:
                continue

            result["tokens"].append({
                "chain": chain,
                "symbol": (tok.get("symbol") or "?")[:12],
                "name": (tok.get("name") or "Unknown")[:40],
                "contract": tok.get("address", ""),
                "balance": round(bal, 6),
                "price_usd": round(price, 6),
                "usd_value": round(usd_value, 2),
                "market_cap_usd": mcap,
            })
    except Exception as e:
        if not result["error"]:
            result["error"] = f"tokens: {e}"

    return result


def fetch_wallet_portfolio(address: str) -> dict:
    """Fetch full EVM multi-chain portfolio (ETH + Base + Arbitrum + Optimism + Polygon)."""
    # Native prices (1 call for all)
    native_prices = {"ethereum": 2400, "matic-network": 0.5}
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": "ethereum,matic-network", "vs_currencies": "usd"}, timeout=6)
        j = r.json()
        for k in j:
            native_prices[k] = j[k].get("usd", native_prices.get(k, 0))
    except Exception:
        pass

    # Fetch all chains in parallel via threads
    from concurrent.futures import ThreadPoolExecutor
    chain_results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_fetch_chain, address, name, cfg, native_prices): name
                for name, cfg in CHAINS.items()}
        for fut, name in futs.items():
            try:
                chain_results[name] = fut.result(timeout=15)
            except Exception as e:
                chain_results[name] = {"chain": name, "error": str(e), "native_balance": 0, "native_usd": 0, "tokens": []}

    # Aggregate
    all_tokens = []
    natives = []
    total_usd = 0
    chains_active = []

    for name, r in chain_results.items():
        if r.get("native_balance", 0) > 0:
            natives.append({
                "chain": name,
                "symbol": r.get("native_symbol", "ETH"),
                "balance": r["native_balance"],
                "usd_value": r["native_usd"],
            })
            total_usd += r["native_usd"]
        all_tokens.extend(r.get("tokens", []))
        total_usd += sum(t["usd_value"] for t in r.get("tokens", []))
        if r.get("native_balance", 0) > 0 or r.get("tokens"):
            chains_active.append(name)

    all_tokens.sort(key=lambda h: h["usd_value"], reverse=True)
    all_tokens = all_tokens[:20]  # top 20 across all chains

    return {
        "address": address,
        "chains_scanned": list(CHAINS.keys()),
        "chains_with_assets": chains_active,
        "total_portfolio_usd": round(total_usd, 2),
        "native_balances": natives,
        "token_count": len(all_tokens),
        "tokens": all_tokens,
    }


def get_eth_price_usd() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": "ethereum", "vs_currencies": "usd"}, timeout=5)
        return r.json().get("ethereum", {}).get("usd", 2400)
    except:
        return 2400

# ── Tool execution ────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict):
    """Returns (result_str, chart_data_or_None)"""
    chart_data = None
    try:
        # ── 1. Crypto price ──────────────────────────────────────────────────
        if name == "get_crypto_price":
            coin = args.get("coin", "bitcoin")
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": coin, "vs_currencies": "usd",
                                     "include_24hr_change": "true", "include_market_cap": "true"},
                             timeout=7)
            data = r.json()
            if coin in data:
                d = data[coin]
                return json.dumps({
                    "coin": coin,
                    "price_usd": d.get("usd", 0),
                    "change_24h_pct": round(d.get("usd_24h_change", 0), 2),
                    "market_cap_usd": d.get("usd_market_cap", 0)
                }), None
            return json.dumps({"error": f"Coin '{coin}' not found"}), None

        # ── 2. Top coins ─────────────────────────────────────────────────────
        elif name == "get_top_coins":
            limit = min(int(args.get("limit", 5)), 10)
            r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                             params={"vs_currency": "usd", "order": "market_cap_desc",
                                     "per_page": limit, "page": 1, "sparkline": "false"},
                             timeout=7)
            return json.dumps({"top_coins": [
                {"rank": i + 1, "name": c["name"], "symbol": c["symbol"].upper(),
                 "price_usd": c["current_price"],
                 "change_24h_pct": round(c.get("price_change_percentage_24h") or 0, 2),
                 "market_cap_usd": c["market_cap"]}
                for i, c in enumerate(r.json())
            ]}), None

        # ── 3. DeFi yields ───────────────────────────────────────────────────
        elif name == "get_defi_yields":
            protocol = args.get("protocol", "aave").lower()
            db = {
                "aave": {"protocol": "Aave V3", "network": "Ethereum + L2", "pools": [
                    {"asset": "USDC", "supply_apy": 4.21, "borrow_apy": 5.83, "tvl": "$2.1B"},
                    {"asset": "ETH",  "supply_apy": 2.84, "borrow_apy": 3.92, "tvl": "$1.8B"},
                    {"asset": "USDT", "supply_apy": 3.95, "borrow_apy": 5.45, "tvl": "$900M"},
                    {"asset": "WBTC", "supply_apy": 0.45, "borrow_apy": 1.23, "tvl": "$600M"},
                ]},
                "compound": {"protocol": "Compound V3", "network": "Ethereum", "pools": [
                    {"asset": "USDC", "supply_apy": 3.82, "borrow_apy": 5.21, "tvl": "$800M"},
                    {"asset": "ETH",  "supply_apy": 2.15, "borrow_apy": 3.10, "tvl": "$600M"},
                ]},
                "uniswap": {"protocol": "Uniswap V3", "network": "Ethereum + L2", "pools": [
                    {"asset": "ETH/USDC 0.05%",  "fee_apy": 12.5, "tvl": "$400M"},
                    {"asset": "WBTC/ETH 0.3%",   "fee_apy": 8.2,  "tvl": "$250M"},
                    {"asset": "ETH/USDT 0.05%",  "fee_apy": 9.8,  "tvl": "$200M"},
                ]}
            }
            if protocol in db:
                return json.dumps(db[protocol]), None
            return json.dumps({"error": f"Unknown protocol '{protocol}'"}), None

        # ── 4. Price chart ───────────────────────────────────────────────────
        elif name == "get_price_chart":
            coin = args.get("coin", "bitcoin")
            days = min(int(args.get("days", 7)), 30)
            r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                params={"vs_currency": "usd", "days": days},
                timeout=10
            )
            data = r.json()
            prices = data.get("prices", [])
            # Downsample to ~60 points for chart
            step = max(1, len(prices) // 60)
            sampled = prices[::step][-60:]
            current = sampled[-1][1] if sampled else 0
            start   = sampled[0][1]  if sampled else 0
            change  = ((current - start) / start * 100) if start else 0

            chart_data = {
                "type": "chart_data",
                "coin": coin,
                "days": days,
                "prices": sampled,
                "change_pct": round(change, 2)
            }
            return json.dumps({
                "coin": coin, "days": days,
                "current_price": round(current, 4),
                "period_change_pct": round(change, 2),
                "data_points": len(sampled)
            }), chart_data

        # ── 5. Gas fees ──────────────────────────────────────────────────────
        elif name == "get_gas_fees":
            eth_usd = get_eth_price_usd()
            results = {}
            for chain in ["ethereum", "base", "polygon", "arbitrum"]:
                resp = rpc_call(chain, "eth_gasPrice", [])
                if "result" in resp:
                    gwei = int(resp["result"], 16) / 1e9
                    usd  = (21000 * gwei * 1e-9) * eth_usd
                    results[chain] = {"gwei": round(gwei, 2), "transfer_usd": round(usd, 4)}
                else:
                    # Fallback estimates
                    fallbacks = {"ethereum": 15, "base": 0.05, "polygon": 30, "arbitrum": 0.1}
                    gwei = fallbacks.get(chain, 5)
                    results[chain] = {"gwei": gwei, "transfer_usd": round((21000 * gwei * 1e-9) * eth_usd, 4), "estimated": True}
            return json.dumps({"gas_fees": results, "eth_price_usd": eth_usd}), None

        # ── 6. Wallet checker ────────────────────────────────────────────────
        elif name == "check_wallet":
            address = args.get("address", "")
            chain   = args.get("chain", "ethereum").lower()
            if not address.startswith("0x") or len(address) < 20:
                return json.dumps({"error": "Invalid address format"}), None

            resp = rpc_call(chain, "eth_getBalance", [address, "latest"])
            if "result" not in resp:
                return json.dumps({"error": "RPC call failed", "details": str(resp)}), None

            balance_wei = int(resp["result"], 16)
            balance_eth = balance_wei / 1e18
            eth_usd     = get_eth_price_usd()
            usd_value   = balance_eth * eth_usd
            return json.dumps({
                "address": address,
                "chain": chain,
                "balance_eth": round(balance_eth, 6),
                "usd_value": round(usd_value, 2),
                "eth_price_usd": eth_usd
            }), None

        # ── 7. Market sentiment ──────────────────────────────────────────────
        elif name == "get_market_sentiment":
            # Fear & Greed
            fng_resp = requests.get("https://api.alternative.me/fng/", timeout=5).json()
            fng = fng_resp.get("data", [{}])[0]
            score = int(fng.get("value", 50))
            label = fng.get("value_classification", "Neutral")

            # BTC dominance
            global_resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=5).json()
            btc_dom = global_resp.get("data", {}).get("market_cap_percentage", {}).get("btc", 0)
            total_mcap = global_resp.get("data", {}).get("total_market_cap", {}).get("usd", 0)

            return json.dumps({
                "fear_greed_score": score,
                "fear_greed_label": label,
                "market_signal": "EXTREME GREED — consider taking profits" if score > 75
                                 else "GREED — market bullish but cautious" if score > 55
                                 else "FEAR — potential buying opportunity" if score < 35
                                 else "EXTREME FEAR — high risk, high reward" if score < 20
                                 else "NEUTRAL — market undecided",
                "btc_dominance_pct": round(btc_dom, 2),
                "total_market_cap_usd": total_mcap
            }), None

        # ── 8. Crypto news ────────────────────────────────────────────────────
        elif name == "get_crypto_news":
            source = args.get("source", "cointelegraph").lower()
            limit  = min(int(args.get("limit", 5)), 6)
            articles = fetch_rss_news(source, limit)
            return json.dumps({"source": source, "articles": articles}), None

        # ── 9b. Wallet portfolio analyzer ─────────────────────────────────────
        elif name == "get_wallet_portfolio":
            address = args.get("address", "").strip()
            if not address.startswith("0x") or len(address) != 42:
                return json.dumps({"error": "Invalid Ethereum address"}), None
            data = fetch_wallet_portfolio(address)
            return json.dumps(data), None

        # ── 10. Set price alert ──────────────────────────────────────────────
        elif name == "set_price_alert":
            coin         = args.get("coin", "bitcoin")
            condition    = args.get("condition", ">")
            target_price = float(args.get("target_price", 0))
            alert_id     = str(uuid.uuid4())[:8]

            PRICE_ALERTS.append({
                "id": alert_id,
                "coin": coin,
                "condition": condition,
                "target_price": target_price,
                "created_at": time.time(),
                "triggered": False
            })
            return json.dumps({
                "status": "created",
                "alert_id": alert_id,
                "message": f"Alert set: notify when {coin} {condition} ${target_price:,.2f}"
            }), None

    except Exception as e:
        return json.dumps({"error": str(e)}), None

    return json.dumps({"error": f"Unknown tool: {name}"}), None


# ── API Routes ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    user_query: str


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "has_og_key": bool(os.environ.get("OG_PRIVATE_KEY"))}


@app.get("/api/news")
async def api_news(source: str = "cointelegraph", limit: int = 5):
    """Direct news endpoint for the sidebar"""
    try:
        articles = await asyncio.to_thread(fetch_rss_news, source, limit)
        return {"articles": articles, "source": source}
    except Exception as e:
        return {"articles": [], "error": str(e)}


@app.get("/api/global")
async def api_global():
    """Global market stats: Fear&Greed, BTC dominance, total market cap"""
    def _fetch():
        out = {"fear_greed": None, "fear_greed_label": "", "btc_dominance": 0, "total_mcap": 0, "mcap_change_24h": 0}
        try:
            fng = requests.get("https://api.alternative.me/fng/", timeout=5).json()
            d = fng.get("data", [{}])[0]
            out["fear_greed"] = int(d.get("value", 50))
            out["fear_greed_label"] = d.get("value_classification", "Neutral")
        except Exception:
            pass
        try:
            g = requests.get("https://api.coingecko.com/api/v3/global", timeout=5).json().get("data", {})
            out["btc_dominance"] = round(g.get("market_cap_percentage", {}).get("btc", 0), 2)
            out["total_mcap"] = g.get("total_market_cap", {}).get("usd", 0)
            out["mcap_change_24h"] = round(g.get("market_cap_change_percentage_24h_usd", 0), 2)
        except Exception:
            pass
        return out
    return await asyncio.to_thread(_fetch)


@app.get("/api/gas")
async def api_gas():
    """Gas fees across ETH, Base, Polygon, Arbitrum"""
    def _fetch():
        eth_usd = get_eth_price_usd()
        results = {}
        for chain in ["ethereum", "base", "polygon", "arbitrum"]:
            resp = rpc_call(chain, "eth_gasPrice", [])
            if "result" in resp:
                gwei = int(resp["result"], 16) / 1e9
                results[chain] = {"gwei": round(gwei, 2), "transfer_usd": round((21000 * gwei * 1e-9) * eth_usd, 4)}
            else:
                fb = {"ethereum": 15, "base": 0.05, "polygon": 30, "arbitrum": 0.1}.get(chain, 5)
                results[chain] = {"gwei": fb, "transfer_usd": round((21000 * fb * 1e-9) * eth_usd, 4), "estimated": True}
        return {"gas": results, "eth_price_usd": eth_usd}
    return await asyncio.to_thread(_fetch)


@app.get("/api/wallet")
async def api_wallet(address: str, chain: str = "ethereum"):
    """Check wallet balance directly"""
    def _fetch():
        if not address.startswith("0x") or len(address) < 20:
            return {"error": "Invalid address"}
        resp = rpc_call(chain, "eth_getBalance", [address, "latest"])
        if "result" not in resp:
            return {"error": "RPC failed"}
        wei = int(resp["result"], 16)
        eth = wei / 1e18
        eth_usd = get_eth_price_usd()
        return {
            "address": address, "chain": chain,
            "balance_eth": round(eth, 6),
            "usd_value": round(eth * eth_usd, 2),
            "eth_price_usd": eth_usd
        }
    return await asyncio.to_thread(_fetch)


@app.get("/api/alerts")
async def api_list_alerts():
    return {"alerts": [a for a in PRICE_ALERTS if not a["triggered"]]}


@app.delete("/api/alerts/{alert_id}")
async def api_delete_alert(alert_id: str):
    global PRICE_ALERTS
    PRICE_ALERTS = [a for a in PRICE_ALERTS if a["id"] != alert_id]
    return {"status": "deleted"}


@app.get("/api/alerts/check")
async def api_check_alerts():
    """Check if any price alerts have been triggered"""
    active = [a for a in PRICE_ALERTS if not a["triggered"]]
    if not active:
        return {"triggered": []}

    # Get current prices
    coins = list(set(a["coin"] for a in active))
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": ",".join(coins), "vs_currencies": "usd"}, timeout=5)
        prices = {k: v.get("usd", 0) for k, v in r.json().items()}
    except:
        return {"triggered": []}

    triggered = []
    for alert in active:
        current = prices.get(alert["coin"], 0)
        if not current:
            continue
        hit = (alert["condition"] == ">" and current > alert["target_price"]) or \
              (alert["condition"] == "<" and current < alert["target_price"])
        if hit:
            alert["triggered"] = True
            triggered.append({**alert, "current_price": current})

    return {"triggered": triggered}


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Main streaming chat endpoint with tool calling"""

    async def generate():
        try:
            llm = get_llm()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages

        try:
            # Step 1: tool detection
            result1 = await llm.chat(
                model=og.TEE_LLM.GEMINI_2_5_FLASH,
                messages=full_messages,
                tools=TOOLS,
                max_tokens=600,
                temperature=0.3,
                stream=False,
            )

            payment_hash = result1.payment_hash or ""
            output1      = result1.chat_output or {}
            tool_calls   = output1.get("tool_calls")
            full_response = ""

            if tool_calls:
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                yield f"data: {json.dumps({'type': 'tools_used', 'tools': tool_names})}\n\n"

                tool_msgs   = []
                chart_events = []

                for tc in tool_calls:
                    t_name   = tc["function"]["name"]
                    t_args   = json.loads(tc["function"]["arguments"])
                    t_result, chart_data = execute_tool(t_name, t_args)

                    if chart_data:
                        chart_events.append(chart_data)

                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": t_result
                    })

                # Send chart data events to frontend first
                for evt in chart_events:
                    yield f"data: {json.dumps(evt)}\n\n"

                # Step 2: streaming response
                msgs_v2 = full_messages + [
                    {"role": "assistant", "content": None, "tool_calls": tool_calls}
                ] + tool_msgs

                stream = await llm.chat(
                    model=og.TEE_LLM.GEMINI_2_5_FLASH,
                    messages=msgs_v2,
                    max_tokens=900,
                    temperature=0.3,
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response += text
                        yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

            else:
                content = output1.get("content", "")
                full_response = content
                for i, word in enumerate(content.split(" ")):
                    chunk_text = word + (" " if i < len(content.split(" ")) - 1 else "")
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk_text})}\n\n"
                    await asyncio.sleep(0.018)

            yield f"data: {json.dumps({'type': 'done', 'payment_hash': payment_hash})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Static ────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("public/index.html")

app.mount("/", StaticFiles(directory="public"), name="static")
