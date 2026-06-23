# tools/web/ — Web Data and External API Tools

Tools that fetch external data from the web to supplement OHLCV-based analysis. These provide the agent with sentiment, macro context, insider activity, and earnings data that purely price-based tools cannot capture.

All web tools handle network errors gracefully — they return `{"signal": "HOLD", "strength": 0.0, "error": "..."}` instead of raising exceptions. A failed web fetch should never crash the agent.

---

## Future Tools

### 1. `news_sentiment_fetcher.py` — NewsSentimentFetcherTool

**What it does:** Fetches recent news headlines for the ticker from free sources (Yahoo Finance RSS, Google News RSS, Finviz) and scores sentiment using VADER (rule-based) or FinBERT (transformer-based, optional upgrade path).

**Why it matters:** Sentiment is the fastest-moving signal available. A major negative headline can precede technical breakdown by hours. Time-bucketed scores (`recent_24h_score` vs `recent_week_score`) let the LLM detect sentiment acceleration or reversal.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_articles` | 20 | Maximum articles to fetch |
| `sources` | all | Which sources to query |

**Key output fields:**
- `signal`, `strength`
- `sentiment_score` — aggregate score (-1 to +1)
- `sentiment_label` — POSITIVE / NEUTRAL / NEGATIVE
- `articles_analysed` — count
- `headlines` — list of `{title, score, source, age_hours}`
- `recent_24h_score`, `recent_week_score`
- `high_impact_count`
- `sources_used`

**Signal convention:** Score > 0.25 = BUY. Score < -0.25 = SELL. Between = HOLD.

**Dependencies:** `requests`, `feedparser`, `vaderSentiment`, `beautifulsoup4` (already installed)

---

### 2. `economic_calendar_fetcher.py` — EconomicCalendarFetcherTool

**What it does:** Retrieves upcoming high-impact macroeconomic events (FOMC decisions, CPI, GDP, NFP, jobs reports) from a free API or by scraping Investing.com. Flags events within the configurable lookahead window.

**Why it matters:** The agent should not initiate aggressive directional bets before major macro events. This is the "do not trade" signal that no amount of technical or ML analysis can provide. An FOMC decision in 2 days overrides a BUY from every other tool.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookahead_days` | 7 | Days to look ahead |
| `min_impact` | HIGH | Minimum impact level to report |

**Key output fields:**
- `signal`, `strength`
- `high_impact_events_imminent` — bool
- `events` — list of `{name, date, impact, days_away}`
- `recommendation` — pre-reasoned text for the LLM
- `next_event_days` — days until the nearest high-impact event

**Signal convention:** High-impact event within 2 days = HOLD with high strength. Within 7 days = HOLD with moderate strength. No events = HOLD with zero strength (pass-through).

**Dependencies:** `requests`, `beautifulsoup4`

---

### 3. `earnings_calendar_fetcher.py` — EarningsCalendarFetcherTool

**What it does:** Fetches the next and most recent earnings date for the ticker from `yfinance` (already a dependency), along with historical EPS surprise data and average post-earnings price moves.

**Why it matters:** Earnings are the single most predictable discrete risk event for individual equities. `yfinance.Ticker.calendar` already provides this data, making implementation straightforward. `earnings_imminent` (within 14 days) should trigger a HOLD with high strength regardless of other signals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imminent_threshold_days` | 14 | Days threshold for "imminent" flag |

**Key output fields:**
- `signal`, `strength`
- `next_earnings_date`, `days_to_earnings`
- `earnings_imminent` — bool
- `last_eps_surprise_pct`, `last_eps_direction` — BEAT / MISS / INLINE
- `consensus_eps_estimate`
- `historical_post_earnings_moves` — list of recent post-earnings % moves
- `avg_post_earnings_move` — average absolute move
- `earnings_risk_score` — 0 to 1

**Signal convention:** Earnings within `imminent_threshold_days` = HOLD with high strength. Otherwise HOLD with low strength.

**Dependencies:** `yfinance` (already installed)

---

### 4. `sec_filing_parser.py` — SECFilingParserTool

**What it does:** Queries the SEC EDGAR API (`data.sec.gov/submissions/CIK.json` — free, no API key required) for recent 8-K and 10-Q filings. Extracts filing metadata, scans for risk keywords ("material weakness", "going concern", "restatement", "litigation"), and scores filing sentiment.

**Why it matters:** 8-K filings are the gold standard of material event disclosure. A material weakness, major lawsuit, or guidance revision appears in EDGAR before it is widely reported by news outlets. Only applicable to US-listed equities.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filing_types` | 8-K, 10-Q | Filing types to search |
| `lookback_days` | 30 | How far back to search |

**Key output fields:**
- `signal`, `strength`
- `latest_filing_type`, `latest_filing_date`, `days_since_filing`
- `filing_title`
- `risk_keywords_found` — list of flagged terms
- `sentiment_score`, `filing_sentiment` — POSITIVE / NEUTRAL / SLIGHTLY_NEGATIVE / NEGATIVE
- `cik`, `edgar_url`

**Signal convention:** Risk keywords found in recent filing = SELL with moderate strength. Recent positive filing (earnings beat) = BUY with low strength. No recent filings = HOLD.

**Dependencies:** `requests`

---

### 5. `social_sentiment_fetcher.py` — SocialSentimentFetcherTool

**What it does:** Fetches mentions of the ticker from Reddit (`r/stocks`, `r/wallstreetbets`, `r/investing`) using Reddit's public JSON endpoint (no API key required for read-only access to `.json` URLs). Scores sentiment and measures discussion volume.

**Why it matters:** Retail sentiment from Reddit has been shown to precede short-squeeze dynamics and momentum bursts. The `mention_z_score` (current mentions vs 30-day rolling mean) is the key signal: Z > 2 indicates unusual attention and often precedes a volatility event. Weight low normally, weight heavily when Z-score is extreme.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subreddits` | stocks, wallstreetbets, investing | Subreddits to search |
| `lookback_hours` | 24 | How far back to search |

**Key output fields:**
- `signal`, `strength`
- `mention_count_24h`
- `mention_trend` — RISING / STABLE / FALLING
- `sentiment_score`, `sentiment_label`
- `top_posts` — list of `{title, score, sentiment}`
- `subreddit_breakdown` — dict of mention counts per subreddit
- `unusual_activity` — bool
- `mention_z_score`

**Signal convention:** Z-score > 2 and sentiment positive = BUY with moderate strength. Z-score > 2 and sentiment negative = SELL. Normal activity = HOLD with zero strength.

**Dependencies:** `requests`, `vaderSentiment`

---

### 6. `insider_transaction_fetcher.py` — InsiderTransactionFetcherTool

**What it does:** Queries SEC Form 4 data via OpenInsider (public, no API key) or the EDGAR API to retrieve recent insider buying and selling for the ticker.

**Why it matters:** Insider buying — particularly cluster buying (multiple insiders buying in the same window) — is one of the strongest predictive signals in academic finance literature. Insiders cannot trade on MNPI, but their legal purchases reveal private conviction. A cluster buy from C-suite insiders near 52-week lows is a high-quality BUY signal.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_days` | 90 | How far back to search |

**Key output fields:**
- `signal`, `strength`
- `net_insider_sentiment` — BUYING / SELLING / NEUTRAL
- `recent_transactions` — list of `{insider, type, shares, price, date}`
- `buy_value_30d`, `sell_value_30d`
- `net_buy_ratio` — buy_value / (buy_value + sell_value)
- `cluster_buying` — bool (multiple insiders buying within 5 days)
- `signal_strength_reason` — human-readable explanation

**Signal convention:** Cluster buying = BUY with high strength. Net heavy selling = SELL with moderate strength. Mixed = HOLD.

**Dependencies:** `requests`, `beautifulsoup4`

---

### 7. `options_flow_fetcher.py` — OptionsFlowFetcherTool

**What it does:** Fetches publicly available options data from `yfinance` (already a dependency) to compute put/call ratio, implied volatility, IV rank, and max pain price.

**Why it matters:** Options markets are dominated by professional and institutional traders. The put/call ratio is a widely used confirmation/contrarian indicator. IV rank contextualises whether options are cheap or expensive. Max pain (the price at which the most options expire worthless) acts as a gravitational price target near expiration.

**No configurable parameters** (uses the nearest expiration by default).

**Key output fields:**
- `signal`, `strength`
- `put_call_ratio`, `put_call_interpretation` — BULLISH / NEUTRAL / BEARISH
- `atm_iv` — at-the-money implied volatility
- `iv_rank` — where current IV sits vs its 1-year range
- `iv_regime` — LOW / NORMAL / ELEVATED / EXTREME
- `max_pain_price`
- `current_price_vs_max_pain` — distance as fraction
- `unusual_activity` — bool
- `unusual_strike`, `unusual_type` — details of any unusual volume

**Signal convention:** Put/call > 1.2 = SELL (bearish options flow). Put/call < 0.7 = BUY (bullish flow). Between = HOLD. IV rank > 0.80 overrides to HOLD (event risk priced in).

**Dependencies:** `yfinance` (already installed)

---

## Design Notes for All Web Tools

- **Rate limiting:** Add a 1-second delay between HTTP requests to the same domain. Respect `robots.txt` for scraping targets.
- **Caching:** Consider caching responses for 15-60 minutes to avoid redundant fetches during batch analysis of the same ticker.
- **Timeouts:** All HTTP requests should use a 10-second timeout. Web tool latency should not block the analysis pipeline.
- **Fallback:** If a web source is unavailable, return `{"signal": "HOLD", "strength": 0.0, "error": "Source unavailable", "note": "Defaulting to HOLD — web data could not be fetched"}`.
- **Ticker mapping:** Some sources use different ticker formats. Include a helper that converts between formats (e.g., `BHEL.NS` for Yahoo vs `BHEL` for Indian APIs vs `AAPL` for SEC).
