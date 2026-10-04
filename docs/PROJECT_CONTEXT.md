# Informed-flow detector: project context

The user wants to explore and then build a read-only research system that
estimates whether a Polymarket trade is informed (often discussed as
"insider-like") and whether following it would have been profitable. Do not
place trades. Do not add wallet-linking, reinforcement learning, or a trading
bot unless the user asks.

This is not legal or financial advice. The user is not trying to commit insider
trading. The idea is to read public trades and estimate which ones look
informed.

## Intended system

An individual-scale system that:

1. Watches public Polymarket trades.
2. Collects features about the trade, the wallet's Polymarket history, and the
   market.
3. Later labels each trade with what would have happened if one bought after a
   realistic delay.
4. Trains a simple supervised model that outputs a probability or expected
   profit, not a bare buy/don't-buy decision.
5. Applies a separate paper-trading rule only when that probability beats the
   executable price plus fees and a margin.

## Scope decisions and history

- The initial request was to explore routes, not scaffold a repo. A read-only
  arbitrage scanner (`polyarb`) was built anyway. Do not expand it unless asked.
- The user wants implementation details and, when they say so, code for the
  informed-flow detector. Confirm before large new builds.
- The local checkout is `/Users/matthewhu/Projects/inside-trader-detector`.
  Cloud agents cannot see that path. Work in this repo and push; they pull.
- The prior private remote was
  `https://cursor.com/codebase/matthew-hu-dev/inside-trader-detector`. The prior
  environment had Origin credentials but not GitHub credentials, so the user
  needed to push to GitHub themselves.
- Pure arbitrage is not the path. A scan of roughly 400 neg-risk events found
  one fully covered basket: about $1.39 profit on $329 locked for about 29 days
  (roughly 5% annualized). Large apparent arbitrages were usually false because
  untradable placeholder or “Other” outcomes could still win.
- Prediction is more realistic, particularly in small markets tied to public
  data. The user then focused on following informed flow.

Do not detect “insiders” as a legal category. Predict whether a trade is
informed: did buying after a delay beat the price? A skilled whale and a leaker
both count if following them makes money; a rich gambler does not.

Whales and insider-style bets often differ, but these are features rather than
separate classifiers:

- Whale: old wallet, many markets, large size normal for that wallet, mixed
  prices, continued trading, and results broadly calibrated to market prices.
- Insider-style: new or unused wallet, one market, size huge relative to its
  history, often a cheap outcome shortly before resolution, willingness to pay
  up for fills, followed by inactivity and performance above the entry price.

Market category should be a score, not a hard filter. Suggested priors:

- High: awards, technology/company announcements, appointments, courts,
  geopolitics, and speech “mentions.”
- Medium: political decisions and central banks.
- Low: elections, macro data, crypto prices, and weather.
- Sports is low but nonzero and very noisy.

Eventually measure pre-announcement price drift by category instead of relying
on these guesses.

## Modeling and validation

Do not use reinforcement learning for the buy decision. Replaying past events,
simulating a purchase at the then-price, and learning from profit is useful, but
it is supervised learning: there is one decision and the outcome of both buy and
skip is present in the data. RL would rediscover the same result and risk
overfitting finite history.

- Begin with a hand-written score and no training.
- After enough labels, use logistic regression, then gradient-boosted trees such
  as LightGBM or XGBoost.
- Target the probability that the outcome occurs, or expected profit at the
  executable price after a delay (15 minutes is the working assumption), net of
  fees.
- Keep the action rule separate: act on paper only when predicted probability is
  greater than executable price plus fees plus a safety margin.
- If sizing is explored later, use fractional Kelly with hard per-trade and
  per-day caps. Do not reduce model output to buy/don't-buy.
- Train on older periods and test on newer periods. Never random-split.
- Compute features as of the trade timestamp; never leak later wallet statistics.
- Prevent a few lucky longshots from dominating results.
- The model must outperform market price itself.

RL may be relevant later for genuinely sequential choices such as selling,
budget allocation across open positions, or splitting an order. It is not part
of the current plan.

## Collection principles

Collect broadly. Scoring rules must not decide what is retained: rules can be
changed and rerun, while missing raw data cannot be recovered.

- Log every trade above a low cash threshold (working value: $100), preserving
  the raw API response.
- Optionally side-table five-minute crypto “Up or Down” markets (`updown` in the
  slug); they dominate volume and are mostly noise.
- Enrich trades above roughly $1,000 outside sports and short-crypto, plus a
  random sample of smaller trades so normal behavior is represented.
- Order-book snapshots are the only data that cannot be reconstructed later.
  Save the book, or best bid/ask and depth, when a large or high-scoring trade is
  flagged.
- Attach scores only after storage and make them recomputable.

## Phases

1. **Watch and log.** Poll large trades, store raw rows, enrich a broad subset,
   attach a simple score. No money.
2. **Label.** Record prices at +15 minutes, +1 hour, and +24 hours, then the final
   result at resolution. Report win rate and profit against the price that was
   actually available, grouped by score bucket.
3. **Wallet linking.** Only if phase 2 shows a signal worth improving. Potential
   signals include funding source, shared signer, and clusters of new wallets on
   the same outcome. A cheap phase-1 feature may count new wallets buying the
   same outcome within an hour using already-collected data.
4. **Supervised model.** Train on the log and a historical backfill so useful
   labels do not take months to accumulate.
5. **Paper trading, then possibly tiny real size.** Only if out-of-sample results
   beat market prices, and only if the user explicitly asks to expand scope.
   Prefer Kalshi or Polymarket US for a US user when the same event is listed and
   resolution rules match.

## Wallet identity and linking

- Track the Polymarket trading wallet (`proxyWallet`), not username; usernames
  change.
- Phase 1 uses Polymarket Data API history only: wallet age, trade count, markets
  traded, closed-position wins/losses, and current bet versus usual size.
- Do not pull chain-level history, Polygonscan data, funding sources, or signers
  in phase 1. Many email-signup users have wallets with no outside history.
- Wallet linking is probabilistic. Shared funding, cash-out destination, timing,
  sizes, and signers can help, while mixers, bridges, and exchange hot wallets
  confound it.

## Venue and legal boundaries

| Venue | Access and identity | Public trade data |
| --- | --- | --- |
| Polymarket Global | Non-US; geoblocked for US persons; crypto wallet, no KYC | Public on Polygon; each trade has a wallet; collateral is pUSD |
| Polymarket US | US, CFTC-regulated; KYC in its app | Private accounts; separate API and SDK; not the same code as Global |
| Kalshi | US, CFTC-regulated; KYC | Public trades are anonymous (price, size, time, side); demo API available |

Do not suggest VPN or geoblock workarounds. Polymarket matches buyers and sellers
and charges fees; informed traders take money from market makers and other users,
which can cause quotes to widen before a follower can enter.

A possible later US-friendly design is to watch public Global wallets, then
observe whether Kalshi or Polymarket US has an equivalent event that has not yet
repriced. Resolution rules must match. Do not build execution without an
explicit request.

Public-data copy trading is not the same as trading on stolen information, but
the user should verify their own legal and regulatory situation. Do not help
obtain nonpublic information or evade platform rules.

## Polymarket APIs

Reads are free and keyless. Limits are per IP over rolling ten-second windows and
are enforced by Cloudflare. Excess traffic is throttled or queued. Verify current
documentation before implementation:
<https://docs.polymarket.com/quickstart/introduction/rate-limits>.

| API | Base URL | Role |
| --- | --- | --- |
| Gamma | `https://gamma-api.polymarket.com` | Market metadata: question, tags, end date, liquidity, resolution |
| Data | `https://data-api.polymarket.com` | Trades, wallet activity, positions; primary detector source |
| CLOB | `https://clob.polymarket.com` | Live books and prices; anonymous reads; trading requires keys |

Verified Data `/trades` parameters include:

- `filterType=CASH&filterAmount=5000` for a minimum dollar size
- `side=BUY`
- `takerOnly=true` for aggressive fills and urgency
- `market=<conditionId>` (comma-separated values accepted)
- `user=<wallet>`
- `limit` and `offset`

There is no category filter on `/trades`; identify sports after looking up the
market. Trade responses include side, outcome, price, size, Unix timestamp,
transaction hash, title, slug, `conditionId`, outcome token `asset`, and trader
`proxyWallet`. `eventSlug` may be empty and `outcomeIndex` may be `999`; do not
trust them. Counterparty, wallet age, category, fee, and funding source are not
included.

Useful Data endpoints:

- `/activity?user=&sortBy=TIMESTAMP&sortDirection=ASC&limit=1` — first action and
  therefore approximate wallet age
- `/activity?user=&type=TRADE&start=&end=` — windowed history, Unix timestamps
- `/positions?user=` — open positions, average price, cash P&L
- `/closed-positions` — settled record
- `/trades?user=` — wallet trades

Useful Gamma behavior:

- `/markets?condition_ids=<id>&include_tag=true` returns tags, end dates,
  liquidity, and outcome prices.
- A resolved market with `outcomePrices` such as `["1", "0"]` means the first
  outcome won.
- `/events?tag_slug=politics&closed=false` is supported.
- Working market filters included `closed`, `liquidity_num_min`, `end_date_max`,
  and `order=volume24hr&ascending=false`.
- Sports metadata may include `sportsMarketType` and `gameStartTime`.
- Short crypto markets have not always resolved cleanly via slug lookup; filter
  `updown` by slug or title.
- Cache market lookups because many trades share a market.

CLOB order books:

- `GET /book?token_id=` and `POST /books` with
  `[{"token_id": "..."}]`
- Bids and asks are `{price, size}`. Best bid is the highest buy; best ask is the
  lowest sell. Displayed price is usually midpoint or last trade.
- Books contain no wallet identifiers. Use them only for executable price and
  depth snapshots.

Previously documented rolling rate limits were Data general 1,000; `/trades`
200; `/positions` and `/closed-positions` 150; Data v2 `/trades` 300,
`/activity` 200, `/prices-history` 200; Gamma general 4,000, `/events` 500,
`/markets` 300; CLOB `/book` 1,500, `/books` 500, `/price` 1,500. Check current
docs before relying on these. Minute polling plus a few enrichments is well below
the limits; historical backfills should pause between batches.

## Fees and data durability

A previously documented taker-fee formula was:

`shares × feeRate × p × (1 − p)`

Makers pay no platform fee. Previously observed category rates were crypto 0.07;
sports/economics/culture/weather/general 0.05; finance/politics/mentions/tech
0.04; geopolitics 0. Fees are symmetric around $0.50 and round to five decimals
with a minimum of 0.00001 USDC. Rates vary, so verify each market's
`feeSchedule`; Gamma objects may contain `feesEnabled` and `feeSchedule`.

- Settled trades, balances, and funding live on Polygon and can be reconstructed
  via sources such as Dune, Polymarket subgraphs, or Polygonscan.
- Gamma, Data API, and the live order book are operated services and can change
  or disappear.
- Historical order-book state cannot be reconstructed from final chain trades;
  snapshots are therefore important.
- Store local labels and snapshots in SQLite initially.

## Phase 1 implementation sketch

Read-only, no keys, no orders:

1. Poll `GET /trades?side=BUY&takerOnly=true&filterType=CASH&filterAmount=100&limit=...`
   every minute. Use offsets or a seen-transaction-hash set and overlapping polls.
2. Store the raw trade JSON.
3. Skip or side-table `updown` slugs.
4. For trades above the enrichment threshold and a random sample below it, do
   one cached Gamma lookup per new `conditionId`, then compute wallet age, trade
   count, distinct markets, and closed P&L as of that timestamp.
5. Save a CLOB book snapshot for large trades.
6. Compute a replaceable score. Candidate points: wallet younger than seven
   days, one of its first few trades, notional at least $5,000, price below
   $0.25, resolution within 30 days, high-vulnerability tags, and several new
   wallets on the same outcome within one hour. Never drop low-score rows.
7. Use SQLite tables resembling `trades`, `markets`, `wallet_features`,
   `book_snapshots`, and `labels`.
8. Add a report command for counts, score histograms, and later win rate and
   profit against entry price.

Suggested stack: Python 3.10+, standard library or `httpx`, and SQLite. If the
existing `polyarb` code uses `urllib`, match it unless there is a concrete reason
not to. No new service, authentication, or database server.

## Labeling and backtesting

For every stored trade, after each delay has passed, record:

- The price available at +15 minutes, +1 hour, and +24 hours. Prefer actual CLOB
  history or a price-history endpoint. If only a later trade print is available,
  record it and mark it approximate.
- At resolution, whether the selected outcome won, based on Gamma outcome prices.
- Profit per share at delayed ask: `1 - ask - fee` if it won, otherwise
  `-ask - fee`.

Report mean profit and win rate versus delayed price for every score bucket. A
flag at $0.20 that wins 20% of the time has no edge.

Run the same code path on historical trades in resolved markets, with all
features calculated as of each historical trade timestamp. This yields many
labels without waiting months. Respect rate limits. Never use a wallet's current
win rate as a historical feature.

## Repository notes and current defaults

Earlier context referred to these paths, though a fresh checkout may not contain
them yet:

- `docs/EXPLORATION.md` — stocks versus Polymarket routes; background only
- `polyarb/` — read-only neg-risk basket scanner, not the informed-flow detector
- `README.md` — scanner instructions
- `tests/` — arbitrage math tests
- `.venv/` — local and gitignored

Nothing for the informed-flow detector existed when this context was captured.

Working defaults:

- Store trades at a $100 cash floor.
- Enrich at $1,000 plus a random lower-value sample.
- Label at 15 minutes, 1 hour, and 24 hours.
- Start with a hand score; wait for labels before fitting a model.
- Use Polymarket data only; no chain lookups in phase 1.
- No trading, reinforcement learning, or GitHub migration from this environment.

If the user says **“start,”** implement phase 1 as described and keep it
read-only.
