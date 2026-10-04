"""Compact summaries of observed wallet trades; no lifetime claims."""
from __future__ import annotations

from collections import Counter
from statistics import pstdev

from .core import normalize_trade


class WalletSummary:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end
        self.received = self.malformed = self.outside = 0
        self.notionals: list[int] = []
        self.prices: list[int] = []
        self.sides: Counter = Counter()
        self.side_volume: Counter = Counter()
        self.markets: Counter = Counter()
        self.market_volume: Counter = Counter()
        self.assets: Counter = Counter()
        self.days: Counter = Counter()
        self.first = self.last = None
        self.recent_count = self.recent_volume = 0

    def add(self, raw: dict) -> None:
        self.received += 1
        try:
            row = normalize_trade(raw)
        except (ValueError, TypeError):
            self.malformed += 1
            return
        ts = row['trade_ts']
        if not self.start <= ts <= self.end:
            self.outside += 1
            return
        cash = row['notional_microusd']
        self.notionals.append(cash)
        self.prices.append(row['price_ppm'])
        self.sides[row['side']] += 1
        self.side_volume[row['side']] += cash
        self.markets[row['condition_id']] += 1
        self.market_volume[row['condition_id']] += cash
        self.assets[row['asset_id']] += 1
        self.days[ts // 86400] += 1
        self.first = min(ts, self.first) if self.first is not None else ts
        self.last = max(ts, self.last) if self.last is not None else ts
        if ts >= self.end + 1 - 7 * 86400:
            self.recent_count += 1
            self.recent_volume += cash

    def result(self) -> dict:
        values = sorted(self.notionals)
        n = len(values)
        total = sum(values)
        def quantile(fraction):
            if not n:
                return None
            index = (n - 1) * fraction
            lower = int(index)
            upper = min(lower + 1, n - 1)
            return int(values[lower] + (values[upper] - values[lower]) * (index - lower))
        return {
            'summary_version': 1, 'window_start': self.start, 'window_end': self.end,
            'returned_rows': self.received, 'malformed_rows': self.malformed,
            'outside_window_rows': self.outside, 'observed_trade_count': n,
            'distinct_market_count': len(self.markets), 'distinct_outcome_count': len(self.assets),
            'active_days': len(self.days), 'first_trade_in_window': self.first,
            'last_trade_in_window': self.last,
            'seconds_since_last_trade': self.end + 1 - self.last if self.last is not None else None,
            'side_counts': dict(self.sides), 'side_notional_microusd': dict(self.side_volume),
            'total_notional_microusd': total,
            'mean_notional_microusd': total // n if n else None,
            'min_notional_microusd': min(values) if n else None,
            'median_notional_microusd': quantile(.5), 'p25_notional_microusd': quantile(.25),
            'p75_notional_microusd': quantile(.75), 'p90_notional_microusd': quantile(.9),
            'max_notional_microusd': max(values) if n else None,
            'stddev_notional_microusd': round(pstdev(values)) if n else None,
            'mean_price_ppm': sum(self.prices) // n if n else None,
            'cheap_trade_count': sum(p <= 250000 for p in self.prices),
            'last_7d_trade_count': self.recent_count, 'last_7d_notional_microusd': self.recent_volume,
            'largest_market_volume_share_ppm': max(self.market_volume.values()) * 1000000 // total if total else None,
            'top_markets_by_volume': [
                {'condition_id': market, 'notional_microusd': volume, 'trade_count': self.markets[market]}
                for market, volume in self.market_volume.most_common(5)
            ],
        }


def compact_wallet_history(db) -> tuple[int, int]:
    """Summarize existing history before deleting only enrichment raw rows."""
    import json
    from .core import canonical_json
    fetched = db.rows("""SELECT h.* FROM wallet_history_fetches h
        WHERE EXISTS (SELECT 1 FROM wallet_history_raw r WHERE r.fetch_id=h.fetch_id)
        AND NOT EXISTS (SELECT 1 FROM wallet_history_summaries s WHERE s.fetch_id=h.fetch_id)""")
    for index, fetch in enumerate(fetched, 1):
        summary = WalletSummary(fetch['window_start'], fetch['window_end'])
        for row in db.connection.execute("SELECT raw_json FROM wallet_history_raw WHERE fetch_id=? ORDER BY row_number", (fetch['fetch_id'],)):
            summary.add(json.loads(row['raw_json']))
        with db.connection:
            db.connection.execute("INSERT INTO wallet_history_summaries VALUES (?,?)",
                                  (fetch['fetch_id'], canonical_json(summary.result())))
        print(f"Summarized stored history {index}/{len(fetched)}", flush=True)
    missing = db.row("""SELECT COUNT(*) n FROM wallet_history_raw r
        WHERE NOT EXISTS (SELECT 1 FROM wallet_history_summaries s WHERE s.fetch_id=r.fetch_id)""")['n']
    if missing:
        raise RuntimeError('Raw rows lack summaries; cleanup aborted')
    count = db.row('SELECT COUNT(*) n FROM wallet_history_raw')['n']
    with db.connection:
        db.connection.execute('DELETE FROM wallet_history_raw')
        db.connection.execute("UPDATE wallet_snapshots SET missing_reason=REPLACE(missing_reason,'raw_history_fetch_id:','history_fetch_id:')")
    db.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.connection.execute('VACUUM')
    db.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return len(fetched), count
