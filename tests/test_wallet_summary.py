import json
import tempfile
import unittest
from pathlib import Path

from informed_flow.db import Database
from informed_flow.wallet_summary import WalletSummary, compact_wallet_history
from tests.test_service import trade


class SummaryTests(unittest.TestCase):
    def test_detailed_sample_and_date_bounds(self):
        summary = WalletSummary(100, 1000000)
        summary.add(trade('one', timestamp=200, size='400', price='.25'))
        second = trade('two', timestamp=999999, condition='second', size='800', price='.5')
        second['side'] = 'SELL'
        summary.add(second)
        summary.add(trade('future', timestamp=1000001))
        summary.add({'bad': True})
        values = summary.result()
        self.assertEqual(values['observed_trade_count'], 2)
        self.assertEqual(values['total_notional_microusd'], 500000000)
        self.assertEqual(values['median_notional_microusd'], 250000000)
        self.assertEqual(values['side_counts'], {'BUY': 1, 'SELL': 1})
        self.assertEqual(values['last_7d_trade_count'], 1)
        self.assertEqual(values['largest_market_volume_share_ppm'], 800000)
        self.assertEqual(values['outside_window_rows'], 1)
        self.assertEqual(values['malformed_rows'], 1)

    def test_cleanup_summarizes_raw_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory)/'test.sqlite3')
            db.initialize()
            with db.connection:
                db.connection.execute("INSERT INTO wallet_history_fetches VALUES (1,'0xwallet',1,2000000,'complete',1,2,NULL)")
                db.connection.execute('INSERT INTO wallet_history_raw VALUES (1,0,?)', (json.dumps(trade(timestamp=1000000)),))
            self.assertEqual(compact_wallet_history(db), (1, 1))
            self.assertEqual(db.row('SELECT COUNT(*) FROM wallet_history_raw')[0], 0)
            self.assertEqual(json.loads(db.row('SELECT summary_json FROM wallet_history_summaries')[0])['observed_trade_count'], 1)
            self.assertEqual(compact_wallet_history(db), (0, 0))
            self.assertEqual(db.row('PRAGMA integrity_check')[0], 'ok')
            db.close()
