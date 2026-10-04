from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


class APIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class HistoryTruncated(APIError):
    """The requested window exceeded the page budget; yielded rows are a sample."""


@dataclass(frozen=True)
class Page:
    rows: list[dict[str, Any]]
    next_cursor: str | None


class HTTPClient:
    def __init__(self, timeout: float = 20.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries

    def get_json(self, base: str, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(
            {key: value for key, value in (params or {}).items() if value is not None}
        )
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "informed-flow/0.1 read-only"},
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                last_error = APIError(f"GET {url} returned HTTP {exc.code}", status=exc.code, retryable=retryable)
                if not retryable or attempt >= self.retries:
                    raise last_error from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = APIError(f"GET {url} failed: {exc}", retryable=True)
                if attempt >= self.retries:
                    raise last_error from exc
                delay = 2**attempt
            delay = min(delay, 60.0)
            print(f"API retry {attempt + 1}/{self.retries} for {path} in {delay:.1f}s: {last_error}", flush=True)
            time.sleep(delay + random.random() * 0.25)
        raise APIError(str(last_error or "unknown API error"), retryable=True)


class PolymarketAPI:
    DATA_BASE = "https://data-api.polymarket.com"
    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE = "https://clob.polymarket.com"

    def __init__(self, http: HTTPClient | None = None):
        self.http = http or HTTPClient()

    @staticmethod
    def _page(payload: Any) -> Page:
        if not isinstance(payload, Mapping):
            raise APIError("Data API response is not an object")
        rows = payload.get("data", [])
        if isinstance(rows, Mapping):
            rows = rows.get("items", rows.get("rows", []))
        if not isinstance(rows, list):
            raise APIError("Data API response data is not a list")
        pagination = payload.get("pagination") or {}
        cursor = pagination.get("next_cursor") or pagination.get("nextCursor")
        return Page([dict(row) for row in rows if isinstance(row, Mapping)], cursor)

    def trades_page(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
        cursor: str | None = None,
        user: str | None = None,
        limit: int = 500,
        side: str | None = "BUY",
        taker_only: bool | None = True,
        min_cash: int | None = 100,
        condition: str | None = None,
    ) -> Page:
        payload = self.http.get_json(
            self.DATA_BASE,
            "/v2/trades",
            {
                "side": side,
                "taker_only": "true" if taker_only else None,
                "filter_type": "CASH" if min_cash is not None else None,
                "filter_amount": min_cash,
                "start": start,
                "end": end,
                "user": user,
                "limit": limit,
                "cursor": cursor,
                "condition": condition,
            },
        )
        return self._page(payload)

    def iter_wallet_trades(self, user: str, end: int, *, start: int = 1) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        seen: set[str] = set()
        for page_number in range(1, 101):
            print(f"  wallet {user}: history page {page_number}, cutoff={end}", flush=True)
            page = self.trades_page(
                start=start,
                end=end,
                cursor=cursor,
                user=user,
                side=None,
                taker_only=None,
                min_cash=None,
            )
            yield from page.rows
            if not page.next_cursor:
                return
            if page.next_cursor in seen:
                raise APIError("Wallet history repeated a cursor; enrichment remains incomplete")
            seen.add(page.next_cursor)
            cursor = page.next_cursor
        raise HistoryTruncated("Wallet history exceeded 100 pages; recent-window sample is truncated")

    def markets_page(self, cursor: str | None = None, *, closed: bool = False, limit: int = 100) -> Page:
        payload = self.http.get_json(self.GAMMA_BASE, "/markets/keyset", {
            "limit": limit, "after_cursor": cursor, "order": "id", "ascending": "false",
            "include_tag": "true", "closed": "true" if closed else "false",
        })
        if not isinstance(payload, Mapping) or not isinstance(payload.get("markets"), list):
            raise APIError("Gamma keyset market listing has an unexpected shape")
        return Page([dict(row) for row in payload["markets"] if isinstance(row, Mapping)],
                    payload.get("next_cursor"))

    def market(self, condition_id: str) -> dict[str, Any] | None:
        # Gamma defaults to open markets. Explicitly search both cohorts.
        for closed in ("false", "true"):
            payload = self.http.get_json(
                self.GAMMA_BASE,
                "/markets",
                {"condition_ids": condition_id, "include_tag": "true", "limit": 1,
                 "closed": closed},
            )
            rows = payload.get("data") if isinstance(payload, Mapping) else payload
            if isinstance(rows, list) and rows:
                return dict(rows[0])
        return None

    def book(self, asset_id: str) -> dict[str, Any]:
        payload = self.http.get_json(self.CLOB_BASE, "/book", {"token_id": asset_id})
        if not isinstance(payload, Mapping):
            raise APIError("CLOB book response is not an object")
        return dict(payload)
