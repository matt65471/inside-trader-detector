from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Mapping

MICRO = Decimal("1000000")
FEATURE_VERSION = 1
SCORE_VERSION = 1


def first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def decimal_value(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc


def to_millionths(value: Any) -> int:
    return int((decimal_value(value) * MICRO).to_integral_value(rounding=ROUND_HALF_UP))


def notional_microusd(price: Any, size: Any) -> int:
    result = decimal_value(price) * decimal_value(size) * MICRO
    return int(result.to_integral_value(rounding=ROUND_HALF_UP))


def parse_timestamp(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_trade(raw: Mapping[str, Any]) -> dict[str, Any]:
    price = first(raw, "price")
    size = first(raw, "size")
    normalized = {
        "source_trade_id": first(raw, "id", "trade_id", "tradeId"),
        "transaction_hash": str(first(raw, "transaction_hash", "transactionHash", default="")),
        "proxy_wallet": str(first(raw, "proxy_wallet", "proxyWallet", default="")).lower(),
        "condition_id": str(first(raw, "condition_id", "conditionId", default="")).lower(),
        "asset_id": str(first(raw, "asset_id", "token_id", "tokenId", "asset", default="")),
        "side": str(first(raw, "side", default="")).upper(),
        "price": decimal_value(price),
        "size": decimal_value(size),
        "trade_ts": int(first(raw, "timestamp", "trade_ts", default=0)),
        "title": str(first(raw, "title", default="")),
        "slug": str(first(raw, "slug", default="")),
        "event_slug": str(first(raw, "event_slug", "eventSlug", default="")),
        "outcome_name": str(first(raw, "outcome", "outcome_name", default="")),
        "outcome_index": first(raw, "outcome_index", "outcomeIndex"),
    }
    required = ("proxy_wallet", "condition_id", "asset_id", "side", "trade_ts")
    missing = [name for name in required if not normalized[name]]
    if missing:
        raise ValueError(f"trade missing required fields: {', '.join(missing)}")
    normalized["price_ppm"] = to_millionths(price)
    normalized["size_microshares"] = to_millionths(size)
    normalized["notional_microusd"] = notional_microusd(price, size)
    pieces = [
        normalized["transaction_hash"],
        normalized["proxy_wallet"],
        normalized["condition_id"],
        normalized["asset_id"],
        str(normalized["trade_ts"]),
        normalized["side"],
        format(normalized["price"], "f"),
        format(normalized["size"], "f"),
    ]
    normalized["fingerprint"] = hashlib.sha256("\x1f".join(pieces).encode()).hexdigest()
    normalized["trade_key"] = str(normalized["source_trade_id"] or normalized["fingerprint"])
    return normalized


_SLUG_UPDOWN_5M = re.compile(r"(?:^|-)(?:updown|up-or-down)(?:-|).*5m(?:-|$)", re.I)
_CRYPTO_WORDS = re.compile(
    r"\b(?:btc|bitcoin|eth|ethereum|sol|solana|xrp|doge|dogecoin|crypto)\b", re.I
)
_UP_OR_DOWN = re.compile(r"\bup\s+(?:or|/)\s+down\b", re.I)


def classify_five_minute_updown(
    trade: Mapping[str, Any], market: Mapping[str, Any] | None
) -> tuple[bool, str | None]:
    market = market or {}
    slugs = " ".join(
        str(value or "")
        for value in (
            trade.get("slug"),
            trade.get("event_slug"),
            first(market, "slug"),
            first(market, "eventSlug", "event_slug"),
        )
    ).lower()
    compact = slugs.replace("_", "-")
    if "updown-5m" in compact or _SLUG_UPDOWN_5M.search(compact):
        return True, "slug"

    title = " ".join(
        str(value or "") for value in (trade.get("title"), first(market, "question", "title"))
    )
    start = parse_timestamp(first(market, "startDate", "start_date", "startDateIso"))
    end = parse_timestamp(first(market, "endDate", "end_date", "endDateIso"))
    duration = end - start if start is not None and end is not None else None
    title_match = bool(_CRYPTO_WORDS.search(title) and _UP_OR_DOWN.search(title))
    duration_match = duration is not None and 120 <= duration <= 600
    if title_match and duration_match:
        return True, "title_and_duration"
    return False, None


HIGH_CATEGORY_TERMS = {
    "award", "awards", "appointment", "court", "courts", "geopolitics",
    "mention", "mentions", "technology", "tech", "company", "business",
}
MEDIUM_CATEGORY_TERMS = {"politics", "political", "fed", "central-bank", "central bank"}


def category_prior(category: str | None, raw_market: Mapping[str, Any] | None = None) -> str:
    values = [category or ""]
    if raw_market:
        for tag in first(raw_market, "tags", default=[]) or []:
            if isinstance(tag, Mapping):
                values.extend([str(first(tag, "slug", default="")), str(first(tag, "label", default=""))])
    text = " ".join(values).lower()
    if any(term in text for term in HIGH_CATEGORY_TERMS):
        return "high"
    if any(term in text for term in MEDIUM_CATEGORY_TERMS):
        return "medium"
    return "low"


def deterministic_sample(key: str, numerator: int = 1, denominator: int = 4) -> bool:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % denominator
    return bucket < numerator


def median_int(values: Iterable[int]) -> int | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2
