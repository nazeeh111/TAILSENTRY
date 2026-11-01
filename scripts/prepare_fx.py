#!/usr/bin/env python3
"""Fetch and align public-domain Federal Reserve H.10 FX observations.

No authentication or API key is used. Existing raw files are reused so a
reproduction does not silently change source vintages. To update, select a new
--data-dir. This script prepares data only; it does not fit or select a model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.request import Request, urlopen


SOURCES = {
    "eur": "https://www.federalreserve.gov/releases/h10/hist/dat00_eu.htm",
    "jpy": "https://www.federalreserve.gov/releases/h10/hist/dat00_ja.htm",
    "gbp": "https://www.federalreserve.gov/releases/h10/hist/dat00_uk.htm",
}
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
BEGIN = date(2000, 1, 1)
TRAIN_END = date(2014, 12, 31)
END = date(2025, 12, 31)


class RateTable(HTMLParser):
    """Read dated H.10 table rows without depending on optional HTML libraries."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row = []
        self.cell = None
        self.records = {}

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and len(self.row) >= 2:
            raw_date, raw_value = self.row[:2]
            if not re.fullmatch(r"\d{1,2}-[A-Za-z]{3}-\d{2}", raw_date):
                return
            day, month, year = raw_date.upper().split("-")
            when = date(2000 + int(year), MONTHS[month], int(day))
            if raw_value in {"ND", "N.D.", "NA", ""}:
                value = None
            else:
                value = float(raw_value.replace(",", ""))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid FX observation {raw_date}: {raw_value}")
            if when in self.records and self.records[when] != value:
                raise ValueError(f"Conflicting observations on {when}")
            self.records[when] = value


def read_source(key: str, data_dir: Path, offline: bool):
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_path = raw_dir / f"h10_{key}.html"
    receipt_path = raw_dir / f"h10_{key}.receipt.json"
    if not source_path.exists():
        if offline:
            raise FileNotFoundError(f"Offline source missing: {source_path}")
        request = Request(SOURCES[key], headers={"User-Agent": "TAILSENTRY-research/0.1 (public H.10 data)"})
        # Network failures are surfaced. No credentials, proxy workarounds or
        # permission escalation are attempted.
        with urlopen(request, timeout=45) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
        if b"<html" not in raw.lower() or len(raw) < 20_000:
            raise ValueError(f"Unexpected H.10 response for {key}; source not saved")
        receipt = {
            "url": SOURCES[key],
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        with source_path.open("xb") as stream:
            stream.write(raw)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"url": SOURCES[key], "retrieved_at_utc": None}
    if receipt.get("sha256", digest) != digest:
        raise ValueError(f"Raw source hash differs from receipt: {source_path}")
    parser = RateTable()
    parser.feed(raw.decode("utf-8"))
    if len(parser.records) < 4_000:
        raise ValueError(f"Too few dated observations in {source_path}: {len(parser.records)}")
    selected = {day: value for day, value in parser.records.items() if BEGIN <= day <= END}
    return selected, {**receipt, "sha256": digest, "path": str(source_path.relative_to(data_dir)), "raw_rows": len(parser.records), "selected_rows": len(selected), "selected_missing": sum(v is None for v in selected.values())}


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(data_dir: Path, *, offline: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)
    sources, provenance = {}, {}
    for key in SOURCES:
        sources[key], provenance[key] = read_source(key, data_dir, offline)
    all_dates = sorted(set().union(*(set(values) for values in sources.values())))
    complete = [day for day in all_dates if all(sources[key].get(day) is not None for key in SOURCES)]
    if len(complete) < 4_000:
        raise ValueError("Too few complete observations after alignment")
    levels = []
    for day in complete:
        # EUR and GBP are USD per foreign unit; JPY source is foreign per USD.
        levels.append({"date": day.isoformat(), "usd_per_eur": sources["eur"][day], "usd_per_jpy": 1.0 / sources["jpy"][day], "usd_per_gbp": sources["gbp"][day]})
    returns = []
    currencies = ("eur", "jpy", "gbp")
    for previous, current in zip(levels, levels[1:]):
        when, before = date.fromisoformat(current["date"]), date.fromisoformat(previous["date"])
        row = {"date": current["date"], "previous_date": previous["date"], "calendar_days": (when - before).days, "split": "train" if when <= TRAIN_END else "evaluation"}
        basket = 0.0
        for currency in currencies:
            ratio = current[f"usd_per_{currency}"] / previous[f"usd_per_{currency}"]
            simple = ratio - 1.0
            row[f"simple_{currency}"] = simple
            row[f"log_{currency}"] = math.log(ratio)
            basket += simple / 3.0
        row["basket_simple_loss"] = -basket
        returns.append(row)
    write_csv(data_dir / "fx_levels.csv", list(levels[0]), levels)
    write_csv(data_dir / "fx_returns.csv", list(returns[0]), returns)
    metadata = {
        "sources": provenance,
        "source_description": "Federal Reserve H.10 noon New York buying rates for cable transfers payable in foreign currencies",
        "licensing_url": "https://www.federalreserve.gov/disclaimer.htm",
        "licensing_note": "Board information is public domain unless otherwise indicated; attribution requested. No Board logos are included.",
        "selection": {"start": BEGIN.isoformat(), "end": END.isoformat(), "train_end": TRAIN_END.isoformat(), "evaluation_start": "2015-01-01", "alignment": "Complete-case source dates before computing returns; no forward fill", "return_interval": "Consecutive complete observations, potentially spanning weekends or holidays", "basket": "Equal USD value exposures to EUR, JPY, GBP; one-observation simple loss = minus average simple currency return; no interest, carry, transaction costs, intraday prices or fitted model"},
        "counts": {"source_union_dates": len(all_dates), "complete_levels": len(levels), "dropped_dates": len(all_dates) - len(levels), "returns": len(returns), "train_returns": sum(r["split"] == "train" for r in returns), "evaluation_returns": sum(r["split"] == "evaluation" for r in returns), "multi_calendar_day_intervals": sum(r["calendar_days"] > 1 for r in returns)},
        "first_complete_date": levels[0]["date"],
        "last_complete_date": levels[-1]["date"],
        "outputs": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("fx_levels.csv", "fx_returns.csv")},
        "limitations": "Current revised source vintage, not point-in-time vintages. Historical observations assess financial model fit at observable tail levels, not 1e-5 event probabilities. The split is fixed and no model is fitted by this preparation script.",
    }
    (data_dir / "fx_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"data_dir": str(data_dir), **metadata["counts"], "first": metadata["first_complete_date"], "last": metadata["last_complete_date"]}, indent=2))
    return metadata


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    cli.add_argument("--offline", action="store_true", help="Require saved source HTML; do not make network requests")
    options = cli.parse_args()
    prepare(options.data_dir.resolve(), offline=options.offline)
