"""
Map settled-results JSON (the API's /v1/results shape, verified 2026-09-20
against two 2026-05-26..2026-09-19 exports and against raceform.db on the two
overlap days) onto the `data` table row shape used by raceform.db.

Verified against history (706 matched runners, 2026-05-26/27): pos, prize,
draw, going, off (from off_dt, 24h clock), sp (fractional string) all match
exactly; wgt/or/btn match >= 96%.

Two things the old assumed mapping got wrong (and this fixes):
  * `performance_rating` / `speed_rating` are NOT raceform's rpr / ts: on the
    overlap days history has rpr='–' where the vendor has performance_rating
    113, and history ts=57 where speed_rating is '-'. The vendor's own `rpr`
    and `tsr` fields are empty in every row. So rpr and ts are left NULL for
    loaded rows (prior_rpr / prior_ts will not see these runs).
  * `weight_lbs` -> wgt breaks parse_wgt(); the stone-lb string `weight`
    ("8-13") is what raceform stores.
"""
from __future__ import annotations

import re

_YARDS = re.compile(r"\d+y$")
_TIME = re.compile(r"^(\d+):(\d)\.(\d+)$")


def _blank(v) -> bool:
    return v is None or str(v).strip() in ("", "-", "–", "—")


def _int(v):
    if _blank(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _num(v):
    if _blank(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pos(v):
    """Numeric finish -> int; PU/F/UR/... kept as the code (raceform stores text)."""
    if _blank(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return str(v).strip()


def _dist(d):
    """'2m13y' -> '2m' (raceform's dist has no yardage); '6f' unchanged."""
    if not d:
        return None
    return _YARDS.sub("", str(d).strip()) or None


def _time(t):
    """'4:5.30' -> '4:05.30' (raceform zero-pads seconds)."""
    if _blank(t):
        return None
    m = _TIME.match(str(t).strip())
    return f"{m.group(1)}:0{m.group(2)}.{m.group(3)}" if m else str(t).strip()


def _off(race: dict):
    """24h 'HH:MM' from off_dt (the API's `off` is a 12h clock without am/pm)."""
    dt = race.get("off_dt") or ""
    return dt[11:16] if len(dt) >= 16 else race.get("off")


def results_to_rows(payload: dict, fetched_at: str, regions: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for race in payload.get("results", []):
        if regions is not None and race.get("region") not in regions:
            continue
        runners = race.get("runners", [])
        base = {
            "date": race.get("date"),
            "course": race.get("course"),
            "race_id": race.get("race_id"),
            "off": _off(race),
            "race_name": race.get("race_name"),
            "type": race.get("type"),
            "class": race.get("class"),
            "pattern": race.get("pattern"),
            "rating_band": race.get("rating_band"),
            "age_band": race.get("age_band"),
            "sex_rest": race.get("sex_rest"),
            "dist": _dist(race.get("dist")),
            "going": race.get("going"),
            "ran": len(runners),
            "fetched_at": fetched_at,
        }
        for x in runners:
            row = dict(base)
            row.update({
                "num": _int(x.get("number")),
                "pos": _pos(x.get("position")),
                "draw": _int(x.get("draw")),
                "ovr_btn": _num(x.get("ovr_btn")),
                "btn": _num(x.get("btn")),
                "horse": x.get("horse"),
                "age": _int(x.get("age")),
                "sex": x.get("sex"),
                "wgt": None if _blank(x.get("weight")) else x.get("weight"),
                "hg": x.get("headgear"),
                "time": _time(x.get("time")),
                "sp": None if _blank(x.get("sp")) else x.get("sp"),
                "jockey": x.get("jockey"),
                "trainer": x.get("trainer"),
                "prize": _int(x.get("prize")),
                "or": _int(x.get("or", x.get("ofr"))),
                "rpr": None,   # vendor performance_rating is a different rating; see module docstring
                "ts": None,    # vendor speed_rating likewise
                "sire": x.get("sire"),
                "dam": x.get("dam"),
                "damsire": x.get("damsire"),
                "owner": x.get("owner"),
                "comment": x.get("comment"),
            })
            rows.append(row)
    return rows
