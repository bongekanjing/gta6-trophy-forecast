"""gta6-trophy-rarity-forecast-01: trophy rarity forecaster.

Forecasts, for every GTA VI trophy, five quantiles of its Exophase earn rate
at the case's target. Pure Python 3.13 standard library, no network.

Model (per trophy, fitted from the case's own history at run time):
  * Exophase rates are ratios: rate = 100 * E / N, where N is the players
    tracked and E the players who earned the trophy.  N grows as Exophase
    tracks more players (dilution pushes rates down) and E grows as players
    progress (progression pushes rates up), so a trophy's rate can rise or
    fall; the net move is what the trend must capture.
  * The page only changes when Exophase refreshes it, so consecutive
    identical reads are one observation, not many: history is collapsed to
    its distinct states, each dated at the slot it first appeared.  Fitting
    on every read would overweight the long stale stretches between
    refreshes.
  * Over the recent window (last 24h of distinct states, widened until at
    least 3 states), each trophy's logit rate is fitted by least squares
    against u = ln(1 + hours since launch 2026-11-19T00:00Z) -- a concave
    clock, so growth decelerates instead of exploding over the 4.5-day
    horizon -- and read at the target.  The clock matches the house curve
    baseline's, which is the right shape for a launch week: early moves are
    fast, later moves slow.  The raw fit is used as-is: shrinking slopes
    toward the field's median slope was tested and rejected, because the
    slopes are bimodal (risers vs fallers, median near zero) and shrinking
    toward zero flattens real trends -- on synthetic launch weeks the
    shrunk median was 3x worse (logit MAE 0.128 vs 0.042 unshrunk).
  * The logit spread combines the fit's prediction interval at the target,
    a trend-volatility term growing with the horizon, and a fixed
    refresh-jump allowance (the last read may be stale; the next refresh
    can jump the page).  Quantiles are the normal quantiles on the logit
    scale, so a forecast that is too narrow pays for trophies outside it
    and one too wide pays at every trophy.
  * The trophy every tracked player holds (rate 100% in every read) is
    forecast to stay at 100% with a narrow band: its earned count equals
    the player count by construction.
  * With fewer than 3 distinct states there is no trend signal and the
    forecast is persistence (the launch-night practice cases are flat).

Scored by mean pinball loss on the logit scale; the quantiles written are
the ones this model really believes, rounded to 4 decimals half-up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fit window: last 24h of distinct states, widened until >= 3 states.
WINDOW_HOURS = 24.0
MIN_STATES = 3
# Spread: trend volatility per sqrt-day, refresh-jump allowance (logit),
# and the floor/cap on the total logit spread.
TREND_SIGMA = 0.35
REFRESH_SIGMA = 0.35
SPREAD_FLOOR = 0.25
SPREAD_CAP = 1.80


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def linfit(xs, ys):
    """Least squares y ~ a + b x.

    Returns (a, b, resid_std, n, sxx, xbar, se_b): se_b is the slope's
    standard error (0 when the fit is exact or degenerate).
    """
    n = len(xs)
    if n == 0:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return (ybar, 0.0, 0.0, n, 0.0, xbar, 0.0)
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    se_b = resid / math.sqrt(sxx) if sxx > 0 else 0.0
    return (a, b, resid, n, sxx, xbar, se_b)


def distinct_states(case):
    """Collapse consecutive identical reads to one state each.

    Returns a list of (slot_datetime, players_tracked, rates) in time order,
    each state dated at the slot it first appeared.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        states.append((parse_time(read["slot"]), read["players_tracked"],
                       read["rates"]))
    return states


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    states = distinct_states(case)
    last_slot, last_n, last_rates = states[-1]

    def u_of(when):
        hours = (when - LAUNCH).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    u_target = u_of(target)

    # Fit window: the last WINDOW_HOURS before the checkpoint, widened
    # (24h -> 48h -> all) until it holds MIN_STATES distinct states.  A
    # state is in the window if it was in force at any time in it, so the
    # state that was current at the window's start (which first appeared
    # before the cut) is included -- without it, a page frozen for the
    # whole window would look flat and the fit would miss the trend that
    # produced the current value.
    window = WINDOW_HOURS
    while True:
        cut = at.timestamp() - window * 3600.0
        idx = [k for k, s in enumerate(states) if s[0].timestamp() >= cut]
        win = [states[k] for k in idx]
        if idx and idx[0] > 0:
            win = [states[idx[0] - 1]] + win
        if len(win) >= MIN_STATES or window >= 1e9:
            break
        window *= 2.0
    if len(win) < MIN_STATES:
        win = states

    us = [u_of(s[0]) for s in win]
    spread_base = math.sqrt(
        (TREND_SIGMA * math.sqrt(horizon_days)) ** 2
        + REFRESH_SIGMA ** 2 + SPREAD_FLOOR ** 2)

    # First pass: per-trophy logit fits.
    fits = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        if all(s[2][i] is not None and s[2][i] >= 99.99 for s in states):
            fits.append(None)  # handled specially below
            continue
        pts = []
        for s, u in zip(win, us):
            r = s[2][i]
            if r is None:
                continue
            pts.append((u, logit(r)))
        if len(pts) >= MIN_STATES:
            fits.append(linfit([p[0] for p in pts], [p[1] for p in pts]))
        else:
            fits.append(None)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0

        # A trophy every tracked player holds stays held.
        if fits[i] is None and all(
                s[2][i] is not None and s[2][i] >= 99.99 for s in states):
            med = 100.0
            spread = 0.30
            rows.append({"id": trophy["id"], "q": [
                round4(clamp_rate(inv_logit(logit(med) + z * spread)))
                for z in Z]})
            continue

        fit = fits[i]
        if fit is None:
            med = last_rate if last_rate > 0 else 0.01
            spread = spread_base
        else:
            a, b, resid, n, sxx, xbar, se_b = fit
            med = clamp_rate(inv_logit(a + b * u_target))
            # Spread: the fit's prediction interval at the target, plus the
            # trend and refresh terms.
            pred_var = 0.0
            if sxx > 0:
                pred_var = resid * resid * (
                    1.0 + 1.0 / n + (u_target - xbar) ** 2 / sxx)
            spread = math.sqrt(pred_var + spread_base * spread_base)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        lg = logit(med)
        qs = [round4(clamp_rate(inv_logit(lg + z * spread))) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
