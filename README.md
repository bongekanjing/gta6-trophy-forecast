# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Logit-trend forecaster for GTA VI trophy earn rates at a target date. It collapses the case history to the distinct Exophase refresh states, fits each trophy's logit earn rate by least squares against ln(1 + hours since launch), and reads the fit at the target, with an honest spread (fit prediction interval + trend volatility + a refresh-jump allowance). A trophy every tracked player holds is forecast to stay at 100%. Pure Python 3.13 standard library, no network, deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=yg9xmt6c&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
