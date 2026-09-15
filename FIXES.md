# Scraper Fix Notes (working notes, not a finished patch)

Captured from a live debugging session against racingpost.com and racingtv.com on
2026-09-13. These are the concrete findings needed to get `data_collection/` working
again — not yet applied to the scripts themselves.

## 1. `scrape_results.py` — URL typo

Uses `.../racecard/{date}/` (singular). Always 406s. Must be
`.../racecards/{date}/` (plural) — same pattern `scrape_racecards.py` and
`scrape_odds.py` already use correctly.

## 2. Listing-page selectors are dead, but a working replacement exists

Racing Post moved the racecards listing page to a Next.js build using `data-testid`
attributes instead of the old `RC-*` classes / `data-race-is-over` attribute (which
all three scripts currently key off and which no longer exist anywhere on the page).

Confirmed working, current attributes (plain `requests`, no Selenium needed for this
page — 426.8 KB of real HTML, 200 OK):

- `data-testid="Container__RacesSection"` — wraps the day's races
- `data-testid="Link__Race__<raceid>"` — one per individual race, gives the race ID
  needed to build the detail-page URL directly

## 3. Individual race pages: HTTP 406 is headless-browser detection, NOT a paywall

This was the big misdiagnosis to correct. Every individual race page
(`/racecards/<courseid>/<course>/<date>/<raceid>`) 406s for:
- plain `requests` (any headers/cookies/session tried)
- headless Chromium via Selenium (`--headless=new`, with or without
  `--disable-blink-features=AutomationControlled`)

The response headers looked like an entitlement/paywall block at first
(`x-access-decision: block`, `x-user-role: free`) — that was a red herring on my
part. **Confirmed root cause: it's specifically the `--headless` flag.** Removing
only that flag (identical Chromium binary, identical other args, run against the
real desktop display) flips the exact same URL from 406/5.3KB stub to 200/730KB of
full real content. No login, no cookies, no special headers needed.

**Practical implication:** any fix needs a real (non-headless) rendering browser for
the detail-page step. Plain `requests` can never work here — there's no
"un-headless" a raw HTTP client. `scrape_racecards.py` and `scrape_results.py`
currently use plain `requests` for everything and need to move to Selenium (like
`scrape_odds.py` already does) for this step specifically.

**Unattended/cron caveat (not yet verified):** non-headless normally means a real
display has to exist. Xvfb is already installed on this machine
(`/usr/bin/Xvfb`, `/usr/bin/xvfb-run`) and commonly defeats this class of headless
check since it presents as a normal X display — but this hasn't actually been
tested against racingpost.com yet. Needs verification before relying on it for an
unattended/cron job with nobody logged into the desktop.

## 4. Individual race page — current runner-row selectors (verified against real data)

Old scheme (`RC-runnerRow`, `data-test-selector="RC-cardPage-runnerName"`, etc.) is
completely gone — 0 matches even on a successful, fully-loaded page. Confirmed
current replacement, tested against a real Curragh race (26/26 runners matched):

```
data-testid="Container__RunnerRowDesktop"   — one per runner (26 found = 26 runners, exact match)
data-testid="Container__RunnerRowMobile"    — mobile-layout duplicate of the same data

Inside each row:
  Link__Horse                    — horse name (+ link to horse profile)
  Link__Jockey                   — jockey name
  Link__Trainer                  — trainer name
  Container__RunnerNumber        — saddlecloth / draw number
  Container__HorseInfo           — age / weight / form block
  Container__RunnerStats         — OR / TS / RPR ratings
  Container__RunnerRowFormFigures — form figures string
  Text__DaysSinceLastRun
  Container__SilkAndTips / Image__SilkImage — jockey silk image
```

Odds/price-specific testid not yet located — next thing to check.

## 5. Rate limiting — approx. 50 requests before a ~30 minute block

**From the user's own direct experience running these scrapers**, not something I've
reproduced myself in this session: Racing Post appears to allow roughly **50 URL
hits** in a scraping run before rate-limiting kicks in, after which requests fail
for about **30 minutes**. The user has hit this directly — around 50 races scanned
successfully, the rest of a run failing, and switching IP via a VPN was needed to
get past it and finish the run.

**This number is empirical and may drift** (Racing Post can change rate-limit
thresholds without notice) — treat "50 requests / 30 min" as a working assumption
to design around, not a guaranteed constant. Needs to be addressed in any rewrite:

- Batch/throttle requests to stay under the observed threshold (e.g. pause well
  before 50 real detail-page hits, rather than only reacting after a block starts)
- Detect the block condition explicitly (status code / response signature) rather
  than silently failing partway through a run
- Consider a resume/checkpoint mechanism so a run that does get blocked partway
  through doesn't have to restart from race 1
- A VPN/IP-rotation fallback is a real, already-proven workaround, but a smarter
  request cadence should be the first line of defense before reaching for that

## Net status

Nothing has been patched yet — this file is the accumulated diagnosis to build a
real fix from. Confirmed fixable pieces: URL typo, listing-page selectors,
detail-page runner-row selectors, and the headless-vs-real-browser distinction.
Open items: odds/price testid, Xvfb verification for unattended runs, and building
the rate-limit-aware request pacing described above.
