# No Fallback Rule

**Never add fallback logic that silently substitutes different data.**

When a user selects filters (strength state, season stage, team, league, etc.) that return no data, the correct behavior is to **show zeros, show an empty state, or let the app fail** — never to silently fall back to a different filter and show wrong data.

## Why

- A fallback that switches e.g. `strength_state=ENF` to `strength_state=ALL` produces **incorrect data** without the user knowing.
- Wrong data is far worse than no data. The user can't trust any numbers if fallbacks are silently swapping filters.
- If there's genuinely no data for a filter combination, that's valid information — showing zeros is truthful.

## Examples of forbidden fallbacks

- `if not rows and strength != 'ALL': re-query with strength='ALL'`
- `if not rows and team_code: re-query without team_code`
- `if not data: use_hardcoded_defaults()`

## What to do instead

- Return zeros / empty results
- Show a clear message in the UI: "No data for this filter combination"
- Let the query return whatever it returns (even if empty)
