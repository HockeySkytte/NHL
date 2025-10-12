import csv, requests, sys
from pathlib import Path
from datetime import datetime

# Seasons to process: read existing file or generate range
ROOT = Path(__file__).resolve().parents[1]
last_file = ROOT / 'Last_date.csv'

seasons = []
if last_file.exists():
    with open(last_file, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get('Season')
            if s and s.isdigit():
                seasons.append(int(s))
# Ensure unique & sorted
seasons = sorted(set(seasons))

if not seasons:
    print('No seasons found in Last_date.csv; please populate first.')
    sys.exit(1)

out_rows = [('Season','Last_Date')]

for season in seasons:
    url = f'https://api-web.nhle.com/v1/standings/{season}'
    # Attempt to use schedule endpoints per team is heavy; standings endpoint may not supply last date.
    # Instead, use season schedule aggregated endpoint (requires iterating teams). We'll fallback to existing value.
    # Simpler: query each active team schedule for that season, gather latest regular season game date.
    # Active team list heuristic
    latest_date = None
    # Minimal active set from modern era; schedules include opponent so union should cover.
    active = ['ANA','ARI','BOS','BUF','CGY','CAR','CHI','COL','CBJ','DAL','DET','EDM','FLA','LAK','MIN','MTL','NJD','NSH','NYI','NYR','OTT','PHI','PIT','SEA','SJS','STL','TBL','TOR','UTA','VAN','VGK','WPG','WSH']
    for team in active:
        sched_url = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}'
        try:
            resp = requests.get(sched_url, timeout=20)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for g in data.get('games', []):
                gt = str(g.get('gameType') or g.get('gameTypeId'))
                if gt != '2':
                    continue
                gd = g.get('gameDate') or g.get('startTimeUTC')
                if not gd:
                    continue
                # normalize
                if gd.endswith('Z'):
                    gd = gd[:-1]
                try:
                    dt = datetime.fromisoformat(gd.replace('T',' ').split('+')[0])
                except Exception:
                    continue
                if latest_date is None or dt > latest_date:
                    latest_date = dt
        except Exception:
            continue
    if latest_date:
        out_rows.append((str(season), latest_date.date().isoformat()))
        print(season, latest_date.date().isoformat())
    else:
        # Fallback: keep existing value
        # Find existing row
        existing = None
        with open(last_file, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get('Season') == str(season):
                    existing = row.get('Last_Date')
                    break
        out_rows.append((str(season), existing or 'now'))
        print(season, 'fallback', existing)

# Write new file (backup old first)
backup = ROOT / 'Last_date_backup.csv'
if last_file.exists():
    last_file.replace(backup)
with open(last_file, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerows(out_rows)
print('Rebuilt Last_date.csv; backup saved to', backup)
