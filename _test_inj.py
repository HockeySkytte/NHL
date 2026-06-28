import requests, re, json
s = requests.Session()
r = s.get('http://127.0.0.1:5000/login', timeout=30)
m = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', r.text)
csrf = m.group(1) if m else ''
r = s.post('http://127.0.0.1:5000/login', data={
    'email': 'lars.sunesen.skytte@gmail.com',
    'password': 'Dervarengang3smaagrise',
    'csrf_token': csrf,
}, timeout=30, allow_redirects=False)
print('login', r.status_code)

# Test 20262027 season shift + injuries
sim = s.post('http://127.0.0.1:5000/api/projections/simulate-season', json={
    'season': 20262027,
    'team': 'ANA',
    'lineup': [{'pid': 8478402, 'pos': 'F', 'games': 82}],
    'injuries': [{'injuredPid': 8478402, 'replacementPid': 8478401, 'startDate': '2026-10-01', 'endDate': '2026-12-31'}],
}, timeout=180)
d = sim.json()
print('sim status', sim.status_code)
print('games', d.get('simulatedGames'))
st = (d.get('standings') or [])[:3]
print('top3', [(x['team'], x['points']) for x in st])

# Verify dates in the schedule are 2026+ (test via team-season-points-custom)
pts = s.post('http://127.0.0.1:5000/api/projections/team-season-points-custom', json={
    'team': 'ANA', 'season': 20262027, 'lineup': [{'pid': 8478402, 'pos': 'F', 'games': 82}],
}, timeout=60)
pd = pts.json()
print('KPI pts 20262027:', pd.get('projectedPoints'), 'games:', pd.get('games'))

# Compare to 20252026
pts2 = s.post('http://127.0.0.1:5000/api/projections/team-season-points-custom', json={
    'team': 'ANA', 'season': 20252026, 'lineup': [{'pid': 8478402, 'pos': 'F', 'games': 82}],
}, timeout=60)
pd2 = pts2.json()
print('KPI pts 20252026:', pd2.get('projectedPoints'), 'games:', pd2.get('games'))

# Test injuries on KPI for a long injury
pts_inj = s.post('http://127.0.0.1:5000/api/projections/team-season-points-custom', json={
    'team': 'ANA', 'season': 20252026,
    'lineup': [{'pid': 8478402, 'pos': 'F', 'games': 82}],
    'injuries': [{'injuredPid': 8478402, 'replacementPid': 8478401, 'startDate': '2025-10-01', 'endDate': '2026-04-30'}],
}, timeout=60)
pd3 = pts_inj.json()
print('KPI pts (full injury):', pd3.get('projectedPoints'), 'games:', pd3.get('games'))