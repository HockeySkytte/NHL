from scripts.lineups import scrape_dailyfaceoff_lineup, TEAM_URLS

if __name__ == '__main__':
    url = TEAM_URLS['ANA']
    lu = scrape_dailyfaceoff_lineup(url, team_abbrev='ANA')
    print('Counts:', len(lu['forwards']), len(lu['defense']), len(lu['goalies']))
    def show(items):
        return [ {k: it.get(k) for k in ('name','jersey','teamCode','pos','unit')} for it in items ]
    print('Forwards:', show(lu['forwards']))
    print('Defense:', show(lu['defense']))
    print('Goalies:', show(lu['goalies']))
