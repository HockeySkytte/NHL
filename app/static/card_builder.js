(function(){
  if(window.__cardBuilderInit) return;
  window.__cardBuilderInit = true;

  const boot = window.CARD_BUILDER_BOOT || {};
  const byId = (id)=> document.getElementById(id);
  const els = {
    teamSelect: byId('teamSelect'),
    seasonSelect: byId('seasonSelect'),
    includeHistoric: byId('includeHistoric'),
    playerSelect: byId('playerSelect'),
    entityWrap: byId('cbEntityWrap'),
    entityLabel: byId('cbEntityLabel'),
    cardTypeSelect: byId('cardTypeSelect'),
    seasonState: byId('builderSeasonState'),
    strengthState: byId('builderStrengthState'),
    rates: byId('builderRates'),
    xgModel: byId('builderXgModel'),
    scope: byId('builderScope'),
    minGp: byId('builderMinGp'),
    minToi: byId('builderMinToi'),
    seasonStateWrap: byId('cbSeasonStateWrap'),
    strengthWrap: byId('cbStrengthStateWrap'),
    ratesWrap: byId('cbRatesWrap'),
    xgWrap: byId('cbXgModelWrap'),
    scopeWrap: byId('cbScopeWrap'),
    minGpWrap: byId('cbMinGpWrap'),
    minToiWrap: byId('cbMinToiWrap'),
    tabFilters: byId('cbTabFilters'),
    tabContent: byId('cbTabContent'),
    sidebarFilters: byId('cbSidebarFilters'),
    sidebarContent: byId('cbSidebarContent'),
    addButtons: Array.from(document.querySelectorAll('[data-block-type]')),
    inspector: byId('cbInspector'),
    savedLayouts: byId('cbSavedLayouts'),
    layoutName: byId('cbLayoutName'),
    newCardBtn: byId('cbNewCardBtn'),
    saveDraftBtn: byId('cbSaveDraftBtn'),
    saveAccountBtn: byId('cbSaveAccountBtn'),
    exportBtn: byId('cbExportBtn'),
    gridToggleBtn: byId('cbGridToggleBtn'),
    gridLessBtn: byId('cbGridLessBtn'),
    gridMoreBtn: byId('cbGridMoreBtn'),
    gridReadout: byId('cbGridReadout'),
    status: byId('cbStatus'),
    selectionMeta: byId('cbSelectionMeta'),
    canvas: byId('cbCanvas'),
    canvasBlocks: byId('cbCanvasBlocks'),
    starterModal: byId('cbStarterModal'),
    closeStarterModal: byId('cbCloseStarterModal'),
    starterTemplateGrid: byId('cbStarterTemplateGrid'),
  };

  if(!els.canvas || !els.teamSelect || !els.seasonSelect || !els.cardTypeSelect) return;

  const teamsData = (function(){
    try{
      const tag = byId('teams-data');
      return tag ? JSON.parse(tag.textContent || '[]') : [];
    }catch(_){
      return [];
    }
  })();

  const state = {
    layout: null,
    defsByType: new Map(),
    requestCache: new Map(),
    savedLayouts: [],
    draftKey: 'cardBuilderDraft_v1',
    renderVersion: 0,
    entityLoadSeq: 0,
    dragging: null,
    applyingControls: false,
    lastRenderPromise: Promise.resolve([]),
    zonesGeoJson: null,
  };

  const CARD_SPECS = {
    skater: {
      label: 'Skater',
      entityLabel: 'Player',
      defsUrl: '/api/skaters/card/defs',
      playersUrl: '/api/skaters/players',
      cardUrl: '/api/skaters/card',
      tableUrl: '/api/skaters/table',
      scopeOptions: [
        { value: 'season', label: 'Season' },
        { value: 'career', label: 'Career' },
      ],
      ratesOptions: [
        { value: 'Totals', label: 'Totals' },
        { value: 'Per60', label: 'Per 60' },
        { value: 'PerGame', label: 'Per Game' },
      ],
      defaultStats: ['Ice Time|GP', 'Production|Points', 'Shooting|ixG', 'Play Driving|xGF%'],
      defaultTable: ['Production|Points', 'Shooting|ixG', 'Play Driving|xGF%'],
      defaultSort: 'Production|Points',
    },
    goalie: {
      label: 'Goalie',
      entityLabel: 'Goalie',
      defsUrl: '/api/goalies/card/defs',
      playersUrl: '/api/goalies/players',
      cardUrl: '/api/goalies/card',
      tableUrl: '/api/goalies/table',
      scopeOptions: [
        { value: 'season', label: 'Season' },
        { value: 'career', label: 'Career' },
      ],
      ratesOptions: [
        { value: 'Totals', label: 'Totals' },
        { value: 'Per60', label: 'Per 60' },
        { value: 'PerGame', label: 'Per Game' },
      ],
      defaultStats: ['Workload|SA', 'Save Percentage|Sv% or FSv%', 'Results|GSAA', 'Results|GSAx'],
      defaultTable: ['Results|GSAx', 'Results|GSAA', 'Save Percentage|Sv% or FSv%'],
      defaultSort: 'Results|GSAx',
    },
    team: {
      label: 'Team',
      entityLabel: 'Team',
      defsUrl: '/api/teams/card/defs',
      cardUrl: '/api/teams/card',
      tableUrl: '/api/teams/table',
      scopeOptions: [
        { value: 'season', label: 'Season' },
        { value: 'total', label: 'Total' },
      ],
      ratesOptions: [
        { value: 'Totals', label: 'Totals' },
        { value: 'PerGame', label: 'Per Game' },
      ],
      defaultStats: ['Offense|GF', 'Defense|GA', 'Play Driving|xGF%', 'Context|GDAx'],
      defaultTable: ['Offense|GF', 'Defense|GA', 'Play Driving|xGF%'],
      defaultSort: 'Play Driving|xGF%',
    },
  };

  const BLOCK_META = {
    header: { label: 'Header', w: 10, h: 4 },
    stats: { label: 'Stats', w: 14, h: 4 },
    table: { label: 'Table', w: 12, h: 9 },
    bar_chart: { label: 'Bar Chart', w: 12, h: 9 },
    shot_map: { label: 'Shot Map', w: 12, h: 8 },
    heat_map: { label: 'Heat Map', w: 12, h: 8 },
    text: { label: 'Text', w: 10, h: 4 },
  };

  const STARTER_TEMPLATES = [
    {
      id: 'skater-spotlight',
      cardType: 'skater',
      name: 'Skater Spotlight',
      description: 'Header, four player KPIs, team scoring bars, and a league top-10 table.',
      meta: 'Best for player share cards',
      blocks: [
        { type: 'header', title: 'Player Spotlight', x: 0, y: 0, w: 10, h: 4 },
        { type: 'stats', title: 'Key Stats', x: 10, y: 0, w: 14, h: 4, metricIds: ['Ice Time|GP', 'Production|Points', 'Shooting|ixG', 'Play Driving|xGF%'] },
        { type: 'bar_chart', title: 'Team Leaders', x: 0, y: 4, w: 12, h: 9, metricIds: ['Production|Points'], sortMetricId: 'Production|Points', limit: 8, excludedFilters: ['playerId'] },
        { type: 'table', title: 'League Top 10', x: 12, y: 4, w: 12, h: 9, metricIds: ['Production|Points', 'Shooting|ixG', 'Play Driving|xGF%'], sortMetricId: 'Production|Points', limit: 10, excludedFilters: ['playerId', 'team'] },
      ],
    },
    {
      id: 'skater-play-driving',
      cardType: 'skater',
      name: 'Skater Play Driving',
      description: 'A skater template built around possession and chance creation metrics.',
      meta: 'Useful for deeper analytics posts',
      blocks: [
        { type: 'header', title: 'Play Driver', x: 0, y: 0, w: 9, h: 4 },
        { type: 'stats', title: 'On-Ice Impact', x: 9, y: 0, w: 15, h: 4, metricIds: ['Play Driving|CF%', 'Play Driving|xGF%', 'Play Driving|xG+/-', 'Context|PDO'] },
        { type: 'table', title: 'Team Context', x: 0, y: 4, w: 12, h: 9, metricIds: ['Play Driving|CF%', 'Play Driving|xGF%', 'Context|PDO'], sortMetricId: 'Play Driving|xGF%', limit: 8, excludedFilters: ['playerId'] },
        { type: 'bar_chart', title: 'League xGF% Leaders', x: 12, y: 4, w: 12, h: 9, metricIds: ['Play Driving|xGF%'], sortMetricId: 'Play Driving|xGF%', limit: 10, excludedFilters: ['playerId', 'team'] },
      ],
    },
    {
      id: 'goalie-form',
      cardType: 'goalie',
      name: 'Goalie Form Card',
      description: 'Goalie identity, core workload and save metrics, plus league leaderboard context.',
      meta: 'Good for goalie recap posts',
      blocks: [
        { type: 'header', title: 'Goalie Spotlight', x: 0, y: 0, w: 10, h: 4 },
        { type: 'stats', title: 'Goalie KPIs', x: 10, y: 0, w: 14, h: 4, metricIds: ['Workload|SA', 'Save Percentage|Sv% or FSv%', 'Results|GSAA', 'Results|GSAx'] },
        { type: 'bar_chart', title: 'League GSAx Leaders', x: 0, y: 4, w: 12, h: 9, metricIds: ['Results|GSAx'], sortMetricId: 'Results|GSAx', limit: 8, excludedFilters: ['playerId', 'team'] },
        { type: 'table', title: 'League Top 10 Goalies', x: 12, y: 4, w: 12, h: 9, metricIds: ['Results|GSAx', 'Results|GSAA', 'Save Percentage|Sv% or FSv%'], sortMetricId: 'Results|GSAx', limit: 10, excludedFilters: ['playerId', 'team'] },
      ],
    },
    {
      id: 'team-snapshot',
      cardType: 'team',
      name: 'Team Snapshot',
      description: 'Team identity, core team KPIs, and league rank context in one 16:9 frame.',
      meta: 'Best for team-wide social posts',
      blocks: [
        { type: 'header', title: 'Team Snapshot', x: 0, y: 0, w: 10, h: 4 },
        { type: 'stats', title: 'Team KPIs', x: 10, y: 0, w: 14, h: 4, metricIds: ['Offense|GF', 'Defense|GA', 'Play Driving|xGF%', 'Context|GDAx'] },
        { type: 'bar_chart', title: 'League xGF% Leaders', x: 0, y: 4, w: 12, h: 9, metricIds: ['Play Driving|xGF%'], sortMetricId: 'Play Driving|xGF%', limit: 8, excludedFilters: ['team'] },
        { type: 'table', title: 'Top Teams', x: 12, y: 4, w: 12, h: 9, metricIds: ['Play Driving|xGF%', 'Offense|GF', 'Defense|GA'], sortMetricId: 'Play Driving|xGF%', limit: 10, excludedFilters: ['team'] },
      ],
    },
  ];

  function deepClone(value){
    try{ return JSON.parse(JSON.stringify(value)); }catch(_){ return null; }
  }

  function clamp(value, min, max){
    return Math.max(min, Math.min(max, value));
  }

  function escapeHtml(value){
    return String(value == null ? '' : value).replace(/[&<>"']/g, (ch)=>({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[ch] || ch));
  }

  function safeJsonParse(value){
    try{ return JSON.parse(String(value || '')); }catch(_){ return null; }
  }

  function slugify(value){
    return String(value || 'card').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'card';
  }

  function smartNumber(value){
    const num = Number(value);
    if(!Number.isFinite(num)) return '—';
    if(Math.abs(num - Math.round(num)) < 0.001) return String(Math.round(num));
    if(Math.abs(num) >= 100) return num.toFixed(1);
    if(Math.abs(num) >= 10) return num.toFixed(1);
    return num.toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
  }

  function formatMetricValue(metricId, value){
    const num = Number(value);
    if(!Number.isFinite(num)) return value == null ? '—' : String(value);
    const id = String(metricId || '');
    const isPct = id.includes('%') || id.toLowerCase().includes('pctg');
    const wantsSign = /(?:\+\/-|\+\/-|GAx|GSAx|GDAx|GSAA|dSv|dFSv|dSh|dFSh)/.test(id);
    if(isPct) return `${smartNumber(num)}%`;
    if(wantsSign && num > 0) return `+${smartNumber(num)}`;
    return smartNumber(num);
  }

  function formatSeasonLabel(value){
    const raw = String(value || '').replace(/\D/g, '');
    if(raw.length !== 8) return String(value || '');
    return `${raw.slice(0,4)}-${raw.slice(6,8)}`;
  }

  function formatDateTime(value){
    if(!value) return 'Never';
    try{
      return new Date(value).toLocaleString();
    }catch(_){
      return String(value || '');
    }
  }

  function previousSeasonId(value){
    const raw = String(value || '').replace(/\D/g, '');
    if(raw.length !== 8) return '';
    const start = Number(raw.slice(0, 4));
    const end = Number(raw.slice(4));
    if(!Number.isFinite(start) || !Number.isFinite(end)) return '';
    return `${start - 1}${String(end - 1).padStart(4, '0')}`;
  }

  function missingSeasonStatsMessage(filters){
    const season = String(filters && filters.season || '');
    const seasonLabel = season ? formatSeasonLabel(season) : 'this season';
    const fallback = previousSeasonId(season);
    const hint = fallback ? ` Try ${formatSeasonLabel(fallback)}.` : '';
    return `No SeasonStats found for ${seasonLabel} with these filters.${hint}`;
  }

  function playerHeadshotUrl(playerId, season, team){
    const pid = String(playerId || '').trim();
    if(!pid) return '';
    const params = new URLSearchParams();
    if(season) params.set('season', String(season || '').trim());
    if(team) params.set('team', String(team || '').trim().toUpperCase());
    const qs = params.toString();
    return `/api/player-headshot/${encodeURIComponent(pid)}.png${qs ? `?${qs}` : ''}`;
  }

  function setStatus(message, kind){
    if(!els.status) return;
    els.status.textContent = String(message || '');
    els.status.classList.toggle('error', kind === 'error');
  }

  function hasOption(select, value){
    return Array.from(select.options || []).some((opt)=> String(opt.value) === String(value));
  }

  function waitFor(predicate, timeoutMs){
    const timeout = Number(timeoutMs) || 4000;
    const started = Date.now();
    return new Promise((resolve)=>{
      const tick = ()=>{
        if(predicate()) return resolve(true);
        if(Date.now() - started >= timeout) return resolve(false);
        window.setTimeout(tick, 60);
      };
      tick();
    });
  }

  function cardType(){
    return String((state.layout && state.layout.cardType) || els.cardTypeSelect.value || 'skater');
  }

  function cardSpec(){
    return CARD_SPECS[cardType()] || CARD_SPECS.skater;
  }

  function findTeamMeta(team){
    const code = String(team || '').toUpperCase();
    return teamsData.find((row)=> String(row.Team || row.team || '').toUpperCase() === code) || null;
  }

  function setSelectOptions(select, options, wantedValue){
    if(!select) return;
    const safeOptions = Array.isArray(options) ? options : [];
    select.innerHTML = safeOptions.map((opt)=> `<option value="${escapeHtml(opt.value)}">${escapeHtml(opt.label)}</option>`).join('');
    const wanted = String(wantedValue || '');
    if(wanted && hasOption(select, wanted)) select.value = wanted;
    else if(select.options.length) select.value = select.options[0].value;
  }

  function layoutFiltersFromControls(){
    const currentType = cardType();
    const spec = CARD_SPECS[currentType] || CARD_SPECS.skater;
    return {
      team: String(els.teamSelect.value || '').toUpperCase(),
      season: String(els.seasonSelect.value || ''),
      playerId: currentType === 'team' ? '' : String(els.playerSelect.value || ''),
      seasonState: String(els.seasonState.value || 'regular'),
      strengthState: String(els.strengthState.value || '5v5'),
      rates: String(els.rates.value || (spec.ratesOptions[0] && spec.ratesOptions[0].value) || 'Totals'),
      xgModel: String(els.xgModel.value || 'xG_F'),
      scope: String(els.scope.value || (spec.scopeOptions[0] && spec.scopeOptions[0].value) || 'season'),
      minGP: clamp(Number(els.minGp.value || 0) || 0, 0, 500),
      minTOI: clamp(Number(els.minToi.value || 0) || 0, 0, 5000),
      includeHistoric: !!(els.includeHistoric && els.includeHistoric.checked),
    };
  }

  function defaultFilters(cardTypeValue){
    const spec = CARD_SPECS[cardTypeValue] || CARD_SPECS.skater;
    return {
      team: String(els.teamSelect.value || '').toUpperCase(),
      season: String(els.seasonSelect.value || ''),
      playerId: '',
      seasonState: 'regular',
      strengthState: '5v5',
      rates: (spec.ratesOptions[0] && spec.ratesOptions[0].value) || 'Totals',
      xgModel: 'xG_F',
      scope: (spec.scopeOptions[0] && spec.scopeOptions[0].value) || 'season',
      minGP: 0,
      minTOI: 0,
      includeHistoric: !!(els.includeHistoric && els.includeHistoric.checked),
    };
  }

  function createBlock(type, overrides){
    const meta = BLOCK_META[type] || BLOCK_META.text;
    const layout = state.layout || { grid: { cols: 24, rows: 13 }, blocks: [] };
    const count = Array.isArray(layout.blocks) ? layout.blocks.length : 0;
    const baseX = (count * 5) % Math.max(1, layout.grid.cols - meta.w + 1);
    const baseY = Math.min(layout.grid.rows - meta.h, Math.floor(count / 3) * 3);
    return normalizeBlock({
      id: `block_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      type,
      title: meta.label,
      x: baseX,
      y: Math.max(0, baseY),
      w: meta.w,
      h: meta.h,
      metricIds: [],
      sortMetricId: '',
      limit: 10,
      excludedFilters: [],
      showTitle: true,
      fontScale: 1,
      text: 'Add your own notes here.',
    }, overrides || {});
  }

  function normalizeBlock(block, overrides){
    const merged = Object.assign({}, block || {}, overrides || {});
    const meta = BLOCK_META[merged.type] || BLOCK_META.text;
    const cols = (state.layout && state.layout.grid && state.layout.grid.cols) || 24;
    const rows = (state.layout && state.layout.grid && state.layout.grid.rows) || 13;
    merged.id = String(merged.id || `block_${Date.now()}`);
    merged.type = BLOCK_META[merged.type] ? merged.type : 'text';
    merged.title = String(merged.title || meta.label);
    merged.x = clamp(Math.round(Number(merged.x) || 0), 0, Math.max(0, cols - 1));
    merged.y = clamp(Math.round(Number(merged.y) || 0), 0, Math.max(0, rows - 1));
    merged.w = clamp(Math.round(Number(merged.w) || meta.w), 1, cols);
    merged.h = clamp(Math.round(Number(merged.h) || meta.h), 1, rows);
    merged.metricIds = Array.isArray(merged.metricIds) ? merged.metricIds.map(String).filter(Boolean) : [];
    merged.sortMetricId = String(merged.sortMetricId || '');
    merged.limit = clamp(Math.round(Number(merged.limit) || 10), 1, 20);
    merged.excludedFilters = Array.isArray(merged.excludedFilters) ? merged.excludedFilters.map(String).filter(Boolean) : [];
    merged.showTitle = merged.showTitle !== false;
    merged.fontScale = clamp(Number(merged.fontScale) || 1, 0.7, 1.7);
    merged.text = String(merged.text || '');
    if(merged.x + merged.w > cols) merged.x = Math.max(0, cols - merged.w);
    if(merged.y + merged.h > rows) merged.y = Math.max(0, rows - merged.h);
    return merged;
  }

  function normalizeLayout(layout){
    const currentType = String(layout && layout.cardType || 'skater');
    const spec = CARD_SPECS[currentType] || CARD_SPECS.skater;
    const filters = Object.assign({}, defaultFilters(currentType), (layout && layout.filters) || {});
    const grid = Object.assign({ cols: 24, rows: 13, show: true, snap: true }, (layout && layout.grid) || {});
    const normalized = {
      id: String(layout && layout.id || ''),
      name: String(layout && layout.name || `${spec.label} card`),
      cardType: currentType,
      filters: {
        team: String(filters.team || '').toUpperCase(),
        season: String(filters.season || ''),
        playerId: currentType === 'team' ? '' : String(filters.playerId || ''),
        seasonState: String(filters.seasonState || 'regular'),
        strengthState: String(filters.strengthState || '5v5'),
        rates: String(filters.rates || ((spec.ratesOptions[0] && spec.ratesOptions[0].value) || 'Totals')),
        xgModel: String(filters.xgModel || 'xG_F'),
        scope: String(filters.scope || ((spec.scopeOptions[0] && spec.scopeOptions[0].value) || 'season')),
        minGP: clamp(Number(filters.minGP || 0) || 0, 0, 500),
        minTOI: clamp(Number(filters.minTOI || 0) || 0, 0, 5000),
        includeHistoric: !!filters.includeHistoric,
      },
      grid: {
        cols: clamp(Math.round(Number(grid.cols) || 24), 8, 48),
        rows: clamp(Math.round(Number(grid.rows) || 13), 5, 32),
        show: grid.show !== false,
        snap: grid.snap !== false,
      },
      blocks: Array.isArray(layout && layout.blocks) ? layout.blocks.map((block)=> normalizeBlock(block)) : [],
      selectedBlockId: String(layout && layout.selectedBlockId || ''),
      starterTemplate: String(layout && layout.starterTemplate || ''),
    };
    if(currentType === 'team') normalized.filters.playerId = '';
    if(!normalized.blocks.some((block)=> block.id === normalized.selectedBlockId)) normalized.selectedBlockId = '';
    return normalized;
  }

  function defaultMetricIds(type, blockType){
    const spec = CARD_SPECS[type] || CARD_SPECS.skater;
    return (blockType === 'stats' ? spec.defaultStats : spec.defaultTable).slice();
  }

  async function ensureDefs(type){
    const wanted = String(type || 'skater');
    if(state.defsByType.has(wanted)) return state.defsByType.get(wanted);
    const spec = CARD_SPECS[wanted] || CARD_SPECS.skater;
    const data = await requestJson(`defs:${wanted}`, spec.defsUrl, { cache: 'no-store' });
    const defs = {
      categories: Array.isArray(data && data.categories) ? data.categories : [],
      metrics: Array.isArray(data && data.metrics) ? data.metrics : [],
    };
    state.defsByType.set(wanted, defs);
    return defs;
  }

  function metricMap(defs){
    const out = {};
    const list = (defs && Array.isArray(defs.metrics)) ? defs.metrics : [];
    list.forEach((metric)=>{
      const id = String(metric && metric.id || '');
      if(id) out[id] = metric;
    });
    return out;
  }

  async function validMetricIds(type, wantedIds, fallbackBlockType){
    const defs = await ensureDefs(type);
    const byId = metricMap(defs);
    const wanted = Array.isArray(wantedIds) ? wantedIds.map(String).filter((id)=> byId[id]) : [];
    if(wanted.length) return wanted;
    return defaultMetricIds(type, fallbackBlockType).filter((id)=> byId[id]);
  }

  async function validSortMetricId(type, wantedId, fallbackBlockType){
    const ids = await validMetricIds(type, [wantedId], fallbackBlockType);
    if(ids.length) return ids[0];
    const spec = CARD_SPECS[type] || CARD_SPECS.skater;
    const defs = await ensureDefs(type);
    const byId = metricMap(defs);
    if(byId[spec.defaultSort]) return spec.defaultSort;
    const fallback = await validMetricIds(type, [], fallbackBlockType);
    return fallback[0] || '';
  }

  async function requestJson(cacheKey, url, options){
    if(state.requestCache.has(cacheKey)) return state.requestCache.get(cacheKey);
    const promise = fetch(url, options).then(async (response)=>{
      let data = null;
      try{ data = await response.json(); }catch(_){ data = {}; }
      if(!response.ok || (data && data.error)){
        throw new Error(String((data && data.error) || `${response.status} ${response.statusText}`));
      }
      return data;
    }).catch((error)=>{
      state.requestCache.delete(cacheKey);
      throw error;
    });
    state.requestCache.set(cacheKey, promise);
    return promise;
  }

  function syncCardTypeControls(){
    const type = cardType();
    const spec = CARD_SPECS[type] || CARD_SPECS.skater;
    els.entityLabel.textContent = spec.entityLabel;
    els.entityWrap.hidden = type === 'team';
    els.xgWrap.hidden = type === 'team';
    els.minGpWrap.hidden = type === 'team';
    els.minToiWrap.hidden = type === 'team';
    setSelectOptions(els.scope, spec.scopeOptions, state.layout && state.layout.filters && state.layout.filters.scope);
    setSelectOptions(els.rates, spec.ratesOptions, state.layout && state.layout.filters && state.layout.filters.rates);
  }

  async function applyLayoutToControls(){
    if(!state.layout) return;
    state.applyingControls = true;
    try{
      els.layoutName.value = state.layout.name || '';
      els.cardTypeSelect.value = state.layout.cardType;
      syncCardTypeControls();

      if(els.includeHistoric && els.includeHistoric.checked !== !!state.layout.filters.includeHistoric){
        els.includeHistoric.checked = !!state.layout.filters.includeHistoric;
        els.includeHistoric.dispatchEvent(new Event('change', { bubbles: true }));
      }

      await waitFor(()=> els.teamSelect.options.length > 0, 4000);
      if(state.layout.filters.team && hasOption(els.teamSelect, state.layout.filters.team) && els.teamSelect.value !== state.layout.filters.team){
        els.teamSelect.value = state.layout.filters.team;
        els.teamSelect.dispatchEvent(new Event('change', { bubbles: true }));
      }

      await waitFor(()=> els.seasonSelect.options.length > 0, 5000);
      if(state.layout.filters.season && hasOption(els.seasonSelect, state.layout.filters.season) && els.seasonSelect.value !== state.layout.filters.season){
        els.seasonSelect.value = state.layout.filters.season;
        els.seasonSelect.dispatchEvent(new Event('change', { bubbles: true }));
      }

      els.seasonState.value = state.layout.filters.seasonState || 'regular';
      els.strengthState.value = state.layout.filters.strengthState || '5v5';
      if(hasOption(els.rates, state.layout.filters.rates)) els.rates.value = state.layout.filters.rates;
      els.xgModel.value = state.layout.filters.xgModel || 'xG_F';
      if(hasOption(els.scope, state.layout.filters.scope)) els.scope.value = state.layout.filters.scope;
      els.minGp.value = String(state.layout.filters.minGP || 0);
      els.minToi.value = String(state.layout.filters.minTOI || 0);
    }finally{
      state.applyingControls = false;
    }

    state.layout.filters = layoutFiltersFromControls();
    await refreshEntities(state.layout.filters.playerId || '');
    updateGridUi();
    renderCanvas();
    renderInspector();
  }

  async function refreshEntities(preferredId){
    if(!state.layout) return [];
    const type = cardType();
    if(type === 'team'){
      els.playerSelect.innerHTML = '';
      state.layout.filters.playerId = '';
      return [];
    }

    const team = String(els.teamSelect.value || state.layout.filters.team || '').toUpperCase();
    const season = String(els.seasonSelect.value || state.layout.filters.season || '');
    const seasonState = String(els.seasonState.value || state.layout.filters.seasonState || 'regular');
    state.layout.filters.team = team;
    state.layout.filters.season = season;

    if(!team || !season){
      els.playerSelect.innerHTML = '<option value="">Select team and season</option>';
      return [];
    }

    const seq = ++state.entityLoadSeq;
    els.playerSelect.disabled = true;
    els.playerSelect.innerHTML = '<option value="">Loading…</option>';
    try{
      const spec = cardSpec();
      const params = new URLSearchParams();
      params.set('team', team);
      params.set('season', season);
      params.set('seasonState', seasonState);
      const data = await requestJson(`players:${type}:${params.toString()}`, `${spec.playersUrl}?${params.toString()}`, { cache: 'no-store' });
      if(seq !== state.entityLoadSeq) return [];
      const players = Array.isArray(data && data.players) ? data.players.slice() : [];
      players.sort((a, b)=> String(a && a.name || '').localeCompare(String(b && b.name || '')));
      els.playerSelect.innerHTML = '<option value="">Select player</option>' + players.map((player)=> `<option value="${escapeHtml(player.playerId)}">${escapeHtml(player.name || player.playerId)}</option>`).join('');
      const wanted = String(preferredId || state.layout.filters.playerId || '');
      if(wanted && players.some((player)=> String(player.playerId) === wanted)) els.playerSelect.value = wanted;
      else if(players.length) els.playerSelect.value = String(players[0].playerId || '');
      state.layout.filters.playerId = String(els.playerSelect.value || '');
      return players;
    }catch(error){
      els.playerSelect.innerHTML = '<option value="">Error loading players</option>';
      setStatus(`Could not load ${cardSpec().entityLabel.toLowerCase()} list.`, 'error');
      return [];
    }finally{
      els.playerSelect.disabled = false;
    }
  }

  function currentBlock(){
    if(!state.layout) return null;
    return (state.layout.blocks || []).find((block)=> block.id === state.layout.selectedBlockId) || null;
  }

  function selectBlock(blockId){
    if(!state.layout) return;
    state.layout.selectedBlockId = String(blockId || '');
    setSidebarPane('content');
    updateSelectionMeta();
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  function updateSelectionMeta(){
    if(!els.selectionMeta || !state.layout) return;
    const block = currentBlock();
    if(!block){
      els.selectionMeta.textContent = `${state.layout.grid.cols} × ${state.layout.grid.rows} grid`;
      return;
    }
    els.selectionMeta.textContent = `${BLOCK_META[block.type].label} · x ${block.x} · y ${block.y} · w ${block.w} · h ${block.h}`;
  }

  function updateCanvasSelection(blockId){
    Array.from(els.canvasBlocks.querySelectorAll('.cb-block-shell')).forEach((node)=>{
      node.classList.toggle('selected', String(node.dataset.blockId || '') === String(blockId || ''));
    });
  }

  function updateBlockShellMeta(shell, block){
    if(!shell || !block) return;
    const sub = shell.querySelector('.cb-block-sub');
    if(sub) sub.textContent = `${block.w} × ${block.h}`;
    shell.style.setProperty('--cb-font-scale', String(block.fontScale || 1));
  }

  function syncInspectorRectInputs(block){
    if(!block) return;
    ['X', 'Y', 'W', 'H'].forEach((suffix)=>{
      const input = byId(`cbInspector${suffix}`);
      if(input && document.activeElement !== input) input.value = String(block[suffix.toLowerCase()] || 0);
    });
    const sizeValue = byId('cbInspectorFontScaleValue');
    if(sizeValue) sizeValue.textContent = `${Math.round((Number(block.fontScale) || 1) * 100)}%`;
  }

  function duplicateBlock(blockId){
    if(!state.layout) return;
    const block = (state.layout.blocks || []).find((item)=> item.id === blockId);
    if(!block) return;
    const duplicate = normalizeBlock(deepClone(block) || {}, {
      id: `block_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      x: clamp(block.x + 1, 0, Math.max(0, state.layout.grid.cols - block.w)),
      y: clamp(block.y + 1, 0, Math.max(0, state.layout.grid.rows - block.h)),
    });
    state.layout.blocks.push(duplicate);
    state.layout.selectedBlockId = duplicate.id;
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  function removeBlock(blockId){
    if(!state.layout || !blockId) return;
    const before = state.layout.blocks.length;
    state.layout.blocks = state.layout.blocks.filter((item)=> item.id !== blockId);
    if(state.layout.blocks.length === before) return;
    if(state.layout.selectedBlockId === blockId) state.layout.selectedBlockId = '';
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  function isFormControlTarget(target){
    return !!(target && target.closest && target.closest('input, textarea, select, button, [contenteditable="true"]'));
  }

  function updateGridUi(){
    if(!state.layout) return;
    els.canvas.style.setProperty('--cb-cols', String(state.layout.grid.cols));
    els.canvas.style.setProperty('--cb-rows', String(state.layout.grid.rows));
    els.canvas.classList.toggle('grid-hidden', !state.layout.grid.show);
    els.gridReadout.textContent = `${state.layout.grid.cols} × ${state.layout.grid.rows}`;
    els.gridToggleBtn.textContent = state.layout.grid.show ? 'Hide Grid' : 'Show Grid';
  }

  function saveDraft(silent){
    if(!state.layout) return;
    try{
      localStorage.setItem(state.draftKey, JSON.stringify(state.layout));
      if(!silent) setStatus('Draft saved locally.');
    }catch(_){
      setStatus('Could not save the local draft.', 'error');
    }
  }

  function loadDraft(){
    const raw = safeJsonParse(localStorage.getItem(state.draftKey));
    if(!raw || typeof raw !== 'object') return null;
    return normalizeLayout(raw);
  }

  function setSidebarPane(pane){
    const want = pane === 'content' ? 'content' : 'filters';
    els.tabFilters.classList.toggle('active', want === 'filters');
    els.tabContent.classList.toggle('active', want === 'content');
    els.tabFilters.setAttribute('aria-selected', want === 'filters' ? 'true' : 'false');
    els.tabContent.setAttribute('aria-selected', want === 'content' ? 'true' : 'false');
    els.sidebarFilters.hidden = want !== 'filters';
    els.sidebarContent.hidden = want !== 'content';
  }

  function supportedExcludedFilters(block){
    const type = block && block.type;
    if(type === 'table' || type === 'bar_chart'){
      return cardType() === 'team'
        ? ['team', 'seasonState', 'strengthState', 'rates', 'scope', 'includeHistoric']
        : ['team', 'playerId', 'seasonState', 'strengthState', 'rates', 'xgModel', 'scope', 'minGP', 'minTOI'];
    }
    if(type === 'stats'){
      return cardType() === 'team'
        ? ['seasonState', 'strengthState', 'rates', 'scope']
        : ['seasonState', 'strengthState', 'rates', 'xgModel', 'scope', 'minGP', 'minTOI'];
    }
    if(type === 'shot_map' || type === 'heat_map'){
      return cardType() === 'team'
        ? ['seasonState', 'strengthState']
        : ['playerId', 'seasonState', 'strengthState', 'xgModel'];
    }
    return [];
  }

  function buildEffectiveFilters(block){
    const filters = Object.assign({}, state.layout ? state.layout.filters : {});
    const excluded = new Set(Array.isArray(block && block.excludedFilters) ? block.excludedFilters.map(String) : []);
    if(excluded.has('team')) delete filters.team;
    if(excluded.has('playerId')) delete filters.playerId;
    if(excluded.has('seasonState')) filters.seasonState = 'all';
    if(excluded.has('strengthState')) filters.strengthState = 'all';
    if(excluded.has('rates')) filters.rates = 'Totals';
    if(excluded.has('xgModel')) filters.xgModel = 'xG_F';
    if(excluded.has('scope')) filters.scope = cardType() === 'team' ? 'season' : 'season';
    if(excluded.has('minGP')) filters.minGP = 0;
    if(excluded.has('minTOI')) filters.minTOI = 0;
    if(excluded.has('includeHistoric')) filters.includeHistoric = true;
    return filters;
  }

  function applyBlockRect(shell, block){
    const cols = state.layout.grid.cols;
    const rows = state.layout.grid.rows;
    shell.style.left = `${(block.x / cols) * 100}%`;
    shell.style.top = `${(block.y / rows) * 100}%`;
    shell.style.width = `${(block.w / cols) * 100}%`;
    shell.style.height = `${(block.h / rows) * 100}%`;
  }

  async function fetchCardMetrics(metricIds, block){
    const type = cardType();
    const filters = buildEffectiveFilters(block);
    const mids = await validMetricIds(type, metricIds, 'stats');
    if(type === 'team'){
      if(!filters.team) throw new Error('team_required');
      const params = new URLSearchParams();
      params.set('team', filters.team);
      params.set('season', filters.season || '');
      params.set('seasonState', filters.seasonState || 'regular');
      params.set('strengthState', filters.strengthState || '5v5');
      params.set('rates', filters.rates || 'Totals');
      params.set('scope', filters.scope || 'season');
      params.set('metricIds', mids.join(','));
      return requestJson(`card:${type}:${params.toString()}`, `${cardSpec().cardUrl}?${params.toString()}`, { cache: 'no-store' });
    }
    if(!filters.playerId) throw new Error('player_required');
    const params = new URLSearchParams();
    params.set('season', filters.season || '');
    params.set('playerId', filters.playerId);
    params.set('seasonState', filters.seasonState || 'regular');
    params.set('strengthState', filters.strengthState || '5v5');
    params.set('xgModel', filters.xgModel || 'xG_F');
    params.set('rates', filters.rates || 'Totals');
    params.set('scope', filters.scope || 'season');
    params.set('minGP', String(filters.minGP || 0));
    params.set('minTOI', String(filters.minTOI || 0));
    params.set('metricIds', mids.join(','));
    return requestJson(`card:${type}:${params.toString()}`, `${cardSpec().cardUrl}?${params.toString()}`, { cache: 'no-store' });
  }

  async function fetchPlayerLanding(playerId){
    return requestJson(`landing:${playerId}`, `/api/player/${encodeURIComponent(playerId)}/landing`, { cache: 'no-store' });
  }

  async function ensureZonesGeoJson(){
    if(state.zonesGeoJson) return state.zonesGeoJson;
    try{
      state.zonesGeoJson = await requestJson('zones:geojson', '/static/zones.json', { cache: 'no-store' });
    }catch(_){
      state.zonesGeoJson = { type: 'FeatureCollection', features: [] };
    }
    return state.zonesGeoJson;
  }

  async function fetchSpatialRows(block){
    const filters = buildEffectiveFilters(block);
    if(!filters.team) throw new Error('team_required');
    if(!filters.season) throw new Error('season_required');
    const params = new URLSearchParams();
    params.set('team', filters.team);
    params.set('season', filters.season || '');
    params.set('seasonState', filters.seasonState || 'regular');
    params.set('strengthState', filters.strengthState || '5v5');
    params.set('xgModel', filters.xgModel || 'xG_F');
    if(filters.playerId) params.set('player', filters.playerId);
    const isGoalieView = cardType() === 'goalie';
    const baseUrl = isGoalieView ? '/api/goalies/goaltending' : '/api/skaters/shooting';
    return requestJson(`spatial:${cardType()}:${params.toString()}`, `${baseUrl}?${params.toString()}`, { cache: 'no-store' });
  }

  function zoneLabelPoint(coords, zoneId){
    const xs = coords.map((point)=> Number(point[0]) || 0);
    const ys = coords.map((point)=> Number(point[1]) || 0);
    let cx = 0;
    let cy = 0;
    if(zoneId === 'O04' || zoneId === 'O10'){
      cx = xs.reduce((sum, value)=> sum + value, 0) / Math.max(1, xs.length);
      cy = ys.reduce((sum, value)=> sum + value, 0) / Math.max(1, ys.length);
    }else{
      cx = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2;
      cy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    }
    if(Math.max.apply(null, xs) >= 99) cx = Math.min(cx, 92);
    if(Math.min.apply(null, ys) <= -41) cy = Math.max(cy, -38);
    if(Math.max.apply(null, ys) >= 41) cy = Math.min(cy, 38);
    return { x: cx, y: cy };
  }

  function shotMarkerMarkup(event){
    const x = Number(event && event.x);
    const y = Number(event && event.y);
    if(!Number.isFinite(x) || !Number.isFinite(y)) return '';
    const cy = -y;
    if(event && event.goal){
      return `<circle class="cb-shot-marker ${cardType() === 'goalie' ? 'goal-against' : 'goal'}" cx="${x}" cy="${cy}" r="1.6"></circle>`;
    }
    if(Number(event && event.shot) === 1){
      return `<circle class="cb-shot-marker shot" cx="${x}" cy="${cy}" r="1.35"></circle>`;
    }
    return `<polygon class="cb-shot-marker miss" points="${x},${cy - 1.8} ${x - 1.55},${cy + 1.05} ${x + 1.55},${cy + 1.05}"></polygon>`;
  }

  function heatFill(intensity, isAgainst){
    if(intensity <= 0) return 'rgba(200,210,230,0.14)';
    if(isAgainst){
      const r = Math.round(255 - 18 * intensity);
      const g = Math.round(144 - 66 * intensity);
      const b = Math.round(144 - 66 * intensity);
      const a = 0.2 + 0.55 * intensity;
      return `rgba(${r},${g},${b},${a})`;
    }
    const r = Math.round(176 - 96 * intensity);
    const g = Math.round(214 - 84 * intensity);
    const b = Math.round(255 - 20 * intensity);
    const a = 0.2 + 0.55 * intensity;
    return `rgba(${r},${g},${b},${a})`;
  }

  function nextFrame(){
    return new Promise((resolve)=> window.requestAnimationFrame(()=> resolve()));
  }

  async function waitForImages(container){
    const pending = Array.from(container.querySelectorAll('img')).filter((img)=> !img.complete);
    if(!pending.length) return;
    await Promise.allSettled(pending.map((img)=> new Promise((resolve)=>{
      img.addEventListener('load', resolve, { once: true });
      img.addEventListener('error', resolve, { once: true });
    })));
  }

  async function fetchLeaderboardRows(block, metricIds, sortMetricId){
    const type = cardType();
    const filters = buildEffectiveFilters(block);
    const mids = Array.from(new Set([].concat(metricIds || [], sortMetricId || []))).filter(Boolean);

    if(type === 'team'){
      const params = new URLSearchParams();
      params.set('season', filters.season || '');
      params.set('seasonState', filters.seasonState || 'regular');
      params.set('strengthState', filters.strengthState || '5v5');
      params.set('rates', filters.rates || 'Totals');
      params.set('scope', filters.scope || 'season');
      params.set('includeHistoric', filters.includeHistoric ? '1' : '0');
      params.set('metricIds', mids.join(','));
      const data = await requestJson(`table:${type}:${params.toString()}`, `${cardSpec().tableUrl}?${params.toString()}`, { cache: 'no-store' });
      let rows = Array.isArray(data && data.rows) ? data.rows.slice() : [];
      if(filters.team) rows = rows.filter((row)=> String(row && row.team || '').toUpperCase() === String(filters.team || '').toUpperCase());
      return rows.map((row)=> ({
        id: String(row && row.team || ''),
        name: (findTeamMeta(row && row.team) || {}).Name || String(row && row.team || ''),
        team: String(row && row.team || ''),
        metrics: mids.reduce((acc, id)=>{ acc[id] = row ? row[id] : null; return acc; }, {}),
      }));
    }

    const playersParams = new URLSearchParams();
    playersParams.set('season', filters.season || '');
    playersParams.set('seasonState', filters.seasonState || 'regular');
    if(filters.team) playersParams.set('team', filters.team);
    else playersParams.set('scope', 'league');
    const playersData = await requestJson(`players:${type}:${playersParams.toString()}`, `${cardSpec().playersUrl}?${playersParams.toString()}`, { cache: 'no-store' });
    let players = Array.isArray(playersData && playersData.players) ? playersData.players.slice() : [];
    if(filters.playerId) players = players.filter((player)=> String(player && player.playerId || '') === String(filters.playerId || ''));
    if(!players.length) return [];
    const playerIds = players.map((player)=> player.playerId).filter(Boolean);
    const body = {
      season: filters.season || '',
      seasonState: filters.seasonState || 'regular',
      strengthState: filters.strengthState || '5v5',
      xgModel: filters.xgModel || 'xG_F',
      rates: filters.rates || 'Totals',
      scope: filters.scope || 'season',
      minGP: filters.minGP || 0,
      minTOI: filters.minTOI || 0,
      playerIds,
      metricIds: mids,
    };
    const cacheKey = `table:${type}:${JSON.stringify(body)}`;
    const data = await requestJson(cacheKey, cardSpec().tableUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(body),
    });
    const metricsByPlayer = {};
    (Array.isArray(data && data.players) ? data.players : []).forEach((row)=>{
      const pid = String(row && row.playerId || '');
      if(pid) metricsByPlayer[pid] = row && row.metrics || {};
    });
    return players.filter((player)=> metricsByPlayer[String(player.playerId || '')]).map((player)=> ({
      id: String(player.playerId || ''),
      name: String(player.name || player.playerId || ''),
      team: String(player.team || filters.team || ''),
      pos: String(player.position || player.pos || ''),
      metrics: metricsByPlayer[String(player.playerId || '')] || {},
    }));
  }

  function sortRows(rows, sortMetricId){
    return rows.slice().sort((left, right)=>{
      const a = Number(left && left.metrics && left.metrics[sortMetricId]);
      const b = Number(right && right.metrics && right.metrics[sortMetricId]);
      if(Number.isFinite(a) && Number.isFinite(b)) return b - a;
      if(Number.isFinite(a)) return -1;
      if(Number.isFinite(b)) return 1;
      return String(left && left.name || '').localeCompare(String(right && right.name || ''));
    });
  }

  async function renderBlockContent(block, body, version){
    try{
      if(block.type === 'text'){
        body.innerHTML = `<div class="cb-text-copy">${escapeHtml(block.text || '')}</div>`;
        return;
      }

      if(block.type === 'header'){
        const filters = buildEffectiveFilters(block);
        if(cardType() === 'team'){
          if(!filters.team){
            body.innerHTML = '<div class="cb-empty-state">Pick a team to render the header.</div>';
            return;
          }
          const meta = findTeamMeta(filters.team) || {};
          if(version !== state.renderVersion || !body.isConnected) return;
          body.innerHTML = `
            <div class="cb-header-card">
              <img class="cb-team-logo" src="/api/team-logo/${encodeURIComponent(filters.team)}.svg" alt="${escapeHtml(filters.team)} logo" />
              <div class="cb-header-meta">
                <div class="cb-header-name">${escapeHtml(meta.Name || filters.team)}</div>
                <div class="cb-header-sub">
                  <span class="cb-header-pill"><strong>Team</strong> ${escapeHtml(filters.team)}</span>
                  <span class="cb-header-pill"><strong>Season</strong> ${escapeHtml(formatSeasonLabel(filters.season))}</span>
                  <span class="cb-header-pill"><strong>State</strong> ${escapeHtml(filters.seasonState || 'regular')}</span>
                </div>
              </div>
            </div>`;
          return;
        }

        if(!filters.playerId){
          body.innerHTML = '<div class="cb-empty-state">Header blocks need a selected player or goalie.</div>';
          return;
        }
        const landing = await fetchPlayerLanding(filters.playerId);
        if(version !== state.renderVersion || !body.isConnected) return;
        const team = String(landing && landing.currentTeamAbbrev || filters.team || '').toUpperCase();
        const fullName = `${landing && landing.firstName && landing.firstName.default || ''} ${landing && landing.lastName && landing.lastName.default || ''}`.trim() || 'Unknown player';
        const headshot = playerHeadshotUrl(filters.playerId, filters.season, team || filters.team || '');
        const fallbackLogo = `/api/team-logo/${encodeURIComponent(team || filters.team || '')}.svg`;
        const details = [team, landing && landing.position || landing && landing.positionCode || '', landing && landing.sweaterNumber ? `#${landing.sweaterNumber}` : ''].filter(Boolean);
        body.innerHTML = `
          <div class="cb-header-card">
            <img class="cb-avatar" src="${escapeHtml(headshot || fallbackLogo)}" alt="${escapeHtml(fullName)}" onerror="this.onerror=null;this.src='${escapeHtml(fallbackLogo)}';" />
            <div class="cb-header-meta">
              <div class="cb-header-name">${escapeHtml(fullName)}</div>
              <div class="cb-header-sub">
                ${details.map((item)=> `<span class="cb-header-pill"><strong>${escapeHtml(item)}</strong></span>`).join('')}
                <span class="cb-header-pill"><strong>Season</strong> ${escapeHtml(formatSeasonLabel(filters.season))}</span>
              </div>
            </div>
          </div>`;
        return;
      }

      if(block.type === 'stats'){
        const defs = await ensureDefs(cardType());
        const mids = await validMetricIds(cardType(), block.metricIds, 'stats');
        const data = await fetchCardMetrics(mids, block);
        if(version !== state.renderVersion || !body.isConnected) return;
        if(data && data.seasonStatsMissing){
          body.innerHTML = `<div class="cb-empty-state">${escapeHtml(missingSeasonStatsMessage(buildEffectiveFilters(block)))}</div>`;
          return;
        }
        const byMetric = metricMap(defs);
        const metrics = data && data.metrics || {};
        body.innerHTML = `<div class="cb-metric-grid">${mids.map((id)=>{
          const entry = metrics[id] || {};
          const pct = Number(entry && entry.pct);
          const pctWidth = Number.isFinite(pct) ? clamp(pct, 0, 100) : 0;
          const name = (byMetric[id] && byMetric[id].name) || id.split('|')[1] || id;
          return `
            <div class="cb-metric-card">
              <div class="cb-metric-name">${escapeHtml(name)}</div>
              <div class="cb-metric-value">${escapeHtml(formatMetricValue(id, entry && entry.value))}</div>
              <div class="cb-meter"><div class="cb-meter-fill" style="width:${pctWidth}%"></div></div>
              <div class="cb-meter-label">${Number.isFinite(pct) ? `${Math.round(pct)}th percentile` : 'No percentile available'}</div>
            </div>`;
        }).join('')}</div>`;
        return;
      }

      if(block.type === 'table' || block.type === 'bar_chart'){
        const defs = await ensureDefs(cardType());
        const mids = await validMetricIds(cardType(), block.metricIds, 'table');
        const sortMetricId = await validSortMetricId(cardType(), block.sortMetricId, 'table');
        const rows = sortRows(await fetchLeaderboardRows(block, mids, sortMetricId), sortMetricId).slice(0, clamp(Number(block.limit || 10) || 10, 1, 20));
        const byMetric = metricMap(defs);
        if(version !== state.renderVersion || !body.isConnected) return;
        if(!rows.length){
          body.innerHTML = `<div class="cb-empty-state">${escapeHtml(missingSeasonStatsMessage(buildEffectiveFilters(block)))}</div>`;
          return;
        }

        if(block.type === 'table'){
          const displayMetrics = mids.slice(0, 3);
          body.innerHTML = `
            <table class="cb-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  ${displayMetrics.map((id)=> `<th class="num">${escapeHtml((byMetric[id] && byMetric[id].name) || id.split('|')[1] || id)}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${rows.map((row, index)=> `
                  <tr>
                    <td class="cb-rank">${index + 1}</td>
                    <td>${escapeHtml(row.name || row.team || '')}${row.team && cardType() !== 'team' ? ` <span class="cb-bar-value">• ${escapeHtml(row.team)}</span>` : ''}</td>
                    ${displayMetrics.map((id)=> `<td class="num">${escapeHtml(formatMetricValue(id, row.metrics[id]))}</td>`).join('')}
                  </tr>`).join('')}
              </tbody>
            </table>`;
          return;
        }

        const maxValue = Math.max.apply(null, rows.map((row)=> Math.abs(Number(row.metrics[sortMetricId]) || 0)).concat([1]));
        body.innerHTML = `<div class="cb-bar-list">${rows.map((row)=>{
          const value = Number(row.metrics[sortMetricId]);
          const width = Number.isFinite(value) ? (Math.abs(value) / maxValue) * 100 : 0;
          return `
            <div class="cb-bar-row">
              <div class="cb-bar-head">
                <div class="cb-bar-name">${escapeHtml(row.name || row.team || '')}${row.team && cardType() !== 'team' ? ` <span class="cb-bar-value">• ${escapeHtml(row.team)}</span>` : ''}</div>
                <div class="cb-bar-value">${escapeHtml(formatMetricValue(sortMetricId, row.metrics[sortMetricId]))}</div>
              </div>
              <div class="cb-bar-track"><div class="cb-bar-fill" style="width:${width}%"></div></div>
            </div>`;
        }).join('')}</div>`;
        return;
      }

      if(block.type === 'shot_map'){
        const data = await fetchSpatialRows(block);
        if(version !== state.renderVersion || !body.isConnected) return;
        const events = Array.isArray(data && data.events) ? data.events : [];
        const label = cardType() === 'goalie' ? 'shots against' : 'shot events';
        body.innerHTML = `
          <div class="cb-rink-shell">
            <div class="cb-rink-wrap">
              <img class="cb-rink-img" src="/static/hockey-rink-half.png" alt="Rink" />
              <svg class="cb-rink-overlay" viewBox="25 -42.5 75 85" preserveAspectRatio="xMidYMid meet" aria-label="Shot map">
                ${events.map((event)=> shotMarkerMarkup(event)).join('')}
              </svg>
            </div>
            <div class="cb-map-caption">${escapeHtml(String(events.length || 0))} ${escapeHtml(label)} for the current filters.</div>
          </div>`;
        return;
      }

      if(block.type === 'heat_map'){
        const [data, geoJson] = await Promise.all([fetchSpatialRows(block), ensureZonesGeoJson()]);
        if(version !== state.renderVersion || !body.isConnected) return;
        const events = Array.isArray(data && data.events) ? data.events : [];
        const counts = {};
        events.forEach((event)=>{
          const key = String(event && event.boxId || '');
          if(!key) return;
          counts[key] = (counts[key] || 0) + 1;
        });
        const features = Array.isArray(geoJson && geoJson.features) ? geoJson.features.filter((feature)=>{
          const id = String(feature && (feature.id || (feature.properties && feature.properties.id)) || '');
          return id.startsWith('O');
        }) : [];
        const maxCount = Math.max.apply(null, Object.values(counts).concat([1]));
        const isAgainst = cardType() === 'goalie';
        body.innerHTML = `
          <div class="cb-rink-shell">
            <div class="cb-rink-wrap">
              <img class="cb-rink-img" src="/static/hockey-rink-half.png" alt="Rink" />
              <svg class="cb-rink-overlay" viewBox="25 -42.5 75 85" preserveAspectRatio="xMidYMid meet" aria-label="Heat map">
                ${features.map((feature)=>{
                  const id = String(feature && (feature.id || (feature.properties && feature.properties.id)) || '');
                  const coords = feature && feature.geometry && feature.geometry.coordinates && feature.geometry.coordinates[0];
                  if(!Array.isArray(coords) || !coords.length) return '';
                  const count = counts[id] || 0;
                  const point = zoneLabelPoint(coords, id);
                  const points = coords.map((coord)=> `${Number(coord[0]) || 0},${-(Number(coord[1]) || 0)}`).join(' ');
                  return `
                    <polygon class="cb-zone-polygon" points="${points}" fill="${heatFill(count / maxCount, isAgainst)}"></polygon>
                    <text class="cb-zone-count" x="${point.x}" y="${-point.y}">${count}</text>`;
                }).join('')}
              </svg>
            </div>
            <div class="cb-map-caption">Zone totals reflect the current filters and update when block filters change.</div>
          </div>`;
          return;
      }
    }catch(error){
      if(version !== state.renderVersion || !body.isConnected) return;
      body.innerHTML = `<div class="cb-empty-state">${escapeHtml(String(error && error.message || 'Could not render block.'))}</div>`;
    }
  }

  function renderCanvas(){
    if(!state.layout){
      state.lastRenderPromise = Promise.resolve([]);
      return state.lastRenderPromise;
    }
    state.renderVersion += 1;
    const version = state.renderVersion;
    updateGridUi();
    updateSelectionMeta();
    els.canvasBlocks.innerHTML = '';

    if(!state.layout.blocks.length){
      els.canvasBlocks.innerHTML = '<div class="cb-empty-state" style="position:absolute; inset:0;">Choose a starter template or add blocks from the Content tab.</div>';
      state.lastRenderPromise = Promise.resolve([]);
      return state.lastRenderPromise;
    }

    const renderPromises = [];
    state.layout.blocks.forEach((rawBlock)=>{
      const block = normalizeBlock(rawBlock);
      Object.assign(rawBlock, block);
      const shell = document.createElement('div');
      shell.className = 'cb-block-shell' + (state.layout.selectedBlockId === block.id ? ' selected' : '');
      shell.dataset.blockId = block.id;
      applyBlockRect(shell, block);
      shell.style.setProperty('--cb-font-scale', String(block.fontScale || 1));
      const titleHtml = block.showTitle !== false ? `
          <div class="cb-block-head">
            <div class="cb-block-title">${escapeHtml(block.title || ((BLOCK_META[block.type] && BLOCK_META[block.type].label) || 'Block'))}</div>
            <div class="cb-block-sub">${escapeHtml(block.w)} × ${escapeHtml(block.h)}</div>
          </div>` : '';
      shell.innerHTML = `
        <div class="cb-block-badge">${escapeHtml((BLOCK_META[block.type] && BLOCK_META[block.type].label) || block.type)}</div>
        <div class="cb-block-tools">
          <button class="cb-block-tool" type="button" data-block-action="duplicate" aria-label="Duplicate block">+</button>
          <button class="cb-block-tool" type="button" data-block-action="delete" aria-label="Delete block">×</button>
        </div>
        <div class="cb-block${block.showTitle === false ? ' no-title' : ''}">
          ${titleHtml}
          <div class="cb-block-body${block.showTitle === false ? ' drag-handle' : ''}"><div class="cb-empty-state">Loading…</div></div>
        </div>
        <button class="cb-resize-handle" type="button" aria-label="Resize block"></button>`;
      const head = shell.querySelector('.cb-block-head');
      const body = shell.querySelector('.cb-block-body');
      const resizeHandle = shell.querySelector('.cb-resize-handle');
      const tools = shell.querySelectorAll('[data-block-action]');
      shell.addEventListener('click', ()=>{ selectBlock(block.id); });
      if(head) head.addEventListener('pointerdown', (event)=> startDrag(event, 'move', block.id, shell));
      else body.addEventListener('pointerdown', (event)=> startDrag(event, 'move', block.id, shell));
      resizeHandle.addEventListener('pointerdown', (event)=> startDrag(event, 'resize', block.id, shell));
      tools.forEach((button)=>{
        button.addEventListener('click', (event)=>{
          event.preventDefault();
          event.stopPropagation();
          const action = String(button.getAttribute('data-block-action') || '');
          if(action === 'duplicate') duplicateBlock(block.id);
          if(action === 'delete') removeBlock(block.id);
        });
      });
      els.canvasBlocks.appendChild(shell);
      renderPromises.push(Promise.resolve(renderBlockContent(block, body, version)).catch(()=>{}));
    });
    state.lastRenderPromise = Promise.allSettled(renderPromises);
    return state.lastRenderPromise;
  }

  function startDrag(event, mode, blockId, shell){
    if(!state.layout) return;
    event.preventDefault();
    event.stopPropagation();
    const block = (state.layout.blocks || []).find((item)=> item.id === blockId);
    if(!block) return;
    if(state.layout.selectedBlockId !== blockId){
      state.layout.selectedBlockId = String(blockId || '');
      setSidebarPane('content');
      updateCanvasSelection(blockId);
      updateSelectionMeta();
      renderInspector();
      saveDraft(true);
    }
    const rect = els.canvas.getBoundingClientRect();
    const liveShell = els.canvasBlocks.querySelector(`[data-block-id="${blockId}"]`) || shell;
    state.dragging = {
      mode,
      blockId,
      shell: liveShell,
      startX: event.clientX,
      startY: event.clientY,
      rect,
      startBlock: { x: block.x, y: block.y, w: block.w, h: block.h },
    };
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp, { once: true });
  }

  function onPointerMove(event){
    if(!state.dragging || !state.layout) return;
    const block = (state.layout.blocks || []).find((item)=> item.id === state.dragging.blockId);
    if(!block) return;
    const cellW = state.dragging.rect.width / state.layout.grid.cols;
    const cellH = state.dragging.rect.height / state.layout.grid.rows;
    const dx = (event.clientX - state.dragging.startX) / (cellW || 1);
    const dy = (event.clientY - state.dragging.startY) / (cellH || 1);
    const snap = state.layout.grid.snap !== false;
    const deltaX = snap ? Math.round(dx) : Math.trunc(dx);
    const deltaY = snap ? Math.round(dy) : Math.trunc(dy);
    if(state.dragging.mode === 'move'){
      block.x = clamp(state.dragging.startBlock.x + deltaX, 0, Math.max(0, state.layout.grid.cols - block.w));
      block.y = clamp(state.dragging.startBlock.y + deltaY, 0, Math.max(0, state.layout.grid.rows - block.h));
    }else{
      block.w = clamp(state.dragging.startBlock.w + deltaX, 1, state.layout.grid.cols - state.dragging.startBlock.x);
      block.h = clamp(state.dragging.startBlock.h + deltaY, 1, state.layout.grid.rows - state.dragging.startBlock.y);
    }
    applyBlockRect(state.dragging.shell, block);
    updateBlockShellMeta(state.dragging.shell, block);
    syncInspectorRectInputs(block);
    updateSelectionMeta();
  }

  function onPointerUp(){
    window.removeEventListener('pointermove', onPointerMove);
    state.dragging = null;
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  async function renderInspector(){
    const block = currentBlock();
    if(!block){
      els.inspector.className = 'cb-inspector-empty';
      els.inspector.textContent = 'Select a block on the canvas to edit its size, placement, metrics, and excluded filters.';
      return;
    }
    const currentType = cardType();
    const defs = await ensureDefs(currentType);
    const byMetric = metricMap(defs);
    if(block.id !== (currentBlock() && currentBlock().id)) return;

    const supportedFilters = supportedExcludedFilters(block);
    const filterLabels = {
      team: 'Team',
      playerId: currentType === 'goalie' ? 'Goalie' : 'Player',
      seasonState: 'Season State',
      strengthState: 'Strength State',
      rates: 'Totals / Rates',
      xgModel: 'xG Model',
      scope: 'Scope',
      minGP: 'Min GP',
      minTOI: 'Min TOI',
      includeHistoric: 'Include Historic Teams',
    };
    const metricOptions = (defs.metrics || []).map((metric)=>{
      const id = String(metric && metric.id || '');
      const label = `${metric && metric.category || ''} · ${metric && metric.name || id}`;
      const selected = (block.metricIds || []).includes(id) ? ' selected' : '';
      return `<option value="${escapeHtml(id)}"${selected}>${escapeHtml(label)}</option>`;
    }).join('');

    const sortOptions = (defs.metrics || []).map((metric)=>{
      const id = String(metric && metric.id || '');
      const selected = String(block.sortMetricId || '') === id ? ' selected' : '';
      const label = `${metric && metric.category || ''} · ${metric && metric.name || id}`;
      return `<option value="${escapeHtml(id)}"${selected}>${escapeHtml(label)}</option>`;
    }).join('');

    const excludedHtml = supportedFilters.length
      ? `<div class="cb-chip-grid">${supportedFilters.map((key)=> `
          <label class="cb-check-row"><input class="cb-check-input" type="checkbox" data-filter-key="${escapeHtml(key)}" ${(block.excludedFilters || []).includes(key) ? 'checked' : ''} /> <span class="cb-check-label">${escapeHtml(filterLabels[key] || key)}</span></label>`).join('')}
        </div>`
      : '<div class="cb-help-text">This block always follows the main card filters.</div>';

    els.inspector.className = '';
    els.inspector.innerHTML = `
      <div class="cb-inspector">
        <div class="cb-inspector-row">
          <label>Title
            <input id="cbInspectorTitle" type="text" value="${escapeHtml(block.title || '')}" />
          </label>
        </div>
        <div class="cb-inspector-row cb-toggle-row">
          <label class="cb-check-row"><input id="cbInspectorShowTitle" class="cb-check-input" type="checkbox" ${block.showTitle !== false ? 'checked' : ''} /> <span class="cb-check-label">Show title</span></label>
          <div class="cb-range-row">
            <label>Text Size
              <input id="cbInspectorFontScale" type="range" min="0.7" max="1.7" step="0.05" value="${escapeHtml(block.fontScale || 1)}" />
            </label>
            <div id="cbInspectorFontScaleValue" class="cb-range-note">${escapeHtml(Math.round((Number(block.fontScale) || 1) * 100))}%</div>
          </div>
        </div>
        <div class="cb-inline-grid">
          <label>X<input id="cbInspectorX" type="number" min="0" step="1" value="${escapeHtml(block.x)}" /></label>
          <label>Y<input id="cbInspectorY" type="number" min="0" step="1" value="${escapeHtml(block.y)}" /></label>
        </div>
        <div class="cb-inline-grid">
          <label>Width<input id="cbInspectorW" type="number" min="1" step="1" value="${escapeHtml(block.w)}" /></label>
          <label>Height<input id="cbInspectorH" type="number" min="1" step="1" value="${escapeHtml(block.h)}" /></label>
        </div>
        ${block.type === 'text' ? `
          <div class="cb-inspector-row">
            <label>Text
              <textarea id="cbInspectorText">${escapeHtml(block.text || '')}</textarea>
            </label>
          </div>` : ''}
        ${block.type === 'stats' || block.type === 'table' || block.type === 'bar_chart' ? `
          <div class="cb-inspector-row">
            <label>Metrics
              <select id="cbInspectorMetricIds" class="cb-multi" multiple>${metricOptions}</select>
            </label>
          </div>` : ''}
        ${block.type === 'table' || block.type === 'bar_chart' ? `
          <div class="cb-inline-grid">
            <label>Sort Metric
              <select id="cbInspectorSortMetric">${sortOptions}</select>
            </label>
            <label>Rows
              <input id="cbInspectorLimit" type="number" min="1" max="20" step="1" value="${escapeHtml(block.limit || 10)}" />
            </label>
          </div>` : ''}
        <div class="cb-inspector-row">
          <div class="cb-pane-title">Excluded Filters</div>
          ${excludedHtml}
        </div>
        <div class="cb-inline-grid">
          <button id="cbInspectorDuplicate" class="cb-mini-btn" type="button">Duplicate</button>
          <button id="cbInspectorDelete" class="cb-mini-btn" type="button">Delete</button>
        </div>
      </div>`;

    const bind = (selector, eventName, handler)=>{
      const node = els.inspector.querySelector(selector);
      if(node) node.addEventListener(eventName, handler);
    };

    bind('#cbInspectorTitle', 'input', (event)=>{
      block.title = String(event.target.value || '');
      renderCanvas();
      saveDraft(true);
    });
    bind('#cbInspectorShowTitle', 'change', (event)=>{
      block.showTitle = !!(event.target && event.target.checked);
      renderCanvas();
      saveDraft(true);
    });
    const applyFontScale = (event)=>{
      block.fontScale = clamp(Number(event.target.value) || 1, 0.7, 1.7);
      const shell = els.canvasBlocks.querySelector(`[data-block-id="${block.id}"]`);
      if(shell) updateBlockShellMeta(shell, block);
      syncInspectorRectInputs(block);
      saveDraft(true);
    };
    bind('#cbInspectorFontScale', 'input', applyFontScale);
    bind('#cbInspectorFontScale', 'change', applyFontScale);
    ['X', 'Y', 'W', 'H'].forEach((suffix)=>{
      const applyRect = (event)=>{
        const prop = suffix.toLowerCase();
        block[prop] = clamp(Math.round(Number(event.target.value || 0) || 0), prop === 'w' || prop === 'h' ? 1 : 0, prop === 'x' ? state.layout.grid.cols - 1 : prop === 'y' ? state.layout.grid.rows - 1 : prop === 'w' ? state.layout.grid.cols : state.layout.grid.rows);
        Object.assign(block, normalizeBlock(block));
        const shell = els.canvasBlocks.querySelector(`[data-block-id="${block.id}"]`);
        if(shell){
          applyBlockRect(shell, block);
          updateBlockShellMeta(shell, block);
        }
        syncInspectorRectInputs(block);
        updateSelectionMeta();
        saveDraft(true);
      };
      bind(`#cbInspector${suffix}`, 'input', applyRect);
      bind(`#cbInspector${suffix}`, 'change', ()=>{
        renderCanvas();
        renderInspector();
      });
    });
    bind('#cbInspectorText', 'input', (event)=>{
      block.text = String(event.target.value || '');
      renderCanvas();
      saveDraft(true);
    });
    bind('#cbInspectorMetricIds', 'change', async (event)=>{
      block.metricIds = Array.from(event.target.selectedOptions || []).map((opt)=> String(opt.value || ''));
      if(block.type === 'table' || block.type === 'bar_chart'){
        const validSort = await validSortMetricId(currentType, block.sortMetricId || block.metricIds[0], 'table');
        block.sortMetricId = validSort;
      }
      renderCanvas();
      renderInspector();
      saveDraft(true);
    });
    bind('#cbInspectorSortMetric', 'change', (event)=>{
      block.sortMetricId = String(event.target.value || '');
      renderCanvas();
      saveDraft(true);
    });
    bind('#cbInspectorLimit', 'change', (event)=>{
      block.limit = clamp(Math.round(Number(event.target.value || 10) || 10), 1, 20);
      renderCanvas();
      saveDraft(true);
    });
    Array.from(els.inspector.querySelectorAll('[data-filter-key]')).forEach((node)=>{
      node.addEventListener('change', ()=>{
        block.excludedFilters = Array.from(els.inspector.querySelectorAll('[data-filter-key]:checked')).map((input)=> String(input.getAttribute('data-filter-key') || ''));
        renderCanvas();
        saveDraft(true);
      });
    });
    bind('#cbInspectorDuplicate', 'click', ()=>{
      duplicateBlock(block.id);
    });
    bind('#cbInspectorDelete', 'click', ()=>{
      removeBlock(block.id);
    });
  }

  function renderStarterTemplates(){
    els.starterTemplateGrid.innerHTML = STARTER_TEMPLATES.map((template)=> `
      <div class="cb-template-card">
        <div class="cb-template-kicker">${escapeHtml(CARD_SPECS[template.cardType].label)}</div>
        <div class="cb-template-title">${escapeHtml(template.name)}</div>
        <div class="cb-template-copy">${escapeHtml(template.description)}</div>
        <div class="cb-template-meta">${escapeHtml(template.meta)}</div>
        <div class="cb-template-actions">
          <button class="cb-primary-btn" type="button" data-template-id="${escapeHtml(template.id)}">Use Template</button>
        </div>
      </div>`).join('');
  }

  function openStarterModal(){
    renderStarterTemplates();
    els.starterModal.hidden = false;
  }

  function closeStarterModal(){
    els.starterModal.hidden = true;
  }

  async function applyStarterTemplate(templateId){
    const template = STARTER_TEMPLATES.find((item)=> item.id === templateId);
    if(!template) return;
    const previous = state.layout ? deepClone(state.layout.filters) : defaultFilters(template.cardType);
    state.layout = normalizeLayout({
      id: '',
      name: template.name,
      cardType: template.cardType,
      filters: Object.assign({}, defaultFilters(template.cardType), {
        team: previous && previous.team || '',
        season: previous && previous.season || '',
        includeHistoric: previous && previous.includeHistoric,
      }),
      grid: { cols: 24, rows: 13, show: true, snap: true },
      blocks: template.blocks.map((block)=> createBlock(block.type, block)),
      starterTemplate: template.id,
    });
    closeStarterModal();
    await applyLayoutToControls();
    saveDraft(true);
    setStatus(`Loaded ${template.name}.`);
  }

  function addBlock(type){
    if(!state.layout) return;
    const block = createBlock(type, type === 'text' ? { title: 'Text note' } : {});
    state.layout.blocks.push(block);
    state.layout.selectedBlockId = block.id;
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  async function loadAccountLayouts(){
    if(!(boot.authUser && boot.authUser.user_id)){
      els.savedLayouts.innerHTML = '<div class="cb-help-text">Log in to save cards to your account. Local draft saving still works.</div>';
      return;
    }
    try{
      const response = await fetch(boot.layoutsUrl, { cache: 'no-store' });
      const data = await response.json();
      if(!response.ok) throw new Error(String(data && data.error || response.status));
      state.savedLayouts = Array.isArray(data && data.layouts) ? data.layouts : [];
      renderSavedLayouts();
    }catch(error){
      els.savedLayouts.innerHTML = `<div class="cb-help-text">${escapeHtml(String(error && error.message || 'Could not load saved cards.'))}</div>`;
    }
  }

  function renderSavedLayouts(){
    if(!(boot.authUser && boot.authUser.user_id)) return;
    if(!state.savedLayouts.length){
      els.savedLayouts.innerHTML = '<div class="cb-help-text">No account-saved cards yet.</div>';
      return;
    }
    els.savedLayouts.innerHTML = state.savedLayouts.map((layout)=> `
      <div class="cb-saved-item">
        <div>
          <div class="cb-saved-name">${escapeHtml(layout.name || 'Untitled card')}</div>
          <div class="cb-saved-meta">${escapeHtml(CARD_SPECS[layout.cardType] ? CARD_SPECS[layout.cardType].label : layout.cardType || 'Card')} · Updated ${escapeHtml(formatDateTime(layout.updatedAt))}</div>
        </div>
        <div class="cb-saved-actions">
          <button class="cb-mini-btn" type="button" data-layout-action="load" data-layout-id="${escapeHtml(layout.id)}">Load</button>
          <button class="cb-mini-btn" type="button" data-layout-action="delete" data-layout-id="${escapeHtml(layout.id)}">Delete</button>
        </div>
      </div>`).join('');
  }

  async function saveToAccount(){
    if(!(boot.authUser && boot.authUser.user_id)){
      window.location.href = boot.loginUrl;
      return;
    }
    if(!state.layout) return;
    state.layout.name = String(els.layoutName.value || state.layout.name || `${cardSpec().label} card`).trim();
    const config = {
      version: 1,
      cardType: state.layout.cardType,
      filters: state.layout.filters,
      grid: state.layout.grid,
      blocks: state.layout.blocks,
      selectedBlockId: state.layout.selectedBlockId,
      starterTemplate: state.layout.starterTemplate,
    };
    const payload = {
      id: state.layout.id || '',
      name: state.layout.name,
      cardType: state.layout.cardType,
      config,
    };
    try{
      const response = await fetch(boot.saveLayoutUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': boot.csrfToken || '',
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if(!response.ok) throw new Error(String(data && data.error || response.status));
      const saved = data && data.layout;
      if(saved){
        state.layout.id = String(saved.id || state.layout.id || '');
        state.layout.name = String(saved.name || state.layout.name || '');
      }
      saveDraft(true);
      await loadAccountLayouts();
      setStatus('Saved to account.');
    }catch(error){
      setStatus(String(error && error.message || 'Could not save to account.'), 'error');
    }
  }

  async function loadSavedLayout(layoutId){
    const match = state.savedLayouts.find((item)=> String(item.id) === String(layoutId));
    if(!match) return;
    state.layout = normalizeLayout(Object.assign({}, match.config || {}, {
      id: match.id,
      name: match.name,
      cardType: match.cardType,
    }));
    await applyLayoutToControls();
    saveDraft(true);
    setStatus(`Loaded ${match.name}.`);
  }

  async function deleteSavedLayout(layoutId){
    if(!(boot.authUser && boot.authUser.user_id)) return;
    const match = state.savedLayouts.find((item)=> String(item.id) === String(layoutId));
    if(!match) return;
    if(!window.confirm(`Delete ${match.name}?`)) return;
    try{
      const url = String(boot.deleteLayoutUrlTemplate || '').replace('__LAYOUT_ID__', encodeURIComponent(layoutId));
      const response = await fetch(url, {
        method: 'DELETE',
        headers: { 'X-CSRF-Token': boot.csrfToken || '' },
      });
      const data = await response.json();
      if(!response.ok) throw new Error(String(data && data.error || response.status));
      state.savedLayouts = state.savedLayouts.filter((item)=> String(item.id) !== String(layoutId));
      if(state.layout && String(state.layout.id) === String(layoutId)) state.layout.id = '';
      renderSavedLayouts();
      saveDraft(true);
      setStatus('Deleted saved card.');
    }catch(error){
      setStatus(String(error && error.message || 'Could not delete saved card.'), 'error');
    }
  }

  async function exportPng(){
    if(typeof window.html2canvas !== 'function'){
      setStatus('Export library is unavailable.', 'error');
      return;
    }
    try{
      await renderCanvas();
      await waitForImages(els.canvas);
      await nextFrame();
      await nextFrame();
      els.canvas.classList.add('exporting');
      const snapshot = await window.html2canvas(els.canvas, {
        backgroundColor: null,
        scale: 2,
        useCORS: true,
        logging: false,
      });
      const link = document.createElement('a');
      link.href = snapshot.toDataURL('image/png');
      link.download = `${slugify(state.layout && state.layout.name || 'card')}.png`;
      link.click();
      setStatus('PNG exported.');
    }catch(error){
      setStatus(String(error && error.message || 'Could not export PNG.'), 'error');
    }finally{
      els.canvas.classList.remove('exporting');
    }
  }

  async function handleFilterChange(options){
    if(state.applyingControls || !state.layout) return;
    state.layout.cardType = String(els.cardTypeSelect.value || state.layout.cardType || 'skater');
    syncCardTypeControls();
    state.layout.filters = layoutFiltersFromControls();
    if(options && options.reloadEntities){
      await refreshEntities(options.preferredId || '');
      state.layout.filters = layoutFiltersFromControls();
    }
    if(cardType() === 'team') state.layout.filters.playerId = '';
    renderCanvas();
    renderInspector();
    saveDraft(true);
  }

  function bindEvents(){
    els.tabFilters.addEventListener('click', ()=> setSidebarPane('filters'));
    els.tabContent.addEventListener('click', ()=> setSidebarPane('content'));
    els.addButtons.forEach((button)=>{
      button.addEventListener('click', ()=> addBlock(button.getAttribute('data-block-type')));
    });
    els.newCardBtn.addEventListener('click', openStarterModal);
    els.closeStarterModal.addEventListener('click', closeStarterModal);
    els.starterModal.addEventListener('click', (event)=>{
      if(event.target === els.starterModal) closeStarterModal();
    });
    els.starterTemplateGrid.addEventListener('click', (event)=>{
      const button = event.target.closest('[data-template-id]');
      if(!button) return;
      applyStarterTemplate(button.getAttribute('data-template-id'));
    });
    els.layoutName.addEventListener('input', ()=>{
      if(!state.layout) return;
      state.layout.name = String(els.layoutName.value || '').trim() || `${cardSpec().label} card`;
      saveDraft(true);
    });
    els.saveDraftBtn.addEventListener('click', ()=> saveDraft(false));
    els.saveAccountBtn.addEventListener('click', saveToAccount);
    els.exportBtn.addEventListener('click', exportPng);
    els.gridToggleBtn.addEventListener('click', ()=>{
      if(!state.layout) return;
      state.layout.grid.show = !state.layout.grid.show;
      updateGridUi();
      saveDraft(true);
    });
    els.gridLessBtn.addEventListener('click', ()=>{
      if(!state.layout) return;
      state.layout.grid.cols = clamp(state.layout.grid.cols - 2, 8, 48);
      state.layout.grid.rows = clamp(state.layout.grid.rows - 1, 5, 32);
      state.layout.blocks = state.layout.blocks.map((block)=> normalizeBlock(block));
      renderCanvas();
      renderInspector();
      saveDraft(true);
    });
    els.gridMoreBtn.addEventListener('click', ()=>{
      if(!state.layout) return;
      state.layout.grid.cols = clamp(state.layout.grid.cols + 2, 8, 48);
      state.layout.grid.rows = clamp(state.layout.grid.rows + 1, 5, 32);
      state.layout.blocks = state.layout.blocks.map((block)=> normalizeBlock(block));
      renderCanvas();
      renderInspector();
      saveDraft(true);
    });

    els.cardTypeSelect.addEventListener('change', async ()=>{
      if(state.applyingControls || !state.layout) return;
      state.layout = normalizeLayout(Object.assign({}, state.layout, { cardType: els.cardTypeSelect.value, filters: Object.assign({}, state.layout.filters, { playerId: '' }) }));
      await applyLayoutToControls();
      saveDraft(true);
    });

    document.addEventListener('keydown', (event)=>{
      if(!state.layout || !state.layout.selectedBlockId) return;
      if(isFormControlTarget(event.target)) return;
      if(event.key === 'Delete' || event.key === 'Backspace'){
        event.preventDefault();
        removeBlock(state.layout.selectedBlockId);
      }
      if(event.key === 'Escape'){
        event.preventDefault();
        state.layout.selectedBlockId = '';
        updateCanvasSelection('');
        updateSelectionMeta();
        renderInspector();
        saveDraft(true);
      }
    });

    els.teamSelect.addEventListener('change', async ()=>{
      if(state.applyingControls) return;
      await waitFor(()=> els.seasonSelect.options.length > 0, 4000);
      await handleFilterChange({ reloadEntities: true });
    });
    els.seasonSelect.addEventListener('change', async ()=>{
      if(state.applyingControls) return;
      await handleFilterChange({ reloadEntities: true });
    });
    if(els.includeHistoric){
      els.includeHistoric.addEventListener('change', async ()=>{
        if(state.applyingControls) return;
        await waitFor(()=> els.teamSelect.options.length > 0, 2000);
        await handleFilterChange({ reloadEntities: true });
      });
    }
    [els.seasonState, els.strengthState, els.rates, els.xgModel, els.scope, els.minGp, els.minToi].forEach((node)=>{
      if(!node) return;
      node.addEventListener('change', ()=> handleFilterChange({ reloadEntities: false }));
    });
    els.playerSelect.addEventListener('change', ()=> handleFilterChange({ reloadEntities: false }));
    els.savedLayouts.addEventListener('click', (event)=>{
      const button = event.target.closest('[data-layout-action]');
      if(!button) return;
      const action = String(button.getAttribute('data-layout-action') || '');
      const id = String(button.getAttribute('data-layout-id') || '');
      if(action === 'load') loadSavedLayout(id);
      if(action === 'delete') deleteSavedLayout(id);
    });
  }

  async function init(){
    bindEvents();
    setSidebarPane('filters');
    await waitFor(()=> els.teamSelect.options.length > 0, 5000);
    await waitFor(()=> els.seasonSelect.options.length > 0, 5000);

    const draft = loadDraft();
    state.layout = draft || normalizeLayout({ cardType: 'skater', name: 'Skater card', filters: defaultFilters('skater'), blocks: [] });
    await applyLayoutToControls();
    await loadAccountLayouts();
    if(!(draft && draft.blocks && draft.blocks.length)) openStarterModal();
  }

  init();
})();