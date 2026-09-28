/* Manager Game — franchise color theming.
   Mirrors base.html setTheme(): given a team abbrev, recolor --accent,
   --panel-alt and --value-text from the franchise color and expose the raw
   color as --mgr-team for glows/tints. Reads the team list from the
   #mgr-teams-data JSON tag rendered by the manager templates. */
(function () {
  function channels(hex) {
    hex = String(hex || '').replace('#', '');
    if (hex.length !== 6) return null;
    return {
      r: parseInt(hex.substring(0, 2), 16),
      g: parseInt(hex.substring(2, 4), 16),
      b: parseInt(hex.substring(4, 6), 16),
    };
  }
  function lightenDarken(hex, factor) {
    const c = channels(hex);
    if (!c) return hex;
    let r, g, b;
    if (factor > 1) {
      r = c.r + (255 - c.r) * (factor - 1);
      g = c.g + (255 - c.g) * (factor - 1);
      b = c.b + (255 - c.b) * (factor - 1);
    } else {
      r = c.r * factor; g = c.g * factor; b = c.b * factor;
    }
    r = Math.min(255, Math.max(0, Math.round(r)));
    g = Math.min(255, Math.max(0, Math.round(g)));
    b = Math.min(255, Math.max(0, Math.round(b)));
    return '#' + r.toString(16).padStart(2, '0') + g.toString(16).padStart(2, '0') + b.toString(16).padStart(2, '0');
  }
  function isLight(hex) {
    const c = channels(hex);
    if (!c) return false;
    return (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) > 155;
  }
  function apply(abbrev) {
    const tag = document.getElementById('mgr-teams-data');
    if (!tag) return false;
    let teams;
    try { teams = JSON.parse(tag.textContent); } catch (e) { return false; }
    const want = String(abbrev || '').trim().toUpperCase();
    const row = (teams || []).find(t => String(t.Team || '').toUpperCase() === want);
    if (!row || !row.Color) return false;
    const base = row.Color;
    const light = lightenDarken(base, isLight(base) ? 0.55 : 1.4);
    const valueColor = isLight(base) ? '#0f141b' : '#f1f5f9';
    const root = document.documentElement;
    root.style.setProperty('--accent', light);
    root.style.setProperty('--panel-alt', base);
    root.style.setProperty('--value-text', valueColor);
    root.style.setProperty('--mgr-team', base);
    return true;
  }
  window.mgrApplyTeamTheme = apply;
})();
