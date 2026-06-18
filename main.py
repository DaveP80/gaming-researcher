"""
Chess.com Game History API
==========================

A small FastAPI service that, given a chess.com username, visits every monthly
archive for that player and returns all the game objects it can collect.

Chess.com public API endpoints used:
  - GET /pub/player/{username}/games/archives   -> list of monthly archive URLs
  - GET /pub/player/{username}/games/{YYYY}/{MM} -> games for that month
  - GET /pub/player/{username}                   -> profile (incl. `status`)

Notes:
  * Chess.com requires a descriptive User-Agent header. Requests without one
    are rejected with 403. Set CONTACT below to your real contact info.
  * Archives are fetched concurrently (bounded by a semaphore) so a player with
    many months of history doesn't take forever.

Run:
    pip install fastapi "uvicorn[standard]" httpx
    uvicorn script:app --reload

Then open http://127.0.0.1:8000/docs to try it.
"""

import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path as DiskPath
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import HTMLResponse

CHESS_API_BASE = "https://api.chess.com/pub"

# Chess.com asks API consumers to identify themselves. Replace the email with
# your own; a generic but descriptive User-Agent avoids 403 responses.
CONTACT = "your-email@example.com"
HEADERS = {"User-Agent": f"ChessHistoryAPI/1.0 ({CONTACT})"}

# Cap on how many archives we pull at the same time. Chess.com is fine with
# this; raising it too high risks rate-limit (429) responses.
DEFAULT_CONCURRENCY = 8

# Where saved JSON responses are written. Created on first save.
OUTPUT_DIR = DiskPath("responses")

app = FastAPI(
    title="Chess.com Game History API",
    description="Enter a chess.com username; get back every game from every archive.",
    version="1.0.0",
)


async def fetch_archive_urls(client: httpx.AsyncClient, username: str) -> list[str]:
    """Return the list of monthly archive URLs for a player."""
    url = f"{CHESS_API_BASE}/player/{username}/games/archives"
    resp = await client.get(url)

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Player '{username}' not found.")
    if resp.status_code == 403:
        raise HTTPException(
            status_code=502,
            detail="Chess.com rejected the request (403). Check the User-Agent header.",
        )
    resp.raise_for_status()

    return resp.json().get("archives", [])


async def fetch_games_from_archive(
    client: httpx.AsyncClient,
    archive_url: str,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Fetch all games from a single monthly archive URL."""
    async with semaphore:
        try:
            resp = await client.get(archive_url)
            resp.raise_for_status()
        except httpx.HTTPError:
            # A single bad month shouldn't sink the whole request.
            return []
        return resp.json().get("games", [])


async def collect_all_games(
    client: httpx.AsyncClient,
    username: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch every archive for a player and return (archive_urls, flat_games)."""
    archive_urls = await fetch_archive_urls(client, username)
    tasks = [fetch_games_from_archive(client, url, semaphore) for url in archive_urls]
    per_archive_games = await asyncio.gather(*tasks)
    games = [g for batch in per_archive_games for g in batch]
    return archive_urls, games


def extract_opponents(games: list[dict[str, Any]], username: str) -> Counter:
    """
    Walk every game and tally how many times each opponent was faced.

    Keys are lowercased usernames (chess.com lookups are case-insensitive).
    Bots / accounts missing a username are skipped.
    """
    counts: Counter = Counter()
    for game in games:
        for color in ("white", "black"):
            side = game.get(color) or {}
            name = side.get("username")
            if name and name.lower() != username:
                counts[name.lower()] += 1
    return counts


async def fetch_profile(
    client: httpx.AsyncClient,
    opponent: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Fetch one player's profile and return a trimmed record.

    `status` is the field of interest:
      - "basic" / "premium" / "staff" / "mod"  -> active account
      - "closed"                               -> account closed
      - "closed:fair_play_violations"          -> banned for cheating
      - "not_found"                            -> profile 404 (fully removed)
      - "unknown"                              -> request failed
    """
    async with semaphore:
        try:
            resp = await client.get(f"{CHESS_API_BASE}/player/{opponent}")
        except httpx.HTTPError:
            return {
                "username": opponent,
                "status": "unknown",
                "error": "request_failed",
            }

    if resp.status_code == 404:
        return {"username": opponent, "status": "not_found"}
    if resp.status_code != 200:
        return {
            "username": opponent,
            "status": "unknown",
            "error": f"http_{resp.status_code}",
        }

    data = resp.json()
    # `country` is a URL like ".../country/US"; keep just the code.
    country = (data.get("country") or "").rsplit("/", 1)[-1] or None
    return {
        "username": data.get("username", opponent),
        "status": data.get("status"),
        "name": data.get("name"),
        "country": country,
        "joined": data.get("joined"),
        "last_online": data.get("last_online"),
        "url": data.get("url"),
    }


def save_response(data: dict[str, Any], username: str, endpoint: str) -> str:
    """
    Write a response dict to responses/{username}_{endpoint}_{timestamp}.json
    and return the path as a string. The timestamp keeps repeated runs from
    overwriting each other.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_user = "".join(c for c in username if c.isalnum() or c in "-_") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{safe_user}_{endpoint}_{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chesscom-researcher</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#15181d; --ink-2:#1c2026; --ink-3:#232a31; --line:#2f363f;
    --paper:#ece8df; --muted:#8b929c;
    --green:#7fa650; --green-bright:#9bc26a;
    --danger:#e0524a; --amber:#d99b3c; --stamp:#d6453d;
    --display:'Space Grotesk',system-ui,sans-serif;
    --mono:'Space Mono',ui-monospace,monospace;
    --body:'Inter',system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--ink); color:var(--paper); font-family:var(--body);
    line-height:1.55; -webkit-font-smoothing:antialiased;
    background-image:radial-gradient(circle at 50% -10%, #1d232b 0%, var(--ink) 55%);
    min-height:100vh;
  }
  .wrap{max-width:880px; margin:0 auto; padding:clamp(28px,5vw,72px) clamp(18px,4vw,28px) 96px}
  a{color:inherit}

  /* ---- hero ---- */
  .eyebrow{
    font-family:var(--mono); font-size:13px; letter-spacing:.16em; text-transform:uppercase;
    color:var(--green-bright); margin:0 0 18px;
  }
  .eyebrow .dot{color:var(--muted)}
  h1{
    font-family:var(--display); font-weight:700; letter-spacing:-.02em;
    font-size:clamp(30px,6vw,52px); line-height:1.05; margin:0 0 16px; max-width:16ch;
  }
  .lede{color:var(--muted); font-size:clamp(15px,2.4vw,17px); max-width:54ch; margin:0}

  /* ---- console / form ---- */
  .console{
    margin-top:34px; background:var(--ink-2); border:1px solid var(--line);
    border-radius:14px; padding:clamp(18px,3vw,26px);
  }
  .field-label{
    font-family:var(--mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); display:block; margin:0 0 9px;
  }
  .input-row{display:flex; gap:10px; flex-wrap:wrap}
  .uinput{
    flex:1 1 240px; min-width:0; background:var(--ink); color:var(--paper);
    border:1px solid var(--line); border-radius:10px; padding:14px 15px;
    font-family:var(--mono); font-size:16px; letter-spacing:.01em;
  }
  .uinput::placeholder{color:#5b626c}
  .uinput:focus-visible, .run:focus-visible, .toggle input:focus-visible{
    outline:2px solid var(--green-bright); outline-offset:2px;
  }
  .run{
    flex:0 0 auto; background:var(--green); color:#10140c; border:none; cursor:pointer;
    border-radius:10px; padding:0 22px; font-family:var(--display); font-weight:700;
    font-size:15px; letter-spacing:.01em;
  }
  .run:hover{background:var(--green-bright)}
  .run:disabled{opacity:.5; cursor:progress}
  .toggle{
    display:flex; align-items:center; gap:10px; margin-top:16px; cursor:pointer;
    width:fit-content; color:var(--paper);
  }
  .toggle input{width:18px; height:18px; accent-color:var(--danger); cursor:pointer}
  .toggle span{font-size:14px}
  .toggle small{display:block; color:var(--muted); font-size:12.5px}

  /* ---- status line ---- */
  .status{margin-top:22px; min-height:22px; font-family:var(--mono); font-size:13.5px; color:var(--muted)}
  .status.err{color:var(--danger)}
  .spinner{
    display:inline-block; width:13px; height:13px; vertical-align:-1px; margin-right:8px;
    border:2px solid var(--line); border-top-color:var(--green-bright); border-radius:50%;
    animation:spin .8s linear infinite;
  }
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ---- results ---- */
  .results{margin-top:30px; display:none}
  .results.show{display:block}

  .summary-wrap{position:relative; margin-top:2px}
  .summary{
    display:grid; gap:1px; background:var(--line);
    border:1px solid var(--line); border-radius:14px; overflow:hidden;
    grid-template-columns:repeat(4,1fr);
  }
  .stat{background:var(--ink-2); padding:18px 16px}
  .stat .num{font-family:var(--display); font-weight:700; font-size:clamp(22px,4vw,30px); line-height:1}
  .stat .lbl{font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-top:8px}
  .stat.flag .num{color:var(--danger)}

  /* signature: case-file stamp, used once when cheaters are found */
  .stamp{
    position:absolute; top:-13px; right:14px; transform:rotate(-8deg);
    font-family:var(--mono); font-weight:700; font-size:11px; letter-spacing:.18em;
    text-transform:uppercase; color:var(--stamp); border:2px solid var(--stamp);
    border-radius:6px; padding:6px 10px; opacity:.95; pointer-events:none;
    background:var(--ink); box-shadow:inset 0 0 0 2px rgba(214,69,61,.18);
  }

  .tabs{display:flex; gap:8px; flex-wrap:wrap; margin:24px 0 4px}
  .tab{
    background:transparent; border:1px solid var(--line); color:var(--muted);
    font-family:var(--mono); font-size:12.5px; letter-spacing:.05em; cursor:pointer;
    padding:8px 13px; border-radius:999px;
  }
  .tab:hover{color:var(--paper)}
  .tab.active{background:var(--paper); color:var(--ink); border-color:var(--paper)}
  .tab .c{opacity:.7}

  .list{margin-top:14px; border:1px solid var(--line); border-radius:14px; overflow:hidden}
  .rec{
    display:grid; grid-template-columns:64px 1fr auto; gap:14px; align-items:center;
    padding:13px 16px; background:var(--ink-2); border-top:1px solid var(--line);
  }
  .rec:first-child{border-top:none}
  .rec .gp{font-family:var(--mono); font-size:13px; color:var(--muted)}
  .rec .gp b{display:block; font-size:18px; color:var(--paper); font-weight:700}
  .rec .who{min-width:0}
  .rec .who a{font-family:var(--display); font-weight:600; font-size:16px; text-decoration:none}
  .rec .who a:hover{color:var(--green-bright)}
  .rec .who .meta{font-size:12.5px; color:var(--muted); margin-top:2px}
  .badge{
    font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase;
    padding:5px 10px; border-radius:999px; white-space:nowrap; border:1px solid transparent;
  }
  .badge.active{color:var(--green-bright); border-color:rgba(155,194,106,.35); background:rgba(155,194,106,.08)}
  .badge.banned{color:var(--danger); border-color:rgba(224,82,74,.4); background:rgba(224,82,74,.1)}
  .badge.closed{color:var(--amber); border-color:rgba(217,155,60,.35); background:rgba(217,155,60,.1)}
  .badge.gone{color:var(--muted); border-color:var(--line)}

  .empty{padding:30px 18px; text-align:center; color:var(--muted); font-size:14px; background:var(--ink-2)}
  .empty b{color:var(--paper); font-family:var(--display)}

  footer{margin-top:40px; color:var(--muted); font-family:var(--mono); font-size:12px}
  footer a{color:var(--green-bright)}

  @media (max-width:560px){
    .summary{grid-template-columns:repeat(2,1fr)}
    .rec{grid-template-columns:52px 1fr; row-gap:8px}
    .rec .badge{grid-column:2}
  }
  @media (prefers-reduced-motion:reduce){
    .spinner{animation:none}
    *{transition:none!important}
  }
</style>
</head>
<body>
  <main class="wrap">
    <p class="eyebrow">chesscom<span class="dot">&#8209;</span>researcher</p>
    <h1>Who did you play that got caught cheating?</h1>
    <p class="lede">Pull every opponent from a player's chess.com archives, then check each
       account's current standing &mdash; and surface the ones banned for fair&#8209;play violations.</p>

    <section class="console" aria-label="Search">
      <label class="field-label" for="username">Chess.com username</label>
      <div class="input-row">
        <input id="username" class="uinput" type="text" autocomplete="off" spellcheck="false"
               placeholder="e.g. hikaru" aria-label="Chess.com username">
        <button id="run" class="run" type="button">Run search</button>
      </div>
      <label class="toggle" for="bannedOnly">
        <input id="bannedOnly" type="checkbox">
        <span>Check banned accounts
          <small>Jump straight to opponents banned for cheating</small></span>
      </label>
    </section>

    <p id="status" class="status" role="status" aria-live="polite"></p>

    <section id="results" class="results" aria-live="polite">
      <div class="summary-wrap" id="summaryWrap">
        <div class="summary" id="summary"></div>
      </div>
      <div class="tabs" id="tabs"></div>
      <div class="list" id="list"></div>
    </section>

    <footer>
      Data from the public chess.com API. JSON endpoints: <a href="/docs">/docs</a>
    </footer>
  </main>

<script>
  var input   = document.getElementById('username');
  var runBtn  = document.getElementById('run');
  var bannedOnly = document.getElementById('bannedOnly');
  var statusEl = document.getElementById('status');
  var results = document.getElementById('results');
  var summaryEl = document.getElementById('summary');
  var tabsEl  = document.getElementById('tabs');
  var listEl  = document.getElementById('list');

  var current = null;   // last dataset {all, banned, closed}
  var activeTab = 'all';

  function classify(status){
    if(status === 'closed:fair_play_violations') return 'banned';
    if(typeof status === 'string' && status.indexOf('closed') === 0) return 'closed';
    if(status === 'not_found') return 'gone';
    if(!status || status === 'unknown') return 'unknown';
    return 'active';
  }
  function statusLabel(kind){
    return {banned:'Banned \\u2014 fair play', closed:'Closed', gone:'Removed',
            unknown:'Unknown', active:'Active'}[kind];
  }
  function flag(code){
    if(!code || code.length !== 2) return '';
    var A = 0x1F1E6, base = 'A'.charCodeAt(0);
    return String.fromCodePoint(A + code.toUpperCase().charCodeAt(0) - base) +
           String.fromCodePoint(A + code.toUpperCase().charCodeAt(1) - base);
  }
  function esc(s){
    return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
    });
  }

  function rowHTML(p){
    var kind = classify(p.status);
    var url = p.url || ('https://www.chess.com/member/' + encodeURIComponent(p.username));
    var f = flag(p.country);
    var meta = [];
    if(f) meta.push(f + ' ' + esc(p.country));
    if(p.name) meta.push(esc(p.name));
    return '<div class="rec">' +
      '<div class="gp"><b>' + (p.games_played || 0) + '</b>games</div>' +
      '<div class="who"><a href="' + esc(url) + '" target="_blank" rel="noopener">' +
        esc(p.username) + '</a>' +
        (meta.length ? '<div class="meta">' + meta.join(' &middot; ') + '</div>' : '') +
      '</div>' +
      '<span class="badge ' + kind + '">' + statusLabel(kind) + '</span>' +
    '</div>';
  }

  var EMPTY = {
    all:   '<div class="empty"><b>No opponents found.</b><br>This player has no public games in their archives.</div>',
    banned:'<div class="empty"><b>No cheaters here.</b><br>None of the opponents you faced are banned for fair&#8209;play violations.</div>',
    closed:'<div class="empty"><b>No closed accounts.</b><br>Every opponent still has an open account.</div>'
  };

  function renderList(){
    var rows = current[activeTab] || [];
    listEl.innerHTML = rows.length ? rows.map(rowHTML).join('') : EMPTY[activeTab];
  }

  function renderTabs(){
    var defs = [
      ['all','All opponents', current.all.length],
      ['banned','Banned for cheating', current.banned.length],
      ['closed','Closed accounts', current.closed.length]
    ];
    tabsEl.innerHTML = defs.map(function(d){
      return '<button class="tab' + (d[0]===activeTab?' active':'') + '" data-tab="' + d[0] + '">' +
        d[1] + ' <span class="c">' + d[2] + '</span></button>';
    }).join('');
    Array.prototype.forEach.call(tabsEl.querySelectorAll('.tab'), function(btn){
      btn.addEventListener('click', function(){
        activeTab = btn.getAttribute('data-tab');
        renderTabs(); renderList();
      });
    });
  }

  function renderSummary(d){
    var banned = current.banned.length;
    summaryEl.innerHTML =
      '<div class="stat"><div class="num">' + d.total_games + '</div><div class="lbl">Games</div></div>' +
      '<div class="stat"><div class="num">' + d.unique_opponents + '</div><div class="lbl">Opponents</div></div>' +
      '<div class="stat ' + (banned?'flag':'') + '"><div class="num">' + banned + '</div><div class="lbl">Cheaters faced</div></div>' +
      '<div class="stat"><div class="num">' + current.closed.length + '</div><div class="lbl">Closed</div></div>';
    var wrap = document.getElementById('summaryWrap');
    var old = wrap.querySelector('.stamp');
    if(old) old.remove();
    if(banned){
      var stamp = document.createElement('div');
      stamp.className = 'stamp';
      stamp.innerHTML = banned + ' fair&#8209;play ' + (banned === 1 ? 'ban' : 'bans');
      wrap.appendChild(stamp);
    }
  }

  function render(d){
    var all = (d.opponents || []).slice();
    current = {
      all: all,
      banned: all.filter(function(p){ return classify(p.status) === 'banned'; }),
      closed: all.filter(function(p){ return classify(p.status) === 'closed'; })
    };
    activeTab = bannedOnly.checked ? 'banned' : 'all';
    renderSummary(d); renderTabs(); renderList();
    results.classList.add('show');
  }

  function setBusy(on, msg){
    runBtn.disabled = on;
    runBtn.textContent = on ? 'Searching\\u2026' : 'Run search';
    statusEl.className = 'status';
    statusEl.innerHTML = on ? '<span class="spinner"></span>' + msg : (msg || '');
  }

  async function search(){
    var name = input.value.trim();
    if(!name){ statusEl.className='status err'; statusEl.textContent='Enter a username first.'; input.focus(); return; }

    results.classList.remove('show');
    var t0 = Date.now();
    setBusy(true, 'Reading archives and checking accounts\\u2026');
    var tick = setInterval(function(){
      var s = Math.round((Date.now()-t0)/1000);
      statusEl.innerHTML = '<span class="spinner"></span>Reading archives and checking accounts\\u2026 ' + s + 's';
    }, 1000);

    try{
      var resp = await fetch('/opponents/' + encodeURIComponent(name) + '?save=false&concurrency=12');
      clearInterval(tick);
      if(!resp.ok){
        var detail = '';
        try{ detail = (await resp.json()).detail; }catch(e){}
        if(resp.status === 404) detail = detail || ('No chess.com player named "' + name + '".');
        setBusy(false, '');
        statusEl.className='status err';
        statusEl.textContent = detail || ('Search failed (HTTP ' + resp.status + ').');
        return;
      }
      var data = await resp.json();
      setBusy(false, '');
      statusEl.textContent = 'Checked ' + data.unique_opponents + ' opponents across ' +
                             data.total_games + ' games in ' + Math.round((Date.now()-t0)/1000) + 's.';
      render(data);
    }catch(err){
      clearInterval(tick);
      setBusy(false, '');
      statusEl.className='status err';
      statusEl.textContent = 'Could not reach the server. Is it still running?';
    }
  }

  runBtn.addEventListener('click', search);
  input.addEventListener('keydown', function(e){ if(e.key === 'Enter') search(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def homepage() -> HTMLResponse:
    """Serve the chesscom-researcher single-page front end."""
    return HTMLResponse(INDEX_HTML)


@app.get("/api")
def api_info() -> dict[str, Any]:
    """Machine-readable index of the JSON endpoints (the UI lives at '/')."""
    return {
        "service": "Chess.com Game History API",
        "endpoints": {
            "all_games": "GET /games/{username}",
            "research_opponents": "GET /opponents/{username}",
        },
        "docs": "/docs",
    }


@app.get("/games/{username}")
async def get_all_games(
    username: str = Path(..., description="A chess.com username (case-insensitive)."),
    concurrency: int = Query(
        DEFAULT_CONCURRENCY,
        ge=1,
        le=20,
        description="How many archives to fetch concurrently.",
    ),
    include_games: bool = Query(
        True,
        description="Set false to get only counts/metadata without the (potentially large) game list.",
    ),
    save: bool = Query(
        True,
        description="Write the JSON response to a file under responses/ for later study.",
    ),
) -> dict[str, Any]:
    """
    Visit every monthly archive for `username` and return all collected games.

    The response can be large for active players (thousands of games / many MB).
    Use `include_games=false` if you only want the counts.
    """
    username = username.strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="Username must not be empty.")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        archive_urls, games = await collect_all_games(client, username, semaphore)

    response: dict[str, Any] = {
        "username": username,
        "archive_count": len(archive_urls),
        "game_count": len(games),
    }
    if include_games:
        response["games"] = games

    if save:
        response["saved_to"] = save_response(response, username, "games")

    return response


@app.get("/opponents/{username}")
async def research_opponents(
    username: str = Path(..., description="A chess.com username (case-insensitive)."),
    concurrency: int = Query(
        DEFAULT_CONCURRENCY,
        ge=1,
        le=20,
        description="How many requests to run concurrently.",
    ),
    only_closed: bool = Query(
        False,
        description="If true, the `opponents` list contains only closed/banned/removed accounts.",
    ),
    save: bool = Query(
        True,
        description="Write the JSON response to a file under responses/ for later study.",
    ),
) -> dict[str, Any]:
    """
    Research every opponent `username` has ever played.

    Collects all games, dedupes opponents (with a per-opponent game count),
    then fetches each profile to read its `status`. The response always
    surfaces the closed and cheating-banned accounts separately, since those
    are usually the point of the search.

    Note: this makes one request per unique opponent. A player with hundreds of
    distinct opponents means hundreds of profile lookups, so allow a little time.
    """
    username = username.strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="Username must not be empty.")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        _, games = await collect_all_games(client, username, semaphore)
        opponent_counts = extract_opponents(games, username)

        profiles = await asyncio.gather(
            *(fetch_profile(client, opp, semaphore) for opp in opponent_counts)
        )

    # Attach how many times each opponent was faced, then sort by that.
    for profile in profiles:
        key = profile["username"].lower()
        profile["games_played"] = opponent_counts.get(key, 0)
    profiles.sort(key=lambda p: p["games_played"], reverse=True)

    def has_status(profile: dict[str, Any], prefix: str) -> bool:
        return (profile.get("status") or "").startswith(prefix)

    closed = [p for p in profiles if has_status(p, "closed")]
    banned = [p for p in profiles if p.get("status") == "closed:fair_play_violations"]

    response = {
        "username": username,
        "total_games": len(games),
        "unique_opponents": len(opponent_counts),
        "closed_count": len(closed),
        "banned_for_cheating_count": len(banned),
        "closed_accounts": closed,
        "opponents": closed if only_closed else profiles,
    }

    if save:
        response["saved_to"] = save_response(response, username, "opponents")

    return response
