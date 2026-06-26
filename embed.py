import json
import math
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def _flatten_etymology(ety):
    # Etymology is a clean string now; guard against legacy non-string entries.
    if not ety:
        return ""
    if isinstance(ety, str):
        return ety
    if isinstance(ety, list):
        return "; ".join(_flatten_etymology(x) for x in ety)
    return str(ety)

CACHE_FILE = "words.json"
EMBED_CACHE = "embeddings.npy"
OUT_FILE = "viz.html"
N_CLUSTERS = 8

with open(CACHE_FILE) as f:
    cache = json.load(f)

words = list(cache.keys())
def _first_def(v):
    d = v.get("definition") or v.get("definitions") or []
    return d[0] if d else ""

texts = [f"{w}: {_first_def(cache[w])}" for w in words]

if os.path.exists(EMBED_CACHE):
    stored = np.load(EMBED_CACHE, allow_pickle=True).item()
    if stored.get("words") == words:
        print(f"Loaded cached embeddings for {len(words)} words")
        embeddings = stored["embeddings"]
    else:
        print(f"Word list changed, re-embedding {len(words)} words...")
        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, show_progress_bar=True)
        np.save(EMBED_CACHE, {"words": words, "embeddings": embeddings})
else:
    print(f"Embedding {len(words)} words...")
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, show_progress_bar=True)
    np.save(EMBED_CACHE, {"words": words, "embeddings": embeddings})

print("Running t-SNE to 2D...")
coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

print("Clustering...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto").fit(embeddings)
labels = km.labels_

# palette tuned so black bubble text clears WCAG AA
COLORS = ["#f0907e", "#f0a95a", "#ecd45e", "#7fd093", "#6fcabf", "#84b3e6", "#b89ae4", "#f094c2"]

cluster_names = {}
for ci in range(N_CLUSTERS):
    idxs = np.where(labels == ci)[0]
    dists = np.linalg.norm(embeddings[idxs] - km.cluster_centers_[ci], axis=1)
    top = [words[idxs[j]] for j in np.argsort(dists)[:4]]
    cluster_names[ci] = ", ".join(top)

# normalise t-SNE to -1..1
mn, mx = coords.min(axis=0), coords.max(axis=0)
norm = (coords - mn) / (mx - mn) * 2 - 1

data = []
for i, word in enumerate(words):
    entry = cache[word]
    ci = int(labels[i])
    defs = entry.get("definitions") or entry.get("definition") or []
    exs  = entry.get("examples")   or ([entry["example"]]    if entry.get("example")    else [])
    syns = entry.get("synonyms", [])
    ants = entry.get("antonyms", [])
    data.append({
        "word": word,
        "x": float(norm[i, 0]),
        "y": float(norm[i, 1]),
        "color": COLORS[ci],
        "cluster": cluster_names[ci],
        "phonetic":    entry.get("pronunciation", entry.get("phonetic", "")),
        "pos":         entry.get("part_of_speech", []),
        "definitions": defs,
        "examples":    exs,
        "synonyms":    syns if isinstance(syns, list) else syns.split(", "),
        "antonyms":    ants if isinstance(ants, list) else ants.split(", "),
        "etymology":   _flatten_etymology(entry.get("etymology", "")),
    })

DATA_JSON = json.dumps(data)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Word Map</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3CradialGradient id='g' cx='40%25' cy='35%25' r='60%25'%3E%3Cstop offset='0%25' stop-color='%23c084fc'/%3E%3Cstop offset='55%25' stop-color='%237c3aed'/%3E%3Cstop offset='100%25' stop-color='%234c1d95'/%3E%3C/radialGradient%3E%3C/defs%3E%3Ccircle cx='16' cy='16' r='14' fill='url(%23g)'/%3E%3C/svg%3E">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a0a; font-family: Georgia, serif; overflow: hidden; color: #fff; }}

#search-wrap {{
  position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
  display: flex; align-items: center; gap: 8px;
  background: #1c1c1e; border: 1px solid #2c2c2e;
  border-radius: 999px; padding: 8px 16px;
  width: min(240px, calc(100vw - 100px)); z-index: 30;
  box-shadow: 0 2px 8px rgba(0,0,0,0.5);
}}
#search-wrap svg {{ opacity: 0.5; flex-shrink: 0; color: #aaa; }}
#search {{ border: none; outline: none; font-size: 14px; color: #f0f0f0; background: transparent; width: 100%; font-family: Georgia, serif; -webkit-appearance: none; }}
#search::placeholder {{ color: #555; }}

#count {{
  position: fixed; top: 16px; left: 16px;
  font-size: 12px; color: #888; z-index: 30;
  background: #1c1c1e; border: 1px solid #2c2c2e;
  border-radius: 999px; padding: 6px 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.5);
}}

#panel {{
  position: fixed; bottom: 0; left: 50%; transform: translate(-50%, 100%);
  width: min(480px, 100vw); max-height: 72vh;
  background: #141414; border-radius: 16px 16px 0 0;
  border: 1px solid rgba(255,255,255,0.07); border-bottom: none;
  padding: 16px 24px 52px; overflow-y: auto; z-index: 40;
  transition: transform 0.28s cubic-bezier(0.32, 0.72, 0, 1);
  -webkit-overflow-scrolling: touch;
}}
#panel.open {{ transform: translate(-50%, 0); }}
@media (prefers-reduced-motion: reduce) {{ #panel {{ transition: none; }} }}
@media (max-width: 500px) {{
  #p-word {{ font-size: 20px !important; }}
  #panel {{ padding: 16px 18px 56px; }}
}}
.handle {{ width: 32px; height: 3px; border-radius: 2px; background: rgba(255,255,255,0.12); margin: 0 auto 20px; }}
#p-word {{ font-size: 24px; font-weight: 700; color: #f5f1ea; margin-bottom: 2px; }}
#p-phonetic {{ font-size: 12px; color: rgba(255,255,255,0.5); margin-bottom: 14px; }}
#p-defs {{ font-size: 14px; line-height: 1.75; color: #cfcfcf; margin-bottom: 12px; }}
#p-defs .def {{ margin-bottom: 6px; }}
#p-examples {{ font-size: 13px; font-style: italic; color: rgba(255,255,255,0.55); margin-bottom: 16px; line-height: 1.6; }}
.field-label {{ font-size: 11px; color: rgba(255,255,255,0.4); margin-bottom: 4px; font-style: italic; }}
#p-synonyms {{ font-size: 13px; color: rgba(255,255,255,0.72); margin-bottom: 14px; line-height: 1.6; }}
#p-antonyms {{ font-size: 13px; color: rgba(255,255,255,0.72); margin-bottom: 16px; line-height: 1.6; }}
#p-etymology {{ font-size: 13px; font-style: italic; color: rgba(255,255,255,0.5); margin-bottom: 12px; line-height: 1.6; }}
#close {{ position: absolute; top: 14px; right: 18px; background: none; border: none; color: rgba(255,255,255,0.2); cursor: pointer; font-size: 16px; font-family: sans-serif; }}
#theme-btn {{ position: absolute; top: 14px; right: 44px; background: none; border: none; color: rgba(255,255,255,0.25); cursor: pointer; width: 18px; height: 18px; padding: 0; display: flex; align-items: center; }}

#panel.light {{ background: #fefefe; border-color: rgba(0,0,0,0.1); }}
#panel.light #p-word {{ color: #0a0a0a; }}
#panel.light #p-phonetic {{ color: #666; }}
#panel.light #p-defs {{ color: #222; }}
#panel.light #p-examples {{ color: #555; }}
#panel.light #p-synonyms, #panel.light #p-antonyms {{ color: #333; }}
#panel.light #p-etymology {{ color: #555; }}
#panel.light .field-label {{ color: #999; }}
#panel.light .handle {{ background: rgba(0,0,0,0.15); }}
#panel.light #close, #panel.light #theme-btn {{ color: #999; }}

path.blob {{ cursor: pointer; }}
path.blob.dimmed {{ opacity: 0.07; }}
text.lbl {{ pointer-events: none; dominant-baseline: middle; text-anchor: middle; font-family: Georgia, serif; }}
</style>
</head>
<body>
<svg id="svg" style="position:fixed;inset:0;width:100%;height:100%"></svg>

<div id="count">{len(data)} words</div>
<div id="search-wrap">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
  <input id="search" type="text" placeholder="Search words...">
</div>

<div id="panel">
  <div class="handle"></div>
  <button id="close">✕</button>
  <button id="theme-btn">
    <svg id="icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    <svg id="icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
  </button>
  <div id="p-word"></div>
  <div id="p-phonetic"></div>
  <div id="p-defs"></div>
  <div id="p-examples"></div>
  <div class="field-label" id="lbl-syn" style="display:none">synonyms</div>
  <div id="p-synonyms"></div>
  <div class="field-label" id="lbl-ant" style="display:none">antonyms</div>
  <div id="p-antonyms"></div>
  <div class="field-label" id="lbl-ety" style="display:none">etymology</div>
  <div id="p-etymology"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
const DATA = {DATA_JSON};

const PAD = 0.07;
function toScreen(nx, ny) {{
  const w = window.innerWidth, h = window.innerHeight;
  return [w/2 + nx * w * (0.5 - PAD), h/2 + ny * h * (0.5 - PAD)];
}}

// measure actual text width so long phrases fit
const _mc = document.createElement('canvas').getContext('2d');
_mc.font = '600 11px Georgia, serif';
const R = d => Math.max(28, _mc.measureText(d.word).width / 2 + 16);

const nodes = DATA.map(d => {{
  const [x, y] = toScreen(d.x, d.y);
  // home = the t-SNE position; bubbles are tethered here so they never drift off
  return {{ ...d, x, y, hx: x, hy: y, r: R(d), seed: (d.word.charCodeAt(0) / 90) * Math.PI * 2 }};
}});

const sim = d3.forceSimulation(nodes)
  .force('collide', d3.forceCollide(d => d.r + 4).strength(0.85).iterations(3))
  .force('x', d3.forceX(d => d.hx).strength(0.05))
  .force('y', d3.forceY(d => d.hy).strength(0.05))
  .alphaDecay(0.014);

const svg = d3.select('#svg');
const g = svg.append('g');
const blobG = g.append('g');
const textG = g.append('g');

function blobPath(cx, cy, r, t, seed) {{
  // 8 control points + Catmull-Rom bezier = smooth organic near-circle
  const N = 8;
  const pts = Array.from({{length: N}}, (_, i) => {{
    const a = (i / N) * Math.PI * 2;
    const w = 1
      + 0.018 * Math.sin(2*a + t*0.9 + seed)
      + 0.010 * Math.cos(3*a - t*0.7 + seed + 1.2);
    return [cx + r*w*Math.cos(a), cy + r*w*Math.sin(a)];
  }});
  let d = `M ${{pts[0][0].toFixed(2)}},${{pts[0][1].toFixed(2)}}`;
  for (let i = 0; i < N; i++) {{
    const p0 = pts[(i-1+N)%N], p1 = pts[i], p2 = pts[(i+1)%N], p3 = pts[(i+2)%N];
    const c1x = p1[0] + (p2[0]-p0[0])/6, c1y = p1[1] + (p2[1]-p0[1])/6;
    const c2x = p2[0] - (p3[0]-p1[0])/6, c2y = p2[1] - (p3[1]-p1[1])/6;
    d += ` C ${{c1x.toFixed(2)}},${{c1y.toFixed(2)}} ${{c2x.toFixed(2)}},${{c2y.toFixed(2)}} ${{p2[0].toFixed(2)}},${{p2[1].toFixed(2)}}`;
  }}
  return d + ' Z';
}}

const blobs = blobG.selectAll('path').data(nodes).enter().append('path')
  .attr('class', 'blob')
  .attr('fill', d => d.color)
  .attr('fill-opacity', 0.94)
  .on('click', (e, d) => {{ e.stopPropagation(); showPanel(d); }});

const labels = textG.selectAll('text').data(nodes).enter().append('text')
  .attr('class', 'lbl')
  .style('font-size', d => Math.max(8, Math.min(12, d.r * 0.36)) + 'px')
  .style('font-weight', '600')
  .style('fill', '#000')
  .text(d => d.word);

// fixed organic shape per bubble (phase = its seed)
function draw() {{
  blobs.attr('d', d => blobPath(d.x, d.y, d.r, d.seed, d.seed));
  labels.attr('x', d => d.x).attr('y', d => d.y);
}}
// settle collisions off-screen, then paint once (static)
sim.stop();
for (let i = 0; i < 300; i++) sim.tick();
draw();

svg.call(d3.zoom().scaleExtent([0.25, 5]).on('zoom', e => g.attr('transform', e.transform)));

function set(id, txt) {{ document.getElementById(id).textContent = txt; }}
function show(id, visible) {{ document.getElementById(id).style.display = visible ? '' : 'none'; }}

function showPanel(d) {{
  set('p-word', d.word);
  set('p-phonetic', [d.phonetic, (d.pos || []).join(', ')].filter(Boolean).join('  ·  '));
  document.getElementById('p-defs').innerHTML =
    (d.definitions||[]).map((s,i) => `<div class="def">${{i+1}}. ${{s.replace(/^\\([^)]+\\)\\s*/,'')}}</div>`).join('') || '';
  document.getElementById('p-examples').innerHTML =
    (d.examples||[]).map(s => `<div>"${{s}}"</div>`).join('') || '';
  const syns = d.synonyms || [];
  const ants = d.antonyms || [];
  show('lbl-syn', syns.length > 0); set('p-synonyms', syns.join(', '));
  show('lbl-ant', ants.length > 0); set('p-antonyms', ants.join(', '));
  show('lbl-ety', !!d.etymology); set('p-etymology', d.etymology || '');
  document.getElementById('panel').classList.add('open');
}}
const panel = document.getElementById('panel');
panel.addEventListener('click', e => e.stopPropagation());
document.getElementById('close').onclick = () => panel.classList.remove('open');
document.getElementById('theme-btn').onclick = () => {{
  const isLight = panel.classList.toggle('light');
  document.getElementById('icon-moon').style.display = isLight ? 'none' : '';
  document.getElementById('icon-sun').style.display = isLight ? '' : 'none';
}};

// swipe down to close
let _ty0 = 0;
panel.addEventListener('touchstart', e => {{ _ty0 = e.touches[0].clientY; }}, {{passive: true}});
panel.addEventListener('touchend', e => {{
  if (e.changedTouches[0].clientY - _ty0 > 60) panel.classList.remove('open');
}}, {{passive: true}});
const countEl = document.getElementById('count');
document.getElementById('search').addEventListener('input', e => {{
  const q = e.target.value.toLowerCase();
  let n = 0;
  blobs.classed('dimmed', d => {{ const m = !q || d.word.toLowerCase().includes(q); if(m)n++; return !m; }});
  labels.style('opacity', d => (!q || d.word.toLowerCase().includes(q)) ? 1 : 0.05);
  countEl.textContent = q ? `${{n}}/${{DATA.length}}` : `${{DATA.length}} words`;
}});
</script>
</body>
</html>"""

with open(OUT_FILE, "w") as f:
    f.write(HTML)
print(f"Written to {OUT_FILE}")
