# -*- coding: utf-8 -*-
"""Stylometric analysis of 红楼梦 (Dream of the Red Chamber).

Splits the novel into 120 chapters, tokenizes with jieba, finds the 300 most
common words, z-scores word frequencies per chapter, projects chapters into 3D
with PCA (chapters 1-80 in red, 81-120 in blue), and saves a PNG plot plus an
interactive HTML 3D viewer.
"""

import json
import math
import re

import jieba
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

import qhchina
qhchina.load_fonts()

TEXT_FILE = "hongloumeng.txt"
PNG_FILE = "hongloumeng_pca.png"
HTML_FILE = "hongloumeng_pca.html"
TOP_N = 300

CHAPTER_RE = re.compile(r"^\s*(第[零一二三四五六七八九十百两]{1,7}回)[\s\u3000]+(\S.*)$", re.M)


def load_chapters(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    matches = list(CHAPTER_RE.finditer(text))
    if len(matches) != 120:
        raise ValueError(f"Expected 120 chapters, found {len(matches)}")
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append({"title": m.group(1) + "　" + m.group(2), "text": text[start:end]})
    return chapters


def tokenize(chapters):
    tokenized = []
    for ch in chapters:
        words = [w for w in jieba.cut(ch["text"]) if len(w) >= 2 and re.search(r"[\u4e00-\u9fff]", w)]
        tokenized.append(words)
    return tokenized


def top_words(tokenized, n):
    from collections import Counter
    counter = Counter()
    for words in tokenized:
        counter.update(words)
    return [w for w, _ in counter.most_common(n)]


def frequency_matrix(tokenized, vocab):
    vocab_index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(tokenized), len(vocab)))
    for r, words in enumerate(tokenized):
        for w in words:
            i = vocab_index.get(w)
            if i is not None:
                counts[r, i] += 1
    freqs = counts / counts.sum(axis=1, keepdims=True)
    return freqs


def zscores(freqs):
    mean = freqs.mean(axis=0)
    std = freqs.std(axis=0)
    std[std == 0] = 1.0
    return (freqs - mean) / std


def group_features(Z, vocab, n=20):
    """Top positive/negative features (mean z-score) per group."""
    first = Z[:80].mean(axis=0)
    last = Z[80:].mean(axis=0)
    order_first = np.argsort(first)
    order_last = np.argsort(last)
    return {
        "first80": {
            "positive": [(vocab[i], round(float(first[i]), 3)) for i in order_first[::-1][:n]],
            "negative": [(vocab[i], round(float(first[i]), 3)) for i in order_first[:n]],
        },
        "last40": {
            "positive": [(vocab[i], round(float(last[i]), 3)) for i in order_last[::-1][:n]],
            "negative": [(vocab[i], round(float(last[i]), 3)) for i in order_last[:n]],
        },
    }


def plot_png(coords, chapters):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    red = coords[:80]
    blue = coords[80:]
    ax.scatter(red[:, 0], red[:, 1], red[:, 2], c="red", s=30, alpha=0.8, label="Chapters 1–80")
    ax.scatter(blue[:, 0], blue[:, 1], blue[:, 2], c="blue", s=30, alpha=0.8, label="Chapters 81–120")
    for i, ch in enumerate(chapters):
        if (i + 1) in (1, 80, 81, 120):
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2], str(i + 1), fontsize=8)
    ax.set_title("红楼梦 — PCA of chapter word frequencies (z-scores, top 300 words)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    fig.savefig(PNG_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>红楼梦 — 3D PCA of Chapter Word Frequencies</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
         background: #14161a; color: #e8e8e8; display: flex; flex-wrap: wrap; }
  #viewer { position: relative; flex: 1 1 640px; min-width: 480px; height: 100vh; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
  canvas:active { cursor: grabbing; }
  #hud { position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,.55);
         padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.5; }
  #hud b { color: #ff6b6b; } #hud span.b { color: #6ba8ff; }
  #legend { position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,.55);
            padding: 8px 12px; border-radius: 8px; font-size: 13px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  #tooltip { position: absolute; pointer-events: none; background: rgba(0,0,0,.85);
             border: 1px solid #444; border-radius: 6px; padding: 6px 10px; font-size: 12px;
             display: none; max-width: 320px; z-index: 5; }
  #panel { flex: 0 0 380px; max-width: 420px; height: 100vh; overflow-y: auto;
           padding: 18px 20px; background: #1c1f26; border-left: 1px solid #2b2f38; }
  #panel h1 { font-size: 17px; margin: 4px 0 2px; }
  #panel h2 { font-size: 14px; margin: 18px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #333; }
  #panel h3 { font-size: 13px; margin: 12px 0 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 14px; }
  ul { list-style: none; margin: 0; padding: 0; font-size: 12.5px; }
  li { display: flex; justify-content: space-between; padding: 2px 4px; border-radius: 4px; }
  li:nth-child(odd) { background: rgba(255,255,255,.04); }
  .pos li .w { color: #7ddc8a; } .neg li .w { color: #ff8b8b; }
  .v { color: #9aa3b2; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div id="viewer">
  <canvas id="c"></canvas>
  <div id="hud"><b>Drag</b> rotate &nbsp;·&nbsp; <span class="b">Wheel</span> zoom &nbsp;·&nbsp;
       <b>Shift+drag</b> pan &nbsp;·&nbsp; hover a point for its chapter</div>
  <div id="legend">
    <div><span class="dot" style="background:#e03131"></span>Chapters 1–80 (first 80)</div>
    <div><span class="dot" style="background:#339af0"></span>Chapters 81–120 (last 40)</div>
  </div>
  <div id="tooltip"></div>
</div>
<div id="panel">
  <h1>红楼梦 — PCA of chapter word frequencies</h1>
  <div style="font-size:12px;color:#9aa3b2">300 most common words · z-scored · PCA to 3D ·
    PC1 <span id="ev1"></span>% · PC2 <span id="ev2"></span>% · PC3 <span id="ev3"></span>% of variance</div>

  <h2>Chapters 1–80 (first 80)</h2>
  <h3>Top 20 positive features (overrepresented)</h3>
  <div class="grid"><ul class="pos" id="f80pos1"></ul><ul class="pos" id="f80pos2"></ul></div>
  <h3>Top 20 negative features (underrepresented)</h3>
  <div class="grid"><ul class="neg" id="f80neg1"></ul><ul class="neg" id="f80neg2"></ul></div>

  <h2>Chapters 81–120 (last 40)</h2>
  <h3>Top 20 positive features (overrepresented)</h3>
  <div class="grid"><ul class="pos" id="f40pos1"></ul><ul class="pos" id="f40pos2"></ul></div>
  <h3>Top 20 negative features (underrepresented)</h3>
  <div class="grid"><ul class="neg" id="f40neg1"></ul><ul class="neg" id="f40neg2"></ul></div>
</div>
<script>
const DATA = __DATA__;

function fillList(id, items) {
  const ul = document.getElementById(id);
  items.forEach(([w, v]) => {
    const li = document.createElement("li");
    const s = document.createElement("span"); s.className = "w"; s.textContent = w;
    const n = document.createElement("span"); n.className = "v"; n.textContent = v.toFixed(2);
    li.append(s, n); ul.appendChild(li);
  });
}
["f80pos","f80neg","f40pos","f40neg"].forEach(base => {
  const feats = DATA.features[base];
  const half = Math.ceil(feats.length / 2);
  fillList(base + "1", feats.slice(0, half));
  fillList(base + "2", feats.slice(half));
});
DATA.ev.forEach((e, i) => document.getElementById("ev" + (i + 1)).textContent = e.toFixed(1));

// ---------- minimal 3D orbit viewer ----------
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const tip = document.getElementById("tooltip");
const pts = DATA.coords.map((c, i) => ({ x: c[0], y: c[1], z: c[2], n: i + 1, t: DATA.titles[i] }));

let yaw = 0.6, pitch = 0.35, dist = 3.2, panX = 0, panY = 0;
const near = [], far = [];   // scratch arrays
function center() {
  let mx = 0, my = 0, mz = 0;
  for (const p of pts) { mx += p.x; my += p.y; mz += p.z; }
  mx /= pts.length; my /= pts.length; mz /= pts.length;
  let r = 0;
  for (const p of pts) { p.x -= mx; p.y -= my; p.z -= mz; r = Math.max(r, p.x*p.x + p.y*p.y + p.z*p.z); }
  return Math.sqrt(r);
}
const radius = center();

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener("resize", resize);

function project(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = p.x * cy - p.z * sy;
  const z1 = p.x * sy + p.z * cy;
  const y2 = p.y * cp - z1 * sp;
  const z2 = p.y * sp + z1 * cp;
  const zc = z2 + dist * radius;
  const f = (canvas.clientHeight * 0.5) / (zc * 0.9);
  return { sx: canvas.clientWidth / 2 + x1 * f + panX, sy: canvas.clientHeight / 2 - y2 * f + panY, z: zc, f };
}

function draw() {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  // axes
  const o = project({ x: 0, y: 0, z: 0 });
  const L = radius * 1.15;
  const axes = [["PC1", { x: L, y: 0, z: 0 }, "#888"], ["PC2", { x: 0, y: L, z: 0 }, "#888"], ["PC3", { x: 0, y: 0, z: L }, "#888"]];
  ctx.font = "12px sans-serif";
  for (const [name, end, col] of axes) {
    const e = project(end);
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(o.sx, o.sy); ctx.lineTo(e.sx, e.sy); ctx.stroke();
    ctx.fillStyle = "#aaa"; ctx.fillText(name, e.sx + 4, e.sy);
  }
  // points sorted back-to-front
  const order = pts.map((p, i) => ({ i, pr: project(p) })).sort((a, b) => b.pr.z - a.pr.z);
  for (const { i, pr } of order) {
    const isRed = pts[i].n <= 80;
    const r = Math.max(2.5, Math.min(7, pr.f * radius * 0.02));
    ctx.beginPath();
    ctx.arc(pr.sx, pr.sy, r, 0, Math.PI * 2);
    ctx.fillStyle = isRed ? "rgba(224,49,49,0.85)" : "rgba(51,154,240,0.85)";
    ctx.fill();
    if (pts[i].n === 1 || pts[i].n === 80 || pts[i].n === 81 || pts[i].n === 120) {
      ctx.fillStyle = "#ddd"; ctx.fillText(String(pts[i].n), pr.sx + 6, pr.sy - 6);
    }
  }
}

let dragging = false, shifted = false, lastX = 0, lastY = 0;
canvas.addEventListener("pointerdown", e => {
  dragging = true; shifted = e.shiftKey || e.button === 2;
  lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", e => {
  if (dragging) {
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (shifted) { panX += dx; panY += dy; }
    else { yaw -= dx * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008)); }
    draw();
  } else {
    const mx = e.offsetX, my = e.offsetY;
    let hit = null;
    for (const p of pts) {
      const pr = project(p);
      if ((pr.sx - mx) ** 2 + (pr.sy - my) ** 2 < 100) { if (!hit || pr.z < hit.z) hit = pr, hitP = p; }
    }
    if (hit) {
      tip.style.display = "block";
      tip.style.left = (mx + 14) + "px"; tip.style.top = (my + 14) + "px";
      tip.innerHTML = "<b>Chapter " + hitP.n + "</b><br>" + hitP.t;
    } else tip.style.display = "none";
  }
});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  dist = Math.max(1.2, Math.min(20, dist * (e.deltaY > 0 ? 1.1 : 0.9)));
  draw();
}, { passive: false });

resize();
</script>
</body>
</html>
"""


def main():
    chapters = load_chapters(TEXT_FILE)
    print(f"Loaded {len(chapters)} chapters")

    tokenized = tokenize(chapters)
    vocab = top_words(tokenized, TOP_N)
    print(f"Top {len(vocab)} words: {vocab[:10]} ...")

    freqs = frequency_matrix(tokenized, vocab)
    Z = zscores(freqs)

    pca = PCA(n_components=3)
    coords = pca.fit_transform(Z)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    plot_png(coords, chapters)
    print(f"Saved {PNG_FILE}")

    features = group_features(Z, vocab)
    data = {
        "coords": coords.round(4).tolist(),
        "titles": [ch["title"] for ch in chapters],
        "ev": [round(float(v * 100), 1) for v in pca.explained_variance_ratio_],
        "features": features,
    }
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"Saved {HTML_FILE}")

    for grp in ("first80", "last40"):
        print(f"\n{grp} positive:", features[grp]["positive"][:5])
        print(f"{grp} negative:", features[grp]["negative"][:5])


if __name__ == "__main__":
    main()
