#!/usr/bin/env python3
"""
Memory Machine: the photo-AI sorting layer on top of photo_consolidate.py.

Runs entirely on this Mac. Reads the same SQLite catalog and config as
photo_consolidate.py and adds five stages:

  analyze     -- per-photo quality metrics (sharpness, exposure), a thumbnail, and
                 (if the AI extras are installed) a CLIP image embedding
  curate      -- picks the best shot from each burst / near-duplicate group, sends
                 blurry, black and "utility" shots (screenshots, receipts, documents,
                 memes, pocket shots) to _Triage, tags every photo, rewrites the
                 plan's destination paths accordingly
  gallery     -- writes a browsable HTML contact sheet grouped by event
  search      -- natural-language photo search ("kids surfing at sunset")
  highlights  -- copies the top N shots per event to a separate folder
                 (photo book, slideshow, family share)

Typical run:
  python3 photo_consolidate.py --config config.kelly.json inventory
  python3 photo_consolidate.py --config config.kelly.json plan
  python3 memory_machine.py    --config config.kelly.json analyze
  python3 memory_machine.py    --config config.kelly.json curate
  python3 memory_machine.py    --config config.kelly.json gallery
  python3 photo_consolidate.py --config config.kelly.json execute --apply

Re-running `plan` resets destinations, so run `curate` again after any `plan`.
Originals are never moved or modified.
"""
import argparse
import base64
import io
import logging
import math
import shutil
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

import photo_consolidate as pc
from photo_consolidate import Image, connect_db, load_config, now_iso

log = logging.getLogger("memory_machine")

MM_SCHEMA = """
CREATE TABLE IF NOT EXISTS quality (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    sharpness REAL,
    sharpness_global REAL,
    brightness REAL,
    dark_fraction REAL,
    bright_fraction REAL,
    thumb_path TEXT,
    error TEXT,
    analyzed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    model TEXT NOT NULL,
    vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_tags (
    file_id INTEGER NOT NULL REFERENCES files(id),
    tag TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (file_id, tag)
);

CREATE TABLE IF NOT EXISTS curation (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    role TEXT NOT NULL,           -- keeper | extra | junk | single
    reason TEXT,
    score REAL,
    aesthetic REAL,
    highlight INTEGER NOT NULL DEFAULT 0,
    base_dest TEXT,               -- plan destination before curation
    applied_dest TEXT,            -- destination curation wrote into plan
    curated_at TEXT NOT NULL
);
"""

# Content tags: label -> CLIP prompt. Override or extend via config.
DEFAULT_TAGS = {
    "people": "a photo of people",
    "kids": "a photo of young children",
    "baby": "a photo of a baby",
    "family": "a family photo",
    "selfie": "a selfie",
    "beach": "a photo at the beach",
    "ocean": "a photo of the ocean",
    "surfing": "a photo of someone surfing a wave",
    "boat": "a photo of a boat on the water",
    "underwater": "an underwater photo",
    "fish": "a photo of a fish",
    "sunset": "a photo of a sunset",
    "mountains": "a photo of mountains",
    "snow": "a photo of snow",
    "city": "a photo of a city street",
    "food": "a photo of food on a plate",
    "dog": "a photo of a dog",
    "cat": "a photo of a cat",
    "birthday": "a birthday party with a cake",
    "christmas": "a christmas photo",
    "wedding": "a wedding photo",
    "graduation": "a graduation ceremony",
    "sports": "a photo of people playing sports",
    "aircraft": "a photo of an airplane or helicopter",
    "uniform": "a photo of people in military uniform",
    "home": "a photo inside a house",
    "garden": "a photo of a garden",
}

# Shots that are not memories. label -> prompt. Each competes against BASELINE_PROMPT.
DEFAULT_UTILITY = {
    "screenshot": "a screenshot of a phone or computer screen",
    "document": "a photo of a document, form, or page of text",
    "receipt": "a photo of a receipt or invoice",
    "whiteboard": "a photo of a whiteboard or presentation slide",
    "meme": "a meme or graphic with large overlaid text",
    "accidental": "an accidental blurry photo of the floor or the inside of a pocket",
    "product": "a photo of a product label, serial number, or barcode",
}
BASELINE_PROMPT = "a personal photo of people, places, or a memorable moment"
GENERIC_PROMPT = "a photo"
AESTHETIC_GOOD = "a beautiful, sharp, well composed, well lit photograph"
AESTHETIC_BAD = "a blurry, badly composed, poorly lit photograph"

DEFAULTS = {
    "ai": True,
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "batch_size": 32,
    "analysis_long_side": 768,
    "thumb_long_side": 320,
    "blur_threshold": 40.0,      # max-tile Laplacian variance at analysis size; below = blurry
    "dark_threshold": 0.07,      # mean luminance 0..1; below = too dark
    "utility_threshold": 0.60,   # probability a shot is utility (screenshot etc.) to triage it
    "tag_threshold": 0.70,       # probability a tag applies
    "highlights_per_event": 3,
    "tags": DEFAULT_TAGS,
    "utility": DEFAULT_UTILITY,
}


def mm_config(cfg):
    mm = dict(DEFAULTS)
    mm.update(cfg.get("memory_machine", {}))
    return mm


def connect(cfg):
    conn = connect_db(cfg["db_path"])
    conn.executescript(MM_SCHEMA)
    return conn


def thumb_dir(cfg):
    p = Path(cfg.get("thumb_dir") or (Path(cfg["db_path"]).with_suffix("").as_posix() + "_thumbs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Image quality (pure numpy, no AI)
# --------------------------------------------------------------------------

def open_image(path):
    """Open an image upright (EXIF orientation applied) in RGB, or None."""
    try:
        from PIL import ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except Exception as e:
        log.debug("cannot open %s: %s", path, e)
        return None


def resized(im, long_side):
    w, h = im.size
    scale = long_side / max(w, h)
    if scale >= 1:
        return im.copy()
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def laplacian_var(gray):
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           - 4.0 * gray[1:-1, 1:-1])
    return float(lap.var())


def quality_metrics(im, long_side):
    """Sharpness is the best 3x3 tile, so a sharp subject on a soft background
    is not called blurry. Exposure is mean luminance plus clipped fractions."""
    small = resized(im, long_side)
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    h, w = gray.shape
    tiles = []
    for i in range(3):
        for j in range(3):
            t = gray[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            if t.shape[0] > 4 and t.shape[1] > 4:
                tiles.append(laplacian_var(t))
    lum = gray / 255.0
    return {
        "sharpness": max(tiles) if tiles else laplacian_var(gray),
        "sharpness_global": laplacian_var(gray),
        "brightness": float(lum.mean()),
        "dark_fraction": float((lum < 0.03).mean()),
        "bright_fraction": float((lum > 0.97).mean()),
    }


def technical_score(q):
    """0..1 from sharpness and exposure alone."""
    sharp = min(1.0, math.log1p(max(q["sharpness"], 0.0)) / math.log1p(1500.0))
    exposure = 1.0 - min(1.0, abs(q["brightness"] - 0.45) * 2.0)
    exposure -= 0.5 * max(0.0, q["dark_fraction"] - 0.25) + 0.5 * max(0.0, q["bright_fraction"] - 0.10)
    return 0.65 * sharp + 0.35 * max(0.0, exposure)


# --------------------------------------------------------------------------
# AI (CLIP via open_clip; optional)
# --------------------------------------------------------------------------

class ClipEmbedder:
    def __init__(self, model_name, pretrained):
        import open_clip
        import torch
        self.torch = torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"{model_name}/{pretrained}"
        log.info("AI model %s loaded on %s", self.name, self.device)

    def _norm(self, feats):
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)

    def embed_images(self, images):
        with self.torch.no_grad():
            x = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
            return self._norm(self.model.encode_image(x))

    def embed_texts(self, texts):
        with self.torch.no_grad():
            tokens = self.tokenizer(list(texts)).to(self.device)
            return self._norm(self.model.encode_text(tokens))


_embedder_override = None  # tests inject a fake here


def get_embedder(mm, required=False):
    if _embedder_override is not None:
        return _embedder_override
    if not mm["ai"]:
        return None
    try:
        return ClipEmbedder(mm["clip_model"], mm["clip_pretrained"])
    except ImportError:
        msg = ("AI extras not installed; run: pip3 install -r requirements-ai.txt "
               "(continuing with sharpness/exposure only)")
        if required:
            raise SystemExit(msg.replace("(continuing with sharpness/exposure only)", ""))
        log.warning(msg)
        return None


def softmax_pair(a, b, scale=100.0):
    """P(a) when two cosine similarities compete, CLIP logit scale."""
    m = max(a, b)
    ea, eb = math.exp(scale * (a - m)), math.exp(scale * (b - m))
    return ea / (ea + eb)


def softmax(v, scale=100.0):
    z = np.exp(scale * (v - v.max()))
    return z / z.sum()


# --------------------------------------------------------------------------
# Stage: analyze
# --------------------------------------------------------------------------

def candidate_rows(cur, only_unanalyzed):
    sql = ("SELECT f.id, f.abs_path, f.filename FROM files f "
           "JOIN plan pl ON pl.file_id = f.id "
           "WHERE f.is_video = 0 AND pl.is_exact_dup = 0")
    if only_unanalyzed:
        sql += " AND f.id NOT IN (SELECT file_id FROM quality)"
    return cur.execute(sql + " ORDER BY f.id").fetchall()


def save_thumb(im, tdir, file_id, long_side):
    path = tdir / f"{file_id}.jpg"
    resized(im, long_side).save(path, "JPEG", quality=82)
    return str(path)


def do_analyze(cfg, conn, args):
    mm = mm_config(cfg)
    cur = conn.cursor()
    plan_count = cur.execute("SELECT COUNT(*) c FROM plan").fetchone()["c"]
    if plan_count == 0:
        raise SystemExit("no plan found: run photo_consolidate.py plan first")

    rows = candidate_rows(cur, only_unanalyzed=not args.redo)
    embedder = None if args.no_ai else get_embedder(mm)
    if embedder:
        have_vec = {r["file_id"] for r in cur.execute(
            "SELECT file_id FROM embedding WHERE model=?", (embedder.name,))}
        # Pick up photos analyzed earlier without AI.
        extra = cur.execute(
            "SELECT f.id, f.abs_path, f.filename FROM files f JOIN quality q ON q.file_id=f.id "
            "WHERE q.error IS NULL").fetchall()
        seen = {r["id"] for r in rows}
        rows = list(rows) + [r for r in extra if r["id"] not in have_vec and r["id"] not in seen]
    else:
        have_vec = set()

    tdir = thumb_dir(cfg)
    log.info("analyzing %d photos (%s)", len(rows), "AI on" if embedder else "AI off")

    batch_imgs, batch_ids = [], []
    done = 0

    def flush():
        if not batch_ids:
            return
        vecs = embedder.embed_images(batch_imgs)
        cur.executemany(
            "INSERT OR REPLACE INTO embedding (file_id, model, vec) VALUES (?,?,?)",
            [(fid, embedder.name, v.tobytes()) for fid, v in zip(batch_ids, vecs)])
        batch_imgs.clear()
        batch_ids.clear()

    for row in rows:
        im = open_image(row["abs_path"])
        if im is None:
            cur.execute(
                "INSERT OR REPLACE INTO quality (file_id, error, analyzed_at) VALUES (?,?,?)",
                (row["id"], "unreadable (RAW or corrupt?)", now_iso()))
            continue
        q = quality_metrics(im, mm["analysis_long_side"])
        thumb = save_thumb(im, tdir, row["id"], mm["thumb_long_side"])
        cur.execute(
            "INSERT OR REPLACE INTO quality (file_id, sharpness, sharpness_global, brightness, "
            "dark_fraction, bright_fraction, thumb_path, error, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,NULL,?)",
            (row["id"], q["sharpness"], q["sharpness_global"], q["brightness"],
             q["dark_fraction"], q["bright_fraction"], thumb, now_iso()))
        if embedder and (args.redo or row["id"] not in have_vec):
            batch_imgs.append(resized(im, 448))
            batch_ids.append(row["id"])
            if len(batch_ids) >= mm["batch_size"]:
                flush()
        done += 1
        if done % 200 == 0:
            conn.commit()
            log.info("progress: %d / %d", done, len(rows))
    if embedder:
        flush()
    conn.commit()
    log.info("analyze complete: %d photos", done)


# --------------------------------------------------------------------------
# Stage: curate
# --------------------------------------------------------------------------

def load_vectors(cur):
    ids, vecs = [], []
    for r in cur.execute("SELECT file_id, vec FROM embedding"):
        ids.append(r["file_id"])
        vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
    if not ids:
        return [], None
    return ids, np.vstack(vecs)


def ai_judgements(mm, embedder, ids, mat):
    """Per-file tags, utility verdict and aesthetic probability from embeddings."""
    tags_cfg, util_cfg = mm["tags"], mm["utility"]
    tag_names, util_names = list(tags_cfg), list(util_cfg)
    texts = ([tags_cfg[t] for t in tag_names] + [util_cfg[u] for u in util_names]
             + [BASELINE_PROMPT, GENERIC_PROMPT, AESTHETIC_GOOD, AESTHETIC_BAD])
    tvec = embedder.embed_texts(texts)
    sims = mat @ tvec.T
    nt, nu = len(tag_names), len(util_names)
    base_i, gen_i, good_i, bad_i = nt + nu, nt + nu + 1, nt + nu + 2, nt + nu + 3

    out = {}
    for row_i, fid in enumerate(ids):
        s = sims[row_i]
        tags = []
        for k, name in enumerate(tag_names):
            p = softmax_pair(s[k], s[gen_i])
            if p >= mm["tag_threshold"]:
                tags.append((name, p))
        util_probs = softmax(np.concatenate([s[nt:nt + nu], [s[base_i]]]))
        top = int(util_probs[:-1].argmax())
        utility = (util_names[top], float(util_probs[top])) \
            if util_probs[top] >= mm["utility_threshold"] else None
        aesthetic = softmax_pair(s[good_i], s[bad_i])
        out[fid] = {"tags": tags, "utility": utility, "aesthetic": aesthetic}
    return out


def curated_dest(base_dest, role, reason):
    p = Path(base_dest)
    if role == "junk":
        return str(Path("_Triage") / reason / p.name)
    if role == "extra":
        return str(p.parent / "Extras" / p.name)
    return base_dest


def do_curate(cfg, conn, args):
    mm = mm_config(cfg)
    cur = conn.cursor()

    executed = cur.execute("SELECT COUNT(*) c FROM execution WHERE copied=1").fetchone()["c"]
    if executed and not args.force:
        raise SystemExit(f"{executed} files were already copied by execute; curating now would "
                         "change destinations for the rest. Rerun with --force to proceed anyway.")

    rows = cur.execute(
        "SELECT pl.file_id, pl.dest_rel_path, pl.triage_reason, pl.near_dup_group_id, "
        "pl.event_name, pl.resolved_date, q.sharpness, q.sharpness_global, q.brightness, "
        "q.dark_fraction, q.bright_fraction, q.error, c.base_dest, c.applied_dest "
        "FROM plan pl JOIN quality q ON q.file_id = pl.file_id "
        "LEFT JOIN curation c ON c.file_id = pl.file_id "
        "WHERE pl.is_exact_dup = 0").fetchall()
    if not rows:
        raise SystemExit("nothing analyzed yet: run `memory_machine.py analyze` first")

    ai = {}
    ids, mat = load_vectors(cur)
    if ids and not args.no_ai:
        embedder = get_embedder(mm)
        if embedder:
            ai = ai_judgements(mm, embedder, ids, mat)
    log.info("curating %d photos (%s)", len(rows), "with AI" if ai else "sharpness/exposure only")

    util_reasons = set(mm["utility"])
    info = {}
    for r in rows:
        # If curate already rewrote this row and plan has not been rerun since,
        # start from the original plan destination so reruns are idempotent.
        base = r["base_dest"] if (r["applied_dest"] and r["dest_rel_path"] == r["applied_dest"]) \
            else r["dest_rel_path"]
        junk = None
        score = 0.0
        aesthetic = None
        if r["error"] is None:
            q = dict(r)
            score = technical_score(q)
            a = ai.get(r["file_id"])
            if a:
                aesthetic = a["aesthetic"]
                score = 0.55 * score + 0.45 * aesthetic
            if a and a["utility"]:
                junk = a["utility"][0]
            elif r["brightness"] < mm["dark_threshold"]:
                junk = "too_dark"
            elif r["sharpness"] < mm["blur_threshold"]:
                junk = "blurry"
        info[r["file_id"]] = {"row": r, "base": base, "junk": junk, "score": score,
                              "aesthetic": aesthetic}

    # Best shot per near-duplicate group. Within a burst, softer or darker frames go to
    # Extras beside the keeper rather than to Triage; only utility shots leave the event.
    groups = defaultdict(list)
    for fid, d in info.items():
        if d["row"]["near_dup_group_id"] is not None:
            groups[d["row"]["near_dup_group_id"]].append(fid)
    role = {}
    for gid, members in groups.items():
        memories = [m for m in members if info[m]["junk"] not in util_reasons]
        if not memories:
            continue
        keeper = max(memories, key=lambda m: (info[m]["junk"] is None, info[m]["score"]))
        for m in members:
            if m == keeper:
                role[m] = "keeper" if info[m]["junk"] is None else "junk"
            elif m in memories:
                role[m] = "extra"
                info[m]["junk"] = None
            else:
                role[m] = "junk"
    for fid, d in info.items():
        if fid not in role:
            role[fid] = "junk" if d["junk"] else "single"

    # Highlights: top N non-junk, non-extra shots per event (or per day when no event).
    by_event = defaultdict(list)
    for fid, d in info.items():
        if role[fid] in ("keeper", "single") and d["row"]["triage_reason"] is None:
            key = d["row"]["event_name"] or (d["row"]["resolved_date"] or "")[:10]
            by_event[key].append(fid)
    highlights = set()
    for members in by_event.values():
        members.sort(key=lambda m: info[m]["score"], reverse=True)
        highlights.update(members[:mm["highlights_per_event"]])

    now = now_iso()
    cur.execute("DELETE FROM ai_tags")
    curation_rows, plan_updates, tag_rows = [], [], []
    counts = defaultdict(int)
    for fid, d in info.items():
        r = d["row"]
        rl = role[fid]
        reason = d["junk"] if rl == "junk" else None
        # Files photo_consolidate already sent to _Triage stay where it put them.
        if r["triage_reason"]:
            new_dest = d["base"]
        else:
            new_dest = curated_dest(d["base"], rl, reason)
        counts[reason or rl] += 1
        curation_rows.append((fid, rl, reason, d["score"], d["aesthetic"],
                              1 if fid in highlights else 0, d["base"], new_dest, now))
        plan_updates.append((new_dest, fid))
        for tag, p in ai.get(fid, {}).get("tags", []):
            tag_rows.append((fid, tag, p))

    cur.executemany(
        "INSERT OR REPLACE INTO curation (file_id, role, reason, score, aesthetic, highlight, "
        "base_dest, applied_dest, curated_at) VALUES (?,?,?,?,?,?,?,?,?)", curation_rows)
    cur.executemany("UPDATE plan SET dest_rel_path=? WHERE file_id=?", plan_updates)
    cur.executemany("INSERT INTO ai_tags (file_id, tag, score) VALUES (?,?,?)", tag_rows)
    pc.resolve_dest_collisions(cur, Path(cfg["destination"]))
    # Collision resolution may rename; keep applied_dest in sync so reruns stay idempotent.
    cur.execute("UPDATE curation SET applied_dest = (SELECT dest_rel_path FROM plan "
                "WHERE plan.file_id = curation.file_id)")
    conn.commit()

    log.info("curate complete: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log.info("highlights selected: %d", len(highlights))


# --------------------------------------------------------------------------
# Stage: gallery / search output
# --------------------------------------------------------------------------

GALLERY_CSS = """
:root{--bg:#f6f5f2;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#e3e1dc;
--keep:#1f7a4d;--extra:#8a8a8e;--junk:#b3261e;--hi:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#161617;--fg:#f2f2f2;--muted:#a1a1a6;
--card:#222224;--line:#333;--keep:#4cc38a;--extra:#8e8e93;--junk:#ff6b5e;--hi:#e0b04f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",Helvetica,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:14px 16px;z-index:2}h1{font-size:20px;margin:0 0 8px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.bar input{flex:1;min-width:180px;padding:7px 10px;border:1px solid var(--line);
border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
.bar button{padding:6px 11px;border:1px solid var(--line);border-radius:999px;
background:var(--card);color:var(--fg);font:inherit;cursor:pointer}
.bar button.on{background:var(--fg);color:var(--bg)}
main{padding:8px 16px 40px}h2{font-size:16px;margin:22px 0 8px}
h2 small{color:var(--muted);font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.meta{padding:6px 8px;font-size:12px;color:var(--muted)}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:1px 6px;border-radius:5px;
color:#fff;margin-right:4px}.keeper{background:var(--keep)}.extra{background:var(--extra)}
.junk{background:var(--junk)}.hi{background:var(--hi)}.single{display:none}
.card.extra-card img,.card.junk-card img{opacity:.55}.hidden{display:none}
.stats{color:var(--muted);font-size:13px}
"""

GALLERY_JS = """
const q=document.getElementById('q');let mode='all';
function apply(){const t=q.value.toLowerCase().trim();
document.querySelectorAll('.card').forEach(c=>{const okText=!t||c.dataset.k.includes(t);
const okMode=mode==='all'||(mode==='hi'?c.dataset.hi==='1':c.dataset.role===mode);
c.classList.toggle('hidden',!(okText&&okMode));});
document.querySelectorAll('section').forEach(s=>{s.classList.toggle('hidden',
!s.querySelector('.card:not(.hidden)'));});}
q.addEventListener('input',apply);
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{
mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('on',x===b));apply();}));
"""


def thumb_src(thumb_path, out_dir, embed):
    if not thumb_path:
        return ""
    p = Path(thumb_path)
    if embed:
        return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
    try:
        return p.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_uri()


def card_html(r, tags, out_dir, embed, extra_meta=""):
    role = r["role"] or "single"
    reason = r["reason"]
    badges = []
    if r["highlight"]:
        badges.append('<span class="badge hi">best</span>')
    if role == "keeper":
        badges.append('<span class="badge keeper">keeper</span>')
    elif role == "extra":
        badges.append('<span class="badge extra">extra</span>')
    elif role == "junk":
        badges.append(f'<span class="badge junk">{escape(reason or "junk")}</span>')
    tag_text = ", ".join(tags)
    keywords = " ".join([r["filename"], r["event_name"] or "", tag_text, reason or "", role]).lower()
    src = thumb_src(r["thumb_path"], out_dir, embed)
    href = Path(r["abs_path"]).as_uri()
    score = "" if r["score"] is None else f"score {r['score']:.2f}"
    return (f'<div class="card {role}-card" data-role="{role}" data-hi="{r["highlight"] or 0}" '
            f'data-k="{escape(keywords)}"><a href="{escape(href)}" target="_blank">'
            f'<img loading="lazy" src="{escape(src)}" alt="{escape(r["filename"])}"></a>'
            f'<div class="meta">{"".join(badges)}{escape(r["filename"])}<br>'
            f'{escape(tag_text)}{"<br>" if tag_text else ""}{score}{extra_meta}</div></div>')


def page(title, stats, body, filters=True):
    buttons = ""
    if filters:
        buttons = ''.join(
            f'<button data-mode="{m}"{" class=on" if m == "all" else ""}>{label}</button>'
            for m, label in [("all", "All"), ("hi", "Best"), ("keeper", "Keepers"),
                             ("extra", "Extras"), ("junk", "Triage")])
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{GALLERY_CSS}</style></head><body>'
            f'<header><h1>{escape(title)}</h1><div class="bar">'
            f'<input id="q" placeholder="Filter by tag, event or filename">{buttons}</div>'
            f'<div class="stats">{stats}</div></header><main>{body}</main>'
            f'<script>{GALLERY_JS}</script></body></html>')


def gallery_rows(cur):
    return cur.execute(
        "SELECT f.id, f.abs_path, f.filename, pl.dest_rel_path, pl.event_name, pl.resolved_date, "
        "pl.triage_reason, q.thumb_path, c.role, c.reason, c.score, c.highlight "
        "FROM plan pl JOIN files f ON f.id = pl.file_id JOIN quality q ON q.file_id = f.id "
        "LEFT JOIN curation c ON c.file_id = f.id "
        "WHERE pl.is_exact_dup = 0 AND q.error IS NULL "
        "ORDER BY pl.resolved_date, f.filename").fetchall()


def tag_map(cur):
    m = defaultdict(list)
    for r in cur.execute("SELECT file_id, tag FROM ai_tags ORDER BY score DESC"):
        m[r["file_id"]].append(r["tag"])
    return m


def section_key(r):
    dest = Path(r["dest_rel_path"])
    parts = [p for p in dest.parent.parts if p != "Extras"]
    return " / ".join(parts) if parts else "(root)"


def do_gallery(cfg, conn, args):
    cur = conn.cursor()
    rows = gallery_rows(cur)
    if not rows:
        raise SystemExit("nothing to show: run analyze (and curate) first")
    tags = tag_map(cur)
    out = Path(args.output or f"{cfg['person']}_gallery.html")
    out_dir = out.parent

    sections = defaultdict(list)
    for r in rows:
        sections[section_key(r)].append(r)

    def sort_key(k):
        return (k.startswith("_Triage"), k)

    body = []
    for key in sorted(sections, key=sort_key):
        members = sections[key]
        cards = "".join(card_html(r, tags.get(r["id"], []), out_dir, args.embed) for r in members)
        body.append(f'<section><h2>{escape(key)} <small>{len(members)} photos</small></h2>'
                    f'<div class="grid">{cards}</div></section>')

    roles = defaultdict(int)
    for r in rows:
        roles[r["role"] or "uncurated"] += 1
    hi = sum(1 for r in rows if r["highlight"])
    stats = (f"{len(rows)} photos · {hi} best · {roles['keeper']} keepers · "
             f"{roles['extra']} extras · {roles['junk']} triaged")
    out.write_text(page(f"Memory Machine: {cfg['person']}", stats, "".join(body)))
    log.info("gallery written to %s", out)
    print(out.resolve())


def do_search(cfg, conn, args):
    mm = mm_config(cfg)
    cur = conn.cursor()
    ids, mat = load_vectors(cur)
    if not ids:
        raise SystemExit("no AI embeddings yet: install requirements-ai.txt and run analyze")
    embedder = get_embedder(mm, required=True)
    q = embedder.embed_texts([args.query])[0]
    sims = mat @ q
    order = np.argsort(-sims)[:args.top]
    by_id = {r["id"]: r for r in gallery_rows(cur)}
    tags = tag_map(cur)

    hits = [(ids[i], float(sims[i])) for i in order if ids[i] in by_id]
    for fid, s in hits:
        print(f"{s:.3f}  {by_id[fid]['abs_path']}")

    out = Path(args.output or f"{cfg['person']}_search.html")
    cards = "".join(card_html(by_id[fid], tags.get(fid, []), out.parent, args.embed,
                              f"<br>match {s:.3f}") for fid, s in hits)
    out.write_text(page(f"Search: {args.query}", f"top {len(hits)} matches",
                        f'<section><div class="grid">{cards}</div></section>', filters=False))
    log.info("results page written to %s", out)


# --------------------------------------------------------------------------
# Stage: highlights
# --------------------------------------------------------------------------

def do_highlights(cfg, conn, args):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT f.abs_path, pl.dest_rel_path FROM curation c "
        "JOIN files f ON f.id = c.file_id JOIN plan pl ON pl.file_id = c.file_id "
        "WHERE c.highlight = 1 ORDER BY pl.resolved_date").fetchall()
    if not rows:
        raise SystemExit("no highlights: run curate first")
    dest_root = Path(args.dest)
    log.info("%d highlights (%s)", len(rows), "APPLYING" if args.apply else "DRY RUN")
    for r in rows:
        target = dest_root / r["dest_rel_path"]
        if not args.apply:
            log.info("[dry-run] would copy %s -> %s", r["abs_path"], target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(r["abs_path"], target)
    if not args.apply:
        log.info("dry run complete -- rerun with --apply to copy")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="same config JSON as photo_consolidate.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="stage", required=True)

    a = sub.add_parser("analyze", help="quality metrics, thumbnails, AI embeddings")
    a.add_argument("--redo", action="store_true", help="re-analyze photos already done")
    a.add_argument("--no-ai", action="store_true", help="skip the AI model")

    c = sub.add_parser("curate", help="pick best shots, triage junk, tag, update plan")
    c.add_argument("--no-ai", action="store_true", help="ignore AI embeddings")
    c.add_argument("--force", action="store_true", help="curate even after execute has started")

    g = sub.add_parser("gallery", help="write an HTML contact sheet")
    g.add_argument("--output", help="default: <person>_gallery.html")
    g.add_argument("--embed", action="store_true", help="inline thumbnails (portable single file)")

    s = sub.add_parser("search", help="find photos by description")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--output", help="default: <person>_search.html")
    s.add_argument("--embed", action="store_true")

    h = sub.add_parser("highlights", help="copy the best shots per event to a folder")
    h.add_argument("--dest", required=True)
    h.add_argument("--apply", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    conn = connect(cfg)
    {"analyze": do_analyze, "curate": do_curate, "gallery": do_gallery,
     "search": do_search, "highlights": do_highlights}[args.stage](cfg, conn, args)


if __name__ == "__main__":
    sys.exit(main())
