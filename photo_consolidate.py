#!/usr/bin/env python3
"""
Photo/video consolidation pipeline.

Four resumable stages, backed by a per-person SQLite catalog:
  inventory  -- walk source folders, hash files, pull EXIF/metadata, perceptual-hash images
  plan       -- resolve dates, event names, dedup/near-dup groups, destination paths (no disk writes)
  report     -- human-readable summary + near-duplicate review list
  execute    -- copy planned files to destination and verify (--apply to actually write)

Source files are never moved or modified. Destination is only touched by `execute --apply`.
"""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

try:
    import imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False

log = logging.getLogger("photo_consolidate")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    source_root TEXT NOT NULL,
    abs_path TEXT NOT NULL UNIQUE,
    rel_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    ext TEXT NOT NULL,
    is_video INTEGER NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha256 TEXT,
    inventoried_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);

CREATE TABLE IF NOT EXISTS metadata (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    exif_datetime TEXT,
    exif_source_field TEXT,
    camera_make TEXT,
    camera_model TEXT,
    width INTEGER,
    height INTEGER,
    software TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS phash (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    phash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    resolved_date TEXT,
    date_confidence TEXT,
    date_source_detail TEXT,
    event_name TEXT,
    event_source TEXT,
    is_exact_dup INTEGER NOT NULL DEFAULT 0,
    dup_of_file_id INTEGER,
    is_near_dup INTEGER NOT NULL DEFAULT 0,
    near_dup_group_id INTEGER,
    triage_reason TEXT,
    dest_rel_path TEXT,
    planned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    copied INTEGER NOT NULL DEFAULT 0,
    dest_abs_path TEXT,
    copy_sha256 TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    executed_at TEXT,
    error TEXT
);
"""

MONTH_NAMES = [
    "01-January", "02-February", "03-March", "04-April", "05-May", "06-June",
    "07-July", "08-August", "09-September", "10-October", "11-November", "12-December",
]

GENERIC_FOLDER_NAMES = {
    "dcim", "camera", "camera roll", "100apple", "photos", "pictures",
    "download", "downloads", "img", "images", "misc", "unsorted", "new folder",
    "photo library", "photos library", "originals", "masters", "export", "exports",
    "videos", "video", "movies", "movie",
}

FILENAME_DATE_RE = re.compile(
    r"(?P<y>19\d{2}|20\d{2})[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<d>0[1-9]|[12]\d|3[01])"
    r"(?:[-_ ]?(?P<H>[01]\d|2[0-3])[-_.:]?(?P<M>[0-5]\d)?[-_.:]?(?P<S>[0-5]\d)?)?"
)

DEFAULT_SCREENSHOT_PATTERNS = ["^screenshot", "^screen shot", "^screen_shot", "^img_e\\d", "^snip"]


# --------------------------------------------------------------------------
# Config / DB plumbing
# --------------------------------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("exiftool_path", "exiftool")
    cfg.setdefault("near_dup_hamming_threshold", 6)
    cfg.setdefault("screenshot_filename_patterns", DEFAULT_SCREENSHOT_PATTERNS)
    return cfg


def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Stage 1: inventory
# --------------------------------------------------------------------------

def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_sources(cfg):
    image_exts = set(e.lower() for e in cfg["image_extensions"])
    video_exts = set(e.lower() for e in cfg["video_extensions"])
    all_exts = image_exts | video_exts
    for source_root in cfg["sources"]:
        root = Path(source_root)
        if not root.exists():
            log.warning("source root does not exist, skipping: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                ext = Path(name).suffix.lower()
                if ext not in all_exts:
                    continue
                abs_path = Path(dirpath) / name
                yield source_root, abs_path, ext, ext in video_exts


def run_exiftool_batch(paths, exiftool_path):
    """Run exiftool -j across a batch of paths, return {abs_path_str: record_dict}."""
    if not paths:
        return {}
    cmd = [exiftool_path, "-j", "-n", "-fast2",
           "-DateTimeOriginal", "-CreateDate", "-MediaCreateDate", "-TrackCreateDate",
           "-FileModifyDate", "-Make", "-Model", "-Software",
           "-ImageWidth", "-ImageHeight", "-SourceImageWidth", "-SourceImageHeight"]
    cmd += [str(p) for p in paths]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        log.error("exiftool not found at %r; metadata extraction disabled", exiftool_path)
        return {}
    if not out.stdout:
        return {}
    try:
        records = json.loads(out.stdout)
    except json.JSONDecodeError:
        log.warning("exiftool returned unparseable output for a batch of %d files", len(paths))
        return {}
    return {rec.get("SourceFile"): rec for rec in records}


EXIF_DATE_FIELDS = [
    ("DateTimeOriginal", "exif"),
    ("CreateDate", "exif"),
    ("MediaCreateDate", "exif"),
    ("TrackCreateDate", "exif"),
]


def parse_exif_datetime(value):
    """exiftool -n gives 'YYYY:MM:DD HH:MM:SS' (or with subsec/offset variants)."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", value)
    if m:
        y, mo, d, H, M, S = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, H, M, S)
        except ValueError:
            return None
    return None


def extract_exif_date(record):
    if not record:
        return None, None
    for field, _ in EXIF_DATE_FIELDS:
        dt = parse_exif_datetime(record.get(field))
        if dt and dt.year >= 1990:
            return dt, field
    return None, None


def do_inventory(cfg, conn, args):
    cur = conn.cursor()
    existing = {row["abs_path"] for row in cur.execute("SELECT abs_path FROM files")}

    candidates = []
    for source_root, abs_path, ext, is_video in walk_sources(cfg):
        if str(abs_path) in existing:
            continue
        candidates.append((source_root, abs_path, ext, is_video))

    log.info("found %d new files to inventory (skipping %d already known)", len(candidates), len(existing))

    batch_size = 100
    processed = 0
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        exif_records = run_exiftool_batch([b[1] for b in batch], cfg["exiftool_path"])

        for source_root, abs_path, ext, is_video in batch:
            try:
                st = abs_path.stat()
            except OSError as e:
                log.warning("could not stat %s: %s", abs_path, e)
                continue
            try:
                digest = sha256_file(abs_path)
            except OSError as e:
                log.warning("could not read %s: %s", abs_path, e)
                continue

            rel_path = str(abs_path.relative_to(source_root))
            cur.execute(
                "INSERT INTO files (source_root, abs_path, rel_path, filename, ext, is_video, "
                "size, mtime, sha256, inventoried_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (source_root, str(abs_path), rel_path, abs_path.name, ext, int(is_video),
                 st.st_size, st.st_mtime, digest, now_iso()),
            )
            file_id = cur.lastrowid

            record = exif_records.get(str(abs_path))
            exif_dt, exif_field = extract_exif_date(record)
            width = height = None
            camera_make = camera_model = software = None
            raw_json = None
            if record:
                raw_json = json.dumps(record)
                width = record.get("ImageWidth") or record.get("SourceImageWidth")
                height = record.get("ImageHeight") or record.get("SourceImageHeight")
                camera_make = record.get("Make")
                camera_model = record.get("Model")
                software = record.get("Software")
            cur.execute(
                "INSERT INTO metadata (file_id, exif_datetime, exif_source_field, camera_make, "
                "camera_model, width, height, software, raw_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (file_id, exif_dt.isoformat() if exif_dt else None, exif_field,
                 camera_make, camera_model, width, height, software, raw_json),
            )

            if not is_video and HAVE_PIL and HAVE_IMAGEHASH:
                ph = compute_phash(abs_path)
                if ph is not None:
                    cur.execute("INSERT INTO phash (file_id, phash) VALUES (?,?)", (file_id, str(ph)))

            processed += 1

        conn.commit()
        if processed % 500 < batch_size:
            log.info("inventoried %d/%d", processed, len(candidates))

    log.info("inventory complete: %d new files", processed)


def compute_phash(path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return imagehash.phash(im)
    except Exception as e:
        log.debug("phash failed for %s: %s", path, e)
        return None


# --------------------------------------------------------------------------
# Stage 2: plan
# --------------------------------------------------------------------------

def parse_filename_date(filename):
    m = FILENAME_DATE_RE.search(filename)
    if not m:
        return None
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    H = int(m.group("H") or 0)
    M = int(m.group("M") or 0)
    S = int(m.group("S") or 0)
    try:
        return datetime(y, mo, d, H, M, S)
    except ValueError:
        return None


def parse_folder_date(folder_name):
    m = FILENAME_DATE_RE.search(folder_name)
    if m:
        try:
            y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            return datetime(y, mo, d)
        except ValueError:
            pass
    if dateutil_parser is not None:
        try:
            dt = dateutil_parser.parse(folder_name, fuzzy=True, default=datetime(1900, 1, 1))
            if dt.year != 1900 and 1990 <= dt.year <= datetime.now().year + 1:
                return dt
        except (ValueError, OverflowError):
            pass
    return None


def resolve_date(file_row, meta_row):
    """EXIF -> filename -> folder name -> filesystem mtime, each tagged with confidence."""
    if meta_row and meta_row["exif_datetime"]:
        return datetime.fromisoformat(meta_row["exif_datetime"]), "exif", meta_row["exif_source_field"]

    fname_dt = parse_filename_date(file_row["filename"])
    if fname_dt:
        return fname_dt, "filename", None

    rel_parts = Path(file_row["rel_path"]).parts[:-1]
    for part in reversed(rel_parts):
        folder_dt = parse_folder_date(part)
        if folder_dt:
            return folder_dt, "folder", part

    return datetime.fromtimestamp(file_row["mtime"]), "mtime", None


def immediate_parent_folder(rel_path):
    parts = Path(rel_path).parts
    return parts[-2] if len(parts) >= 2 else None


def sanitize_name(name):
    name = re.sub(r'[\/:*?"<>|]', "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:120] if name else "Unnamed"


def is_generic_folder(name):
    n = name.lower().strip()
    if n in GENERIC_FOLDER_NAMES:
        return True
    if re.fullmatch(r"(19|20)\d{2}", n):
        return True
    if re.fullmatch(r"(19|20)\d{2}[-_ ](0[1-9]|1[0-2])", n):
        return True
    return False


CAMERA_PREFIX_RE = re.compile(
    r"^(IMG|IMAGE|DSC|DSCN|DCIM|VID|VIDEO|MOV|PXL|MVIMG|GOPR|GOPRO|PANO|"
    r"WA\d*|Screenshot|Photo|Snapchat)[-_ ]*", re.I
)


def compute_event_name(file_row):
    """origin folder name > filename > (temporal clustering handled separately, conservative last resort).

    Filenames from cameras/phones (IMG_20200304_101500_0.jpg etc.) carry no real event
    info once the date and boilerplate prefix are stripped -- only fall back to a
    filename-derived name when what's left contains an actual descriptive word.
    """
    parent = immediate_parent_folder(file_row["rel_path"])
    if parent and not is_generic_folder(parent):
        cleaned = re.sub(FILENAME_DATE_RE, "", parent).strip(" -_.")
        if len(cleaned) >= 3:
            return sanitize_name(cleaned), "folder"

    stem = Path(file_row["filename"]).stem
    stem = FILENAME_DATE_RE.sub("", stem)
    stem = CAMERA_PREFIX_RE.sub("", stem)
    tokens = [t for t in re.split(r"[-_ .]+", stem) if len(t) >= 3 and re.search(r"[A-Za-z]{3,}", t)]
    if tokens:
        candidate = " ".join(tokens).strip()
        if len(candidate) >= 4:
            return sanitize_name(candidate), "filename"

    return None, None


def cluster_events_by_time(rows_with_dates, gap_hours=36):
    """Conservative last-resort clustering: group files within `gap_hours` of each
    other into a single unnamed cluster, but only keep clusters of a meaningful size
    (>=3 files) as real 'events' -- otherwise leave files loose in the month folder."""
    rows_sorted = sorted(rows_with_dates, key=lambda r: r[1])
    clusters = []
    current = []
    prev_dt = None
    for file_id, dt in rows_sorted:
        if prev_dt is not None and (dt - prev_dt).total_seconds() > gap_hours * 3600:
            clusters.append(current)
            current = []
        current.append((file_id, dt))
        prev_dt = dt
    if current:
        clusters.append(current)

    assignment = {}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        start = cluster[0][1]
        label = f"Cluster {start.strftime('%Y-%m-%d')}"
        for file_id, _ in cluster:
            assignment[file_id] = label
    return assignment


def is_screenshot(file_row, meta_row, patterns):
    name = file_row["filename"].lower()
    for pat in patterns:
        if re.search(pat, name, flags=re.I):
            return True
    if meta_row and meta_row["software"] and "screenshot" in str(meta_row["software"]).lower():
        return True
    return False


def build_exact_dup_map(cur):
    """Return {file_id: canonical_file_id} for exact duplicates (by sha256),
    canonical = lexicographically-first abs_path in the group."""
    groups = defaultdict(list)
    for row in cur.execute("SELECT id, abs_path, sha256 FROM files WHERE sha256 IS NOT NULL"):
        groups[row["sha256"]].append((row["abs_path"], row["id"]))

    dup_map = {}
    for sha, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m[0])
        canonical_id = members[0][1]
        for abs_path, file_id in members[1:]:
            dup_map[file_id] = canonical_id
    return dup_map


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def build_near_dup_groups(cur, threshold, exclude_ids):
    """Group images by resolved date (day bucket) then compare phashes pairwise
    within the bucket -- keeps comparisons tractable instead of all-pairs over
    the whole catalog."""
    rows = cur.execute(
        "SELECT p.file_id, p.phash, pl.resolved_date FROM phash p "
        "JOIN plan pl ON pl.file_id = p.file_id"
    ).fetchall()

    buckets = defaultdict(list)
    for row in rows:
        if row["file_id"] in exclude_ids:
            continue
        day = (row["resolved_date"] or "")[:10]
        buckets[day].append((row["file_id"], row["phash"]))

    group_id_counter = 0
    file_group = {}
    for day, items in buckets.items():
        if len(items) < 2:
            continue
        n = len(items)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                if hamming(items[i][1], items[j][1]) <= threshold:
                    union(i, j)

        root_members = defaultdict(list)
        for i in range(n):
            root_members[find(i)].append(items[i][0])

        for members in root_members.values():
            if len(members) < 2:
                continue
            group_id_counter += 1
            for file_id in members:
                file_group[file_id] = group_id_counter

    return file_group


def do_plan(cfg, conn, args):
    cur = conn.cursor()
    dest_root = Path(cfg["destination"])

    file_rows = {r["id"]: r for r in cur.execute("SELECT * FROM files")}
    meta_rows = {r["file_id"]: r for r in cur.execute("SELECT * FROM metadata")}

    dup_map = build_exact_dup_map(cur)
    log.info("exact-duplicate groups: %d files marked as duplicates", len(dup_map))

    dated = []
    for file_id, file_row in file_rows.items():
        meta_row = meta_rows.get(file_id)
        dt, confidence, detail = resolve_date(file_row, meta_row)
        event_name, event_source = (None, None)
        if confidence != "mtime":
            event_name, event_source = compute_event_name(file_row)
        dated.append((file_id, dt, confidence, detail, event_name, event_source))

    no_reliable_date_ids = {fid for fid, dt, conf, detail, en, es in dated if conf == "mtime"}
    cluster_assignment = cluster_events_by_time(
        [(fid, dt) for fid, dt, conf, detail, en, es in dated if en is None and conf != "mtime"]
    )

    now = now_iso()
    plan_rows = []
    for file_id, dt, confidence, detail, event_name, event_source in dated:
        file_row = file_rows[file_id]
        meta_row = meta_rows.get(file_id)

        if event_name is None and file_id in cluster_assignment:
            event_name, event_source = cluster_assignment[file_id], "cluster"

        triage_reason = None
        if confidence == "mtime":
            triage_reason = "no_date"
        elif is_screenshot(file_row, meta_row, cfg["screenshot_filename_patterns"]):
            triage_reason = "screenshot"

        is_exact_dup = 1 if file_id in dup_map else 0
        dup_of = dup_map.get(file_id)

        if triage_reason:
            dest_rel = str(Path("_Triage") / triage_reason / file_row["filename"])
        else:
            month_folder = MONTH_NAMES[dt.month - 1]
            parts = [f"{dt.year:04d}", month_folder]
            if event_name:
                parts.append(sanitize_name(event_name))
            parts.append(file_row["filename"])
            dest_rel = str(Path(*parts))

        plan_rows.append((
            file_id, dt.isoformat(), confidence, detail, event_name, event_source,
            is_exact_dup, dup_of, 0, None, triage_reason, dest_rel, now,
        ))

    cur.execute("DELETE FROM plan")
    cur.executemany(
        "INSERT INTO plan (file_id, resolved_date, date_confidence, date_source_detail, "
        "event_name, event_source, is_exact_dup, dup_of_file_id, is_near_dup, near_dup_group_id, "
        "triage_reason, dest_rel_path, planned_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        plan_rows,
    )
    conn.commit()

    near_dup_groups = build_near_dup_groups(cur, cfg["near_dup_hamming_threshold"], set(dup_map.keys()))
    for file_id, group_id in near_dup_groups.items():
        cur.execute(
            "UPDATE plan SET is_near_dup=1, near_dup_group_id=? WHERE file_id=?",
            (group_id, file_id),
        )
    conn.commit()
    log.info("near-duplicate groups: %d files across %d groups",
              len(near_dup_groups), len(set(near_dup_groups.values())))

    resolve_dest_collisions(cur, dest_root)
    conn.commit()
    log.info("plan complete: %d files planned", len(plan_rows))


def resolve_dest_collisions(cur, dest_root):
    """Different-content files that would land on the same destination path get
    a disambiguating suffix appended before the extension."""
    rows = cur.execute(
        "SELECT pl.file_id, pl.dest_rel_path, f.sha256 FROM plan pl "
        "JOIN files f ON f.id = pl.file_id WHERE pl.is_exact_dup = 0"
    ).fetchall()

    seen = {}
    updates = []
    for row in rows:
        key = row["dest_rel_path"]
        if key not in seen:
            seen[key] = row["sha256"]
            continue
        if seen[key] == row["sha256"]:
            continue
        p = Path(row["dest_rel_path"])
        n = 2
        while True:
            candidate = str(p.with_name(f"{p.stem} ({n}){p.suffix}"))
            if candidate not in seen or seen[candidate] == row["sha256"]:
                break
            n += 1
        seen[candidate] = row["sha256"]
        updates.append((candidate, row["file_id"]))

    cur.executemany("UPDATE plan SET dest_rel_path=? WHERE file_id=?", updates)
    if updates:
        log.info("resolved %d destination filename collisions", len(updates))


# --------------------------------------------------------------------------
# Stage 3: report
# --------------------------------------------------------------------------

def do_report(cfg, conn, args):
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
    exact_dups = cur.execute("SELECT COUNT(*) c FROM plan WHERE is_exact_dup=1").fetchone()["c"]
    near_dups = cur.execute("SELECT COUNT(*) c FROM plan WHERE is_near_dup=1").fetchone()["c"]
    triage_counts = cur.execute(
        "SELECT triage_reason, COUNT(*) c FROM plan WHERE triage_reason IS NOT NULL GROUP BY triage_reason"
    ).fetchall()
    to_copy = cur.execute("SELECT COUNT(*) c FROM plan WHERE is_exact_dup=0").fetchone()["c"]

    lines = []
    lines.append(f"# Consolidation report -- {cfg['person']}")
    lines.append(f"Generated: {now_iso()}\n")
    lines.append(f"Total files inventoried: {total}")
    lines.append(f"Exact duplicates (will not be copied): {exact_dups}")
    lines.append(f"Files that will be copied: {to_copy}")
    lines.append(f"Near-duplicates flagged for manual review: {near_dups}\n")

    lines.append("## Triage buckets (no reliable EXIF date, or screenshots)")
    for row in triage_counts:
        lines.append(f"- {row['triage_reason']}: {row['c']}")
    lines.append("")

    lines.append("## Near-duplicate groups (review before manual merge -- not auto-merged)")
    groups = cur.execute(
        "SELECT near_dup_group_id, GROUP_CONCAT(f.abs_path, ' || ') paths, COUNT(*) n "
        "FROM plan pl JOIN files f ON f.id = pl.file_id "
        "WHERE pl.is_near_dup=1 GROUP BY near_dup_group_id ORDER BY near_dup_group_id"
    ).fetchall()
    for g in groups:
        lines.append(f"### Group {g['near_dup_group_id']} ({g['n']} files)")
        for p in g["paths"].split(" || "):
            lines.append(f"- {p}")
        lines.append("")

    lines.append("## Planned destination structure (top-level counts)")
    struct = cur.execute(
        "SELECT SUBSTR(dest_rel_path, 1, INSTR(dest_rel_path || '/', '/') - 1) top, COUNT(*) c "
        "FROM plan WHERE is_exact_dup=0 GROUP BY top ORDER BY top"
    ).fetchall()
    for row in struct:
        lines.append(f"- {row['top']}: {row['c']} files")

    out_path = Path(args.output) if args.output else Path(f"{cfg['person']}_report.md")
    out_path.write_text("\n".join(lines))
    log.info("report written to %s", out_path)
    print("\n".join(lines))


# --------------------------------------------------------------------------
# Stage 4: execute
# --------------------------------------------------------------------------

def do_execute(cfg, conn, args):
    cur = conn.cursor()
    dest_root = Path(cfg["destination"])

    rows = cur.execute(
        "SELECT pl.*, f.abs_path, f.sha256 FROM plan pl "
        "JOIN files f ON f.id = pl.file_id "
        "LEFT JOIN execution e ON e.file_id = pl.file_id "
        "WHERE pl.is_exact_dup = 0 AND (e.copied IS NULL OR e.copied = 0)"
    ).fetchall()

    log.info("%d files to copy (%s)", len(rows), "APPLYING" if args.apply else "DRY RUN")

    copied = errors = 0
    for row in rows:
        dest_path = dest_root / row["dest_rel_path"]
        if not args.apply:
            log.info("[dry-run] would copy %s -> %s", row["abs_path"], dest_path)
            continue

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["abs_path"], dest_path)
            copy_hash = sha256_file(dest_path)
            verified = 1 if copy_hash == row["sha256"] else 0
            if not verified:
                log.error("verification FAILED for %s (dest hash mismatch)", dest_path)
            cur.execute(
                "INSERT INTO execution (file_id, copied, dest_abs_path, copy_sha256, verified, executed_at) "
                "VALUES (?,1,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET "
                "copied=1, dest_abs_path=excluded.dest_abs_path, copy_sha256=excluded.copy_sha256, "
                "verified=excluded.verified, executed_at=excluded.executed_at",
                (row["file_id"], str(dest_path), copy_hash, verified, now_iso()),
            )
            copied += 1
        except OSError as e:
            log.error("failed to copy %s: %s", row["abs_path"], e)
            cur.execute(
                "INSERT INTO execution (file_id, copied, error, executed_at) VALUES (?,0,?,?) "
                "ON CONFLICT(file_id) DO UPDATE SET error=excluded.error, executed_at=excluded.executed_at",
                (row["file_id"], str(e), now_iso()),
            )
            errors += 1

        if (copied + errors) % 200 == 0:
            conn.commit()
            log.info("progress: %d copied, %d errors", copied, errors)

    conn.commit()
    if args.apply:
        log.info("execute complete: %d copied, %d errors", copied, errors)
    else:
        log.info("dry run complete -- rerun with --apply to actually copy files")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to config JSON (e.g. config.kelly.json)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("inventory", help="walk sources, hash + extract metadata")
    sub.add_parser("plan", help="resolve dates/events/dups, compute destination paths")

    report_p = sub.add_parser("report", help="write a human-readable summary report")
    report_p.add_argument("--output", help="report output path (default: <person>_report.md)")

    execute_p = sub.add_parser("execute", help="copy planned files to destination")
    execute_p.add_argument("--apply", action="store_true", help="actually copy files (default is dry-run)")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    conn = connect_db(cfg["db_path"])

    if args.stage == "inventory":
        do_inventory(cfg, conn, args)
    elif args.stage == "plan":
        do_plan(cfg, conn, args)
    elif args.stage == "report":
        do_report(cfg, conn, args)
    elif args.stage == "execute":
        do_execute(cfg, conn, args)


if __name__ == "__main__":
    sys.exit(main())
