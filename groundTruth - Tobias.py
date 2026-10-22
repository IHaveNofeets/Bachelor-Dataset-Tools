import argparse
import glob
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove    # imported here so the tool still runs without rembg

# Class names that Space cycles through (the id saved in truth.json is the index in this list).
CLASSES = ["2x4 brick", "Slide", "Brown rock", "gray rock", "2x2 brick", "2x2 brick with letter", "Other"]
BRUSH_DEFAULT = 20       # brush radius in org.png pixels
TOLERANCE_DEFAULT = 10   # flood-fill colour tolerance (per Lab channel)
MAX_VIEW = (2100, 1000)   # the image is scaled down to fit in this size (the sidebar is added to the right of it)
SIDEBAR_W = 380          # width of the text sidebar in window pixels
SIDEBAR_MIN_H = 960      # the window is at least this tall so the whole sidebar fits for small images
POLY_EPSILON = 1.0       # polygon simplification in org.png pixels
DRAFT_THRESHOLD = 127    # rembg mask value (0-255) above which a pixel counts as foreground
DRAFT_MIN_AREA = 5000    # connected components smaller than this (org.png pixels) are ignored in the draft
REMBG_MODEL = None       # None = rembg's default model (slow on CPU in new versions); "u2net" is much faster
REDRAFT_LOG = Path(__file__).with_name("groundTruth_redraft.log")   # what a redraft started in the window printed
SPLIT_THICKNESS = 3      # width (org.png pixels) of the cut made by the split tool
POLY_CLOSE_PX = 10       # clicking within this many window pixels of the first polygon point closes the polygon
SPLIT_MIN_PART = 30     # pieces smaller than this (pixels) are not counted as a separate part when splitting
COVER_GAP = 2            # objects whose outlines are this close (org.png pixels) count as touching
COVER_MIN_CONTACT = 30   # ...if at least this many pixels of one are within COVER_GAP of the other

WINDOW = "groundTruth"
# BGR, one per class (wraps around if there are more classes than colors)
PALETTE = [(60, 60, 255), (60, 200, 60), (255, 140, 40), (0, 200, 255), (255, 0, 200), (255, 255, 0), (0, 0, 0)]


def wrap_text(text, max_width, scale, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Split text into lines that fit in max_width pixels when drawn with cv2.putText."""
    lines, line = [], ""
    for word in text.split(" "):
        candidate = f"{line} {word}" if line else word
        if line and cv2.getTextSize(candidate, font, scale, 1)[0][0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def find_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def fill_holes(mask):
    out = np.zeros_like(mask)
    cv2.drawContours(out, find_contours(mask), -1, 255, cv2.FILLED)
    return out


_rembg_sessions = {}                 # model name -> loaded rembg session (each model is loaded once)
LEGACY_DRAFT_MODEL = "u2net"         # the model that made the drafts from before drafts were marked with a model


def rembg_default_model():
    return inspect.signature(new_session).parameters["model_name"].default


def effective_model(model=None):
    """Name of the rembg model that is used: the given one, else REMBG_MODEL, else rembg's own default."""
    return model or REMBG_MODEL or rembg_default_model()


def draft_labels(img, model=None):
    """Draft ground truth: rembg foreground mask, split into objects with connectedComponents.
    Returns a uint16 label map (0 = background, 1..n = one object per connected component).
    Only ever called in the draft child process (see draft_process_main) or in headless mode."""
    name = effective_model(model)
    if name not in _rembg_sessions:  # load the model once instead of on every call
        _rembg_sessions[name] = new_session(name)
    mask = remove(img[:, :, ::-1].copy(), session=_rembg_sessions[name], only_mask=True)
    fg = (mask > DRAFT_THRESHOLD).astype(np.uint8) * 255
    count, cc, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    labels = np.zeros(fg.shape, np.uint16)
    n = 0
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= DRAFT_MIN_AREA:
            n += 1
            labels[fill_holes((cc == i).astype(np.uint8) * 255) > 0] = n
    return labels


def split_by_path(labels, points, closed, next_id, thickness=SPLIT_THICKNESS):
    """Cut every object that the path through points (a polyline, or a closed polygon outline) touches into the
    separate pieces the path leaves behind. A closed path inside an object cuts out the piece it surrounds.
    The largest piece keeps the object's id, the other pieces get new ids starting at next_id.
    Changes labels in place and returns {new_id: old_id} (empty if nothing was split)."""
    line = np.zeros(labels.shape, np.uint8)
    cv2.polylines(line, [np.array(points, np.int32).reshape(-1, 1, 2)], closed, 255, thickness)
    new_ids = {}
    for obj in [int(i) for i in np.unique(labels[line > 0]) if i]:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((labels == obj).astype(np.uint8)))
        sub = labels[y:y + h, x:x + w]       # a view: writing to it changes labels
        mine = sub == obj
        cut = (mine & (line[y:y + h, x:x + w] == 0)).astype(np.uint8)
        _, cc, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
        areas = stats[:, cv2.CC_STAT_AREA]
        parts = [int(i) for i in np.argsort(-areas) if i != 0 and areas[i] >= SPLIT_MIN_PART]   # 0 = not part of obj
        if len(parts) < 2:
            continue
        piece = np.zeros(cut.shape, np.uint16)
        for k, i in enumerate(parts, 1):
            piece[cc == i] = k
        for _ in range(20):                  # give the cut line and tiny fragments to the neighbouring piece
            todo = mine & (piece == 0)
            grown = cv2.dilate(piece, np.ones((3, 3), np.uint8))
            take = todo & (grown > 0)
            if not take.any():
                break
            piece[take] = grown[take]
        piece[mine & (piece == 0)] = 1
        for k in range(2, len(parts) + 1):
            sub[piece == k] = next_id
            new_ids[next_id] = obj
            next_id += 1
    return new_ids


def brush_roi(p0, p1, r, shape):
    """Boolean mask of a round-capped stroke from p0 to p1, as (x0, y0, x1, y1, sub-mask)."""
    h, w = shape[:2]
    x0, x1 = max(min(p0[0], p1[0]) - r, 0), min(max(p0[0], p1[0]) + r + 1, w)
    y0, y1 = max(min(p0[1], p1[1]) - r, 0), min(max(p0[1], p1[1]) + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    a = (p0[0] - x0, p0[1] - y0)
    b = (p1[0] - x0, p1[1] - y0)
    cv2.line(sub, a, b, 255, 2 * r + 1)
    cv2.circle(sub, a, r, 255, cv2.FILLED)
    cv2.circle(sub, b, r, 255, cv2.FILLED)
    return x0, y0, x1, y1, sub > 0


def flood_region(lab, blocked, seed, tol):
    """Pixels connected to seed whose Lab color is within tol of the seed color. Never includes blocked pixels."""
    h, w = lab.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = blocked
    if mask[seed[1] + 1, seed[0] + 1]:
        return None
    t = (tol, tol, tol)
    cv2.floodFill(lab, mask, seed, 0, t, t, 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    region = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    region = fill_holes(region)
    region[blocked > 0] = 0
    _, cc = cv2.connectedComponents(region, connectivity=4)
    seed_label = cc[seed[1], seed[0]]
    if seed_label == 0:
        return None
    return cc == seed_label


def find_covered(labels):
    """Ids of the objects that touch another object. Objects can't overlap in labels, so touching is the sign
    that one lies on / against another: every object of a touching pair is returned."""
    h, w = labels.shape
    pad = COVER_GAP + 1
    kernel = np.ones((2 * COVER_GAP + 1, 2 * COVER_GAP + 1), np.uint8)
    covered = set()
    for obj in [int(i) for i in np.unique(labels) if i]:
        x, y, bw, bh = cv2.boundingRect(cv2.findNonZero((labels == obj).astype(np.uint8)))
        sub = labels[max(y - pad, 0):min(y + bh + pad, h), max(x - pad, 0):min(x + bw + pad, w)]
        mine = (sub == obj).astype(np.uint8)
        near = sub[(cv2.dilate(mine, kernel) > 0) & (sub != obj) & (sub > 0)]
        for other, count in zip(*np.unique(near, return_counts=True)):
            if count >= COVER_MIN_CONTACT:
                covered |= {obj, int(other)}
    return covered


def labels_to_objects(labels, classes, incomplete=(), letters=None, covered=()):
    letters = letters or {}
    objects = []
    for obj_id in np.unique(labels):
        if obj_id == 0:
            continue
        mask = (labels == obj_id).astype(np.uint8) * 255
        polygons = []
        for c in find_contours(mask):
            poly = cv2.approxPolyDP(c, POLY_EPSILON, True)
            if len(poly) >= 3:
                polygons.append(poly.reshape(-1, 2).tolist())
        if not polygons:
            continue
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        cls = classes.get(int(obj_id), 0) % len(CLASSES)
        obj = {
            "id": int(obj_id),
            "class": cls,
            "class_name": CLASSES[cls],
            "uncompleted": int(obj_id) in incomplete,
            "covered": int(obj_id) in covered}
        if letters.get(int(obj_id)):         # only objects that were given a letter have the field
            obj["letter"] = letters[int(obj_id)]
        obj["bbox"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        obj["polygons"] = polygons
        objects.append(obj)
    return objects


def objects_to_labels(objects, shape):
    labels = np.zeros(shape[:2], np.uint16)
    classes = {}
    incomplete = set()
    covered = set()
    letters = {}
    for o in objects:
        obj_id = int(o["id"])
        for poly in o["polygons"]:
            cv2.fillPoly(labels, [np.array(poly, np.int32).reshape(-1, 1, 2)], obj_id)
        classes[obj_id] = int(o["class"]) % len(CLASSES)
        if o.get("uncompleted"):
            incomplete.add(obj_id)
        if o.get("covered"):
            covered.add(obj_id)
        if o.get("letter"):
            letters[obj_id] = o["letter"]
    return labels, classes, incomplete, letters, covered


_write_lock = threading.Lock()


def make_truth(image_path, shape, labels, classes, incomplete=(), draft_model=None, letters=None, covered=()):
    """The truth.json content. Only an automatic draft has a "draft" field (with the model that made it): saving
    from the window never writes it, so a file without it - or with it removed - has been made / edited by you."""
    h, w = shape[:2]
    data = {"image": Path(image_path).name, "width": w, "height": h}
    if draft_model:
        data["draft"] = {"model": draft_model}
    data["objects"] = labels_to_objects(labels, classes, incomplete, letters, covered)
    return data


def write_truth(path, data):
    """Write a truth.json atomically (a half-written file is never visible)."""
    with _write_lock:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


def file_stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def commit_draft(tmp, final, replace_stamp=None):
    """Move a finished draft into place, unless a truth.json was saved in the meantime (the user's work wins).
    With replace_stamp (file_stamp of an existing file that was judged to be an untouched draft) that file is
    replaced, but only if it is still exactly as it was when it was checked."""
    with _write_lock:
        if final.exists() and (replace_stamp is None or file_stamp(final) != replace_stamp):
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, final)
        return True


def edge_objects(labels):
    """Ids of the objects that touch the top or bottom edge of the image (y == 0 or y + height == image height),
    i.e. objects that are cut off and so not completely in the image."""
    return {int(i) for i in np.unique(np.concatenate((labels[0], labels[-1]))) if i}


def draft_data(path, model=None):
    """The draft truth.json content for one image. Every object gets the first class, and objects touching the
    top / bottom edge are marked uncompleted."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    labels = draft_labels(img, model)
    classes = {int(i): 0 for i in np.unique(labels) if i}
    return make_truth(path, img.shape, labels, classes, edge_objects(labels), effective_model(model))


def draft_to_file(path, out):
    """Make the draft for one image and write it as json to out. Returns (objects, objects marked uncompleted)."""
    data = draft_data(path)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return len(data["objects"]), sum(o["uncompleted"] for o in data["objects"])


def is_untouched_draft(old, fresh):
    """Is old (the content of an unmarked, older truth.json) exactly what the draft procedure gives (fresh)?
    Only then is it certain that nobody edited it. Drafts from before the top / bottom edge rule have no
    uncompleted marks, so a missing mark is fine - a mark that differs from the rule is not."""
    if len(old.get("objects", [])) != len(fresh["objects"]):
        return False
    for a, b in zip(old["objects"], fresh["objects"]):
        if (a["id"], a["class"], a["bbox"], a["polygons"]) != (b["id"], b["class"], b["bbox"], b["polygons"]):
            return False
        if a.get("uncompleted", False) not in (False, b["uncompleted"]):
            return False
        if a.get("letter") or a.get("covered"):
            return False                     # a letter or covered mark was set, so somebody worked on it
    return True


def draft_state(path, target_model):
    """What the image's truth.json is, as far as redrafting goes. Returns (state, file_stamp):
    'missing'  there is no truth.json
    'current'  an untouched draft that was made with target_model: nothing to do
    'old'      an untouched draft made with another model: safe to replace
    'yours'    anything else (edited, saved from the window, or not provably an untouched draft): never touched"""
    final = path.parent / "truth.json"
    if not final.exists():
        return "missing", None
    stamp = file_stamp(final)
    try:
        with open(final) as f:
            old = json.load(f)
    except (OSError, ValueError):
        return "yours", None
    marker = old.get("draft")
    if marker is not None:
        return ("current" if marker.get("model") == target_model else "old"), stamp
    # Older file without a marker: it is only provably an untouched draft if it is exactly what the u2net draft gives.
    if any(o.get("class", 0) != 0 for o in old.get("objects", [])):
        return "yours", None                 # a class was changed, so somebody worked on it
    if is_untouched_draft(old, draft_data(path, LEGACY_DRAFT_MODEL)):
        return ("current" if target_model == LEGACY_DRAFT_MODEL else "old"), stamp
    return "yours", None


def draft_process_main(conn, model):
    """Runs in the child process: for each image path received, make the draft and write it next to the image as
    truth.json.draft, then reply None (ok) or an error text. rembg runs here, so it can never stall the GUI
    (not even while it loads the model, which holds the GIL for seconds)."""
    global REMBG_MODEL
    REMBG_MODEL = model
    while True:
        try:
            path = conn.recv()
        except (EOFError, OSError):
            return
        try:
            path = Path(path)
            draft_to_file(path, path.parent / "truth.json.draft")
            conn.send(None)
        except Exception as e:
            conn.send(f"{type(e).__name__}: {e}")


class DraftWorker(threading.Thread):
    """Background thread that makes a draft truth.json (rembg + connectedComponents) for every image from the
    current one to the end of the list that has none yet. The image being looked at is always done first. The
    heavy work is done by a child process (draft_process_main); this thread only schedules and waits for it, so
    the GUI is never blocked. Drafts go straight to disk, so nothing piles up in memory."""
    MAX_FAILS_IN_A_ROW = 3

    def __init__(self, on_change):
        super().__init__(daemon=True)
        self.proc = self.conn = None        # the child process and our end of the pipe to it
        self.on_change = on_change          # called (from this thread) whenever the status changed
        self.cond = threading.Condition()
        self.files, self.cursor = [], 0
        self.dirty = False                  # position changed since the last scan
        self.stopped = False
        self.done, self.failed = set(), {}  # paths drafted / failed (path -> error text) in this session
        self.current = None                 # path being drafted right now
        self.left = -1                      # images still without a draft (-1 = not scanned yet)
        self.fatal = ""
        self.only_current = False           # only draft the image being looked at (a redraft job does the rest)
        self.restart = False                # the child process was killed on purpose (new model)

    def set_position(self, files, cursor):
        with self.cond:
            self.files, self.cursor, self.dirty = files, cursor, True
            self.cond.notify()

    def set_only_current(self, flag):
        with self.cond:
            self.only_current, self.dirty = flag, True
            self.cond.notify()

    def restart_child(self):
        """Use a fresh child process (it loads REMBG_MODEL when it starts). A draft that is running is redone."""
        with self.cond:
            self.restart = True
            self.cond.notify()
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify()
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()           # abandon a draft that is mid-run; its half-written file is never used

    def status(self, path):
        with self.cond:
            if path in self.done:
                return "done"
            if path in self.failed:
                return "failed"
            return "working" if path == self.current else "queued"

    def draft_in_child(self, path):
        """Make one draft in the child process (started on first use). Raises if it failed."""
        if self.proc is None or not self.proc.is_alive():
            ctx = multiprocessing.get_context("spawn")
            self.conn, child_end = ctx.Pipe()
            self.proc = ctx.Process(target=draft_process_main, args=(child_end, REMBG_MODEL), daemon=True)
            self.proc.start()
            child_end.close()
        self.conn.send(str(path))
        error = self.conn.recv()            # blocks this thread only (and releases the GIL) until the child is done
        if error:
            raise RuntimeError(error)
        commit_draft(path.parent / "truth.json.draft", path.parent / "truth.json")

    def run(self):
        fails = 0
        while True:
            with self.cond:
                if self.stopped:
                    return
                files, cursor, self.dirty = self.files, self.cursor, False
                if not files:               # nothing to look at yet: wait for the first set_position
                    self.cond.wait()
                    continue
            todo = [p for p in files[cursor:] if p not in self.failed and not (p.parent / "truth.json").exists()]
            if self.only_current and todo and todo[0] != files[cursor]:
                todo = []
            with self.cond:
                self.left = len(todo)
                if not todo:
                    self.current = None
                    if not self.dirty and not self.stopped:
                        self.cond.wait()    # until the user moves to another image / folder
                    continue
                self.current = path = todo[0]
                self.restart = False
            self.on_change()
            try:
                self.draft_in_child(path)
            except Exception as e:
                if self.stopped:
                    return
                if self.restart:            # the child was killed on purpose: just do this image again
                    continue
                with self.cond:
                    self.failed[path] = f"{type(e).__name__}: {e}"
                fails += 1
                if fails >= self.MAX_FAILS_IN_A_ROW:
                    self.fatal = f"{fails} failures in a row, last: {self.failed[path]}"
                    self.current = None
                    self.on_change()
                    return
            else:
                with self.cond:
                    self.done.add(path)
                fails = 0
            self.on_change()


class RedraftJob:
    """Runs `groundTruth.py <test folders> --headless --redraft [--dry-run]` in a separate process, with the same
    Python (and so the same rembg and model) as this window, so the window stays smooth. Everything it prints
    is also written to REDRAFT_LOG; the last line (progress) and the final summary are shown in the sidebar."""

    def __init__(self, folders, model, dry_run, on_change, on_finish):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *map(str, folders), "--headless", "--redraft"]
        if dry_run:
            cmd.append("--dry-run")
        if model:
            cmd += ["--model", model]
        self.dry_run, self.on_change, self.on_finish = dry_run, on_change, on_finish
        self.last, self.summary, self.finished, self.code = "starting...", "", False, None
        self.log = open(REDRAFT_LOG, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1,
                                     env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self.read_output, daemon=True).start()

    def read_output(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.log.write(line + "\n")
            self.log.flush()
            if line.strip():
                self.last = line.strip()
            if line.startswith(("Done", "Interrupted", "Stopping", "This rembg")):
                self.summary = line
            self.on_change()
        self.code = self.proc.wait()
        self.finished = True
        self.log.close()
        self.on_finish()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()


class CoverJob(threading.Thread):
    """Background thread that marks the covered objects (see find_covered) in every image of the editor, like a
    redraft but for the "covered" marks. It only ever adds marks, so it is safe on edited truth.json files too.
    The image shown in the window is changed in memory (it is saved with the window's own save); every other
    image is read, changed and written back. All of that happens under editor.io_lock, the lock that saving,
    loading and switching images also hold, so the window and this job never overwrite each other's work."""

    def __init__(self, editor):
        super().__init__(daemon=True)
        self.editor, self.files = editor, list(editor.files)
        self.checked = self.images = self.objects = 0   # images checked / changed, objects marked
        self.finished = self.stopped = False
        self.error = ""

    def stop(self):
        self.stopped = True

    def summary(self):
        text = f"{self.checked}/{len(self.files)} images checked, {self.objects} objects marked in {self.images} images"
        return f"Stopped: {text}" if self.stopped else f"Error: {self.error}" if self.error else text

    def run(self):
        ed = self.editor
        try:
            for path in self.files:
                if self.stopped:
                    break
                with ed.io_lock:
                    marked = self.mark_in_memory() if path == ed.files[ed.index] else self.mark_in_file(path)
                self.checked += 1
                if marked:
                    self.images += 1
                    self.objects += marked
                if marked or self.checked % 20 == 0:
                    ed.redraw = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        self.finished = True
        ed.redraw = True

    def mark_in_memory(self):
        ed = self.editor
        if ed.awaiting_draft:
            return 0
        found = find_covered(ed.labels) - ed.covered
        ed.covered |= found
        if found:
            ed.touch()
        return len(found)

    def mark_in_file(self, path):
        truth = path.parent / "truth.json"
        if not truth.exists():
            return 0
        with open(truth) as f:
            data = json.load(f)
        labels = objects_to_labels(data["objects"], (data["height"], data["width"]))[0]
        found = find_covered(labels) - {o["id"] for o in data["objects"] if o.get("covered")}
        if not found:
            return 0
        for o in data["objects"]:
            o["covered"] = o.get("covered", False) or o["id"] in found
        data.pop("draft", None)              # it has been worked on now, so a redraft must not replace it
        write_truth(truth, data)
        return len(found)


def find_images(folder):
    folder = Path(folder)
    if (folder / "org.png").exists():
        return [folder / "org.png"]
    return sorted(folder.glob("*/org.png"))


def find_test_folders(folder, max_depth=3):
    """The test folders in or under folder. A test folder is one that holds image folders (<folder>\\image_*\\org.png),
    or is itself an image folder (<folder>\\org.png). So a test folder gives itself, and a parent such as data\\out
    gives all the test folders in it (it is not searched further below a test folder)."""
    def has_images(d):
        return (d / "org.png").exists() or any(d.glob("*/org.png"))

    def walk(d, depth):
        if has_images(d):
            return [d]
        if depth >= max_depth:
            return []
        return [t for sub in sorted(p for p in d.iterdir() if p.is_dir()) for t in walk(sub, depth + 1)]

    folder = Path(folder)
    return walk(folder, 0) if folder.is_dir() else []


def images_in(test_folders):
    return [p for folder in test_folders for p in find_images(folder)]


def resolve_folders(args):
    """Folder arguments (test folders, parents like data\\out, wildcards) as a list of test folders. Prints a note
    for every argument that gives nothing."""
    tests = []
    for folder in expand_folders(args):
        found = find_test_folders(folder)
        if not found:
            print(f"No org.png images found in {folder}")
        tests += [t for t in found if t not in tests]
    return tests


def options_dialog(folders=(), start=1, model=None, redraft=False, dry_run=False, cover=False):
    """A window with the same options as the command line: folders (several, a parent such as data\\out, wildcards),
    start image, rembg model, redraft, dry run and covered marking. Returns a dict with those, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    try:
        from rembg.sessions import sessions_names
    except (ImportError, AttributeError):
        sessions_names = []
    default_label = f"(default: {rembg_default_model()})"
    result = {}

    root = tk.Tk()
    root.title("Folders, model and drafts")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    pad = {"padx": 8, "pady": 4}

    ttk.Label(root, text="Folders - a test folder, or a parent such as data\\out (all test folders in it):").grid(
        row=0, column=0, columnspan=3, sticky="w", **pad)
    listbox = tk.Listbox(root, width=80, height=7, selectmode="extended")
    listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", **pad)
    for folder in folders:
        listbox.insert("end", str(folder))
    buttons = ttk.Frame(root)
    buttons.grid(row=1, column=2, sticky="n", **pad)

    def add(path):
        path = path.strip().strip('"')
        if path and path not in listbox.get(0, "end"):
            listbox.insert("end", path)

    def browse():
        last = listbox.get("end") if listbox.size() else ""
        start_dir = Path(last).parent if last and Path(last).parent.exists() else Path("data/out")
        chosen = filedialog.askdirectory(parent=root, title="Add a folder",
                                         initialdir=str(start_dir.resolve()) if start_dir.exists() else None)
        if chosen:
            add(str(Path(chosen)))

    ttk.Button(buttons, text="Browse...", command=browse).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Remove selected",
               command=lambda: [listbox.delete(i) for i in reversed(listbox.curselection())]).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Clear", command=lambda: listbox.delete(0, "end")).pack(fill="x", pady=2)

    typed = tk.StringVar()
    entry = ttk.Entry(root, textvariable=typed, width=80)
    entry.grid(row=2, column=0, columnspan=2, sticky="we", **pad)
    add_typed = lambda *_: (add(typed.get()), typed.set(""))
    entry.bind("<Return>", add_typed)
    ttk.Button(root, text="Add typed path / wildcard", command=add_typed).grid(row=2, column=2, sticky="we", **pad)

    start_var, model_var = tk.IntVar(value=max(1, start)), tk.StringVar(value=model or default_label)
    redraft_var, dry_var = tk.BooleanVar(value=redraft), tk.BooleanVar(value=dry_run)
    cover_var = tk.BooleanVar(value=cover)
    ttk.Label(root, text="Start at image number").grid(row=3, column=0, sticky="w", **pad)
    ttk.Spinbox(root, from_=1, to=10 ** 7, textvariable=start_var, width=10).grid(row=3, column=1, sticky="w", **pad)
    ttk.Label(root, text="rembg model").grid(row=4, column=0, sticky="w", **pad)
    ttk.Combobox(root, textvariable=model_var, values=[default_label] + list(sessions_names), width=30).grid(
        row=4, column=1, sticky="w", **pad)

    dry_check = ttk.Checkbutton(root, text="Dry run: only report what would change, write nothing", variable=dry_var)
    sync = lambda: (dry_check.state(["!disabled"] if redraft_var.get() else ["disabled"]),
                    None if redraft_var.get() else dry_var.set(False))
    ttk.Checkbutton(root, variable=redraft_var, command=sync,
                    text="Redraft: replace drafts nobody has edited that were made by another model\n"
                         "(anything you made or edited is never touched; runs in the background)").grid(
        row=5, column=0, columnspan=3, sticky="w", **pad)
    dry_check.grid(row=6, column=0, columnspan=3, sticky="w", padx=30, pady=2)
    ttk.Checkbutton(root, variable=cover_var,
                    text="Mark covered objects in all images: every object that touches another one\n"
                         "(only adds marks, edited files are kept; runs in the background)").grid(
        row=7, column=0, columnspan=3, sticky="w", **pad)
    sync()

    def ok():
        if not listbox.size():
            messagebox.showinfo("Folders", "Add at least one folder.", parent=root)
            return
        chosen_model = model_var.get().strip()
        result.update(folders=[Path(p) for p in listbox.get(0, "end")], start=max(1, int(start_var.get())),
                      model=None if not chosen_model or chosen_model == default_label else chosen_model,
                      redraft=redraft_var.get(), dry_run=dry_var.get(), cover=cover_var.get())
        root.destroy()

    row = ttk.Frame(root)
    row.grid(row=8, column=0, columnspan=3, sticky="e", **pad)
    ttk.Button(row, text="Open", command=ok).pack(side="left", padx=4)
    ttk.Button(row, text="Cancel", command=root.destroy).pack(side="left", padx=4)
    root.mainloop()
    return result or None


class Editor:
    def __init__(self, folders, files, start=0):
        self.folders = [Path(f) for f in ([folders] if isinstance(folders, (str, Path)) else folders)]
        self.files = files
        self.job = None              # the RedraftJob started from the window (if any)
        self.cover_job = None        # the CoverJob started from the window (if any)
        self.io_lock = threading.RLock()   # held while saving / loading / switching images (see CoverJob)
        self.index = start
        self.merge_sel = None        # object picked as the target of a combine
        self.message = ""
        self.mode = "brush"          # "brush", "fill" or "split"
        self.default_class = 0       # class given to new objects
        self.brush = BRUSH_DEFAULT
        self.tol = TOLERANCE_DEFAULT
        self.pos = (0, 0)            # mouse in org.png pixels
        self.view_pos = (0, 0)       # mouse in window pixels
        self.inside = False          # is the mouse over the image (and not over the sidebar)?
        self.stroke = None           # None, or {"button": "L"/"R", "obj": id, "last": (x, y)}
        self.cut = None              # None, or the split line being dragged: {"p0": (x, y), "p1": (x, y)}
        self.poly = []               # points (org.png pixels) of the polygon / split path being drawn
        self.awaiting_draft = False  # this image has no truth.json yet and its draft is still being made
        self.quit = False
        self.worker = DraftWorker(lambda: setattr(self, "redraw", True))
        self.worker.start()
        self.load()

    # ---------- image / file handling ----------
    def truth_path(self):
        return self.files[self.index].parent / "truth.json"

    def load(self):
        path = self.files[self.index]
        self.img = cv2.imread(str(path))
        if self.img is None:
            raise RuntimeError(f"Could not read {path}")
        h, w = self.img.shape[:2]
        self.scale = min(MAX_VIEW[0] / w, MAX_VIEW[1] / h, 1.0)
        self.vw, self.vh = int(w * self.scale), int(h * self.scale)
        self.view_base = cv2.resize(self.img, (self.vw, self.vh), interpolation=cv2.INTER_AREA)
        self.lab = cv2.cvtColor(cv2.GaussianBlur(self.img, (5, 5), 0), cv2.COLOR_BGR2LAB)

        self.labels = np.zeros((h, w), np.uint16)
        self.classes = {}
        self.incomplete = set()      # ids of objects marked as uncompleted
        self.covered = set()         # ids of objects marked as covered (partly hidden by another object)
        self.letters = {}            # object id -> the letter on the object (only objects that were given one)
        self.letter_target = None    # id of the object that is waiting for a letter key (see set_letter)
        self.changed = False
        self.merge_sel = None
        self.stroke = None
        self.cut = None
        self.poly = []
        self.message = ""
        self.awaiting_draft = not self.truth_path().exists()
        if not self.awaiting_draft:
            self.read_truth()
        self.next_id = int(self.labels.max()) + 1
        self.overlay_dirty = True
        self.redraw = True
        self.worker.set_position(self.files, self.index)   # drafts continue from here to the last image

    def read_truth(self):
        with open(self.truth_path()) as f:
            self.labels, self.classes, self.incomplete, self.letters, self.covered = objects_to_labels(
                json.load(f)["objects"], self.img.shape)

    def poll_draft(self):
        """Picks up the background draft of the current image as soon as it is on disk. Never blocks."""
        if not self.awaiting_draft or self.stroke or self.cut:
            return
        if self.truth_path().exists():
            self.awaiting_draft = False
            if self.changed:
                self.message = "Draft skipped: you had already started editing this image"
            else:
                self.read_truth()
                self.next_id = int(self.labels.max()) + 1
                self.message = ""
                self.overlay_dirty = True
            self.redraw = True
            return
        w = self.worker
        status = w.status(self.files[self.index])
        if status == "failed":
            msg = f"Could not make a draft ({w.failed[self.files[self.index]]})"
        elif w.fatal:
            msg = f"Draft worker stopped ({w.fatal})"
        else:
            msg = "Making a draft in the background..."
        if msg != self.message:
            self.message = msg
            self.redraw = True

    def save(self):
        with self.io_lock:
            if not self.changed and (self.truth_path().exists() or self.awaiting_draft):
                return               # nothing new to save (and never store an empty file that blocks the draft)
            write_truth(self.truth_path(), make_truth(self.files[self.index], self.img.shape,
                                                      self.labels, self.classes, self.incomplete,
                                                      letters=self.letters, covered=self.covered))
            self.changed = False

    def goto(self, index):
        with self.io_lock:
            self.save()
            index = max(0, min(len(self.files) - 1, index))
            if index != self.index:
                self.index = index
                self.load()

    def open_dialog(self):
        """The same options as the command line, in a window: folders, start image, model, redraft, dry run."""
        options = options_dialog(self.folders, self.index + 1, REMBG_MODEL)
        if options is not None:
            self.apply_options(options)

    def apply_options(self, options):
        tests = resolve_folders([str(f) for f in options["folders"]])
        files = images_in(tests)
        if not files:
            self.message = "No org.png images found in the chosen folders"
            self.redraw = True
            return
        if self.cover_job is not None and not self.cover_job.finished:
            self.cover_job.stop()            # it works on the old image list
        with self.io_lock:
            self.save()
            self.set_model(options["model"])
            self.folders, self.files = tests, files
            self.index = max(0, min(len(files) - 1, options["start"] - 1))
            self.message = ""
            self.load()
        if options["redraft"]:
            self.start_redraft(options["dry_run"])
        if options["cover"]:
            self.start_cover_all()

    def start_cover_all(self):
        """Mark the covered objects in all images in the background (see CoverJob)."""
        if self.cover_job is not None and not self.cover_job.finished:
            self.cover_job.stop()
        self.cover_job = CoverJob(self)
        self.cover_job.start()

    def set_model(self, model):
        """Use another rembg model for the drafts made from now on (None = rembg's default)."""
        global REMBG_MODEL
        if (model or None) != REMBG_MODEL:
            REMBG_MODEL = model or None
            self.worker.restart_child()

    def start_redraft(self, dry_run=False):
        """Redraft in the background (see RedraftJob). While it runs, the window's own drafting only does the image
        you are looking at, so the two don't do the same work twice."""
        if self.job is not None and not self.job.finished:
            self.job.stop()
        self.job = RedraftJob(self.folders, REMBG_MODEL, dry_run, lambda: setattr(self, "redraw", True),
                              self.redraft_finished)
        if not dry_run:
            self.worker.set_only_current(True)

    def redraft_finished(self):
        self.worker.set_only_current(False)
        self.redraw = True

    # ---------- editing ----------
    def prune(self):
        present = set(np.unique(self.labels).tolist())
        for obj_id in list(self.classes):
            if obj_id not in present:
                del self.classes[obj_id]
        self.incomplete &= present
        self.covered &= present
        self.letters = {k: v for k, v in self.letters.items() if k in present}
        if self.merge_sel not in present:
            self.merge_sel = None
        if self.letter_target not in present:
            self.letter_target = None

    def hover_obj(self):
        """Id of the object under the mouse (0 = none, or the mouse is over the sidebar)."""
        return int(self.labels[self.pos[1], self.pos[0]]) if self.inside else 0

    def toggle_incomplete(self):
        """Mark / unmark the hovered object as uncompleted. Its class is left unchanged."""
        obj = self.hover_obj()
        if obj:
            self.incomplete ^= {obj}
            self.touch()

    def toggle_covered(self):
        """Mark / unmark the hovered object as covered (partly hidden by another object)."""
        obj = self.hover_obj()
        if obj:
            self.covered ^= {obj}
            self.touch()

    def auto_covered(self):
        """Mark the objects that look covered (see find_covered). Marks that are already set are never removed."""
        found = find_covered(self.labels) - self.covered
        self.covered |= found
        self.message = f"Marked {len(found)} object(s) as covered" if found else "No new covered objects found"
        if found:
            self.touch()
        self.redraw = True

    def start_letter(self):
        """L: the next letter or digit key that is pressed becomes the letter on the hovered object."""
        obj = self.hover_obj()
        if not obj:
            self.message = "Hover an object first, then press L"
        else:
            self.letter_target = obj
        self.redraw = True

    def letter_key(self, raw, key):
        """A key was pressed while an object waits for its letter: a letter / digit sets it, Backspace or Delete
        removes the letter, Esc cancels. Anything else is ignored."""
        if key == 27:
            self.set_letter(self.letter_target, self.letters.get(self.letter_target))
        elif key == 8 or raw == 3014656:                    # Backspace / Delete
            self.set_letter(self.letter_target, None)
        elif 32 < key < 127 and chr(key).isalnum():
            self.set_letter(self.letter_target, chr(key).upper())

    def set_letter(self, obj, letter):
        """Give the object a letter (None removes it) and stop waiting for a letter key."""
        self.letter_target = None
        if self.letters.get(obj) != letter:
            if letter:
                self.letters[obj] = letter
            else:
                self.letters.pop(obj, None)
            self.touch()
        self.redraw = True

    def combine(self):
        """First press over an object picks it, second press over another object merges that one into the first."""
        obj = self.hover_obj()
        if self.merge_sel is None:
            if obj:
                self.merge_sel = obj
        elif obj and obj != self.merge_sel:
            self.labels[self.labels == obj] = self.merge_sel
            if obj in self.incomplete:
                self.incomplete.add(self.merge_sel)   # combined object stays uncompleted if either part was
            if obj in self.covered:
                self.covered.add(self.merge_sel)      # ...and covered
            if obj in self.letters and self.merge_sel not in self.letters:
                self.letters[self.merge_sel] = self.letters[obj]   # ...and keeps a letter (the first one's wins)
            self.merge_sel = None
            self.prune()
            self.touch()
        else:
            self.merge_sel = None    # pressed on empty space or on the same object: cancel
        self.redraw = True

    def touch(self):
        self.changed = True
        self.overlay_dirty = True
        self.redraw = True

    def apply_stroke(self, p0, p1):
        roi = brush_roi(p0, p1, int(self.brush), self.labels.shape)
        if roi is None:
            return
        x0, y0, x1, y1, sub = roi
        region = self.labels[y0:y1, x0:x1]
        if self.stroke["button"] == "L":
            obj = self.stroke["obj"]
            if obj == 0:
                obj = self.stroke["obj"] = self.next_id
                self.next_id += 1
                self.classes[obj] = self.default_class
            region[sub & ((region == 0) | (region == obj))] = obj
        else:
            region[sub] = 0
        self.touch()

    def fill_at(self, p):
        region = flood_region(self.lab, (self.labels > 0).astype(np.uint8), p, int(self.tol))
        if region is None:
            return
        self.labels[region] = self.next_id
        self.classes[self.next_id] = self.default_class
        self.next_id += 1
        self.touch()

    def delete_at(self, p):
        obj = self.labels[p[1], p[0]]
        if obj:
            self.labels[self.labels == obj] = 0
            self.prune()
            self.touch()

    def set_mode(self, mode):
        """Switch to the given mode, or back to brush if it is already the active one."""
        self.mode = "brush" if self.mode == mode else mode
        self.cut = None
        self.poly = []
        self.redraw = True

    def near_point(self, point):
        """Is the mouse within POLY_CLOSE_PX window pixels of point (org.png pixels)?"""
        return np.hypot((point[0] - self.pos[0]) * self.scale, (point[1] - self.pos[1]) * self.scale) <= POLY_CLOSE_PX

    def poly_click(self):
        """Add a point at the mouse. Clicking the first point again closes the polygon / split path."""
        if len(self.poly) >= 3 and self.near_point(self.poly[0]):
            self.finish_points(closed=True)
            return
        if not self.poly or self.poly[-1] != self.pos:
            self.poly.append(self.pos)
        self.redraw = True

    def finish_points(self, closed=False):
        """Enter / closing click: fill the polygon, or make the split along the clicked path."""
        if self.mode == "split":
            self.split(self.poly, closed)
            self.poly = []
        else:
            self.finish_polygon()

    def finish_polygon(self):
        """Fill the polygon. Like the brush: if its first point is on an object the polygon is added to that
        object, otherwise it becomes a new object. It never takes pixels from other objects."""
        if len(self.poly) < 3:
            self.message = "A polygon needs at least 3 points"
            self.redraw = True
            return
        mask = np.zeros(self.labels.shape, np.uint8)
        cv2.fillPoly(mask, [np.array(self.poly, np.int32).reshape(-1, 1, 2)], 255)
        obj = int(self.labels[self.poly[0][1], self.poly[0][0]])
        if obj == 0:
            obj = self.next_id
            self.next_id += 1
            self.classes[obj] = self.default_class
        self.labels[(mask > 0) & ((self.labels == 0) | (self.labels == obj))] = obj
        self.poly = []
        self.prune()
        self.touch()

    def split(self, points, closed=False):
        """Split the object(s) touched by the path through points. Every piece keeps the class and uncompleted state."""
        if len(points) < 2:
            self.message = "A cut needs at least 2 points"
            self.redraw = True
            return
        new_ids = split_by_path(self.labels, points, closed, self.next_id)
        if not new_ids:
            self.message = "Nothing split: the cut must run completely across an object (or close around a part)"
            self.redraw = True
            return
        self.next_id = max(new_ids) + 1
        for new, old in new_ids.items():
            self.classes[new] = self.classes.get(old, 0)
            if old in self.incomplete:
                self.incomplete.add(new)
            if old in self.covered:
                self.covered.add(new)
            if old in self.letters:
                self.letters[new] = self.letters[old]
        self.touch()

    def set_class(self, cls=None):
        """Set the class of the hovered object (or, over empty space, the class for new objects).
        With cls=None the class is cycled to the next one."""
        obj = self.hover_obj()
        if obj:
            self.classes[obj] = (self.classes.get(obj, 0) + 1) % len(CLASSES) if cls is None else cls
            self.touch()
        else:
            self.default_class = (self.default_class + 1) % len(CLASSES) if cls is None else cls
            self.redraw = True

    # ---------- mouse ----------
    def on_mouse(self, event, x, y, flags, _):
        self.view_pos = (x, y)
        self.inside = 0 <= x < self.vw and 0 <= y < self.vh
        self.pos = (min(max(int(x / self.scale), 0), self.img.shape[1] - 1),     # clamped, so a drag that
                    min(max(int(y / self.scale), 0), self.img.shape[0] - 1))     # leaves the image still works
        self.redraw = True

        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                up = cv2.getMouseWheelDelta(flags) > 0
            except AttributeError:
                up = (flags >> 16 if flags >> 16 < 32768 else (flags >> 16) - 65536) > 0
            if self.mode == "brush":
                step = max(1, round(self.brush * 0.15))   # at least 1px so small sizes can't get stuck
                self.brush = min(400, max(2, self.brush + (step if up else -step)))
            elif self.mode == "fill":
                self.tol = min(100, max(1, self.tol + (1 if up else -1)))
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and self.inside:   # clicks on the sidebar do nothing
            left = event == cv2.EVENT_LBUTTONDOWN
            if self.mode == "split":
                self.message = ""
                if left:
                    self.cut = {"p0": self.pos, "p1": self.pos}
                elif self.cut:
                    self.cut = None          # right click cancels a cut being dragged
                elif self.poly:
                    self.poly.pop()          # ...else removes the last point of the path
                    self.redraw = True
            elif self.mode == "polygon":
                self.message = ""
                if left:
                    self.poly_click()
                elif self.poly:
                    self.poly.pop()          # right click removes the last point
                    self.redraw = True
            elif self.mode == "fill":
                self.fill_at(self.pos) if left else self.delete_at(self.pos)
            else:
                self.stroke = {"button": "L" if left else "R", "obj": int(self.labels[self.pos[1], self.pos[0]]),
                               "last": self.pos}
                self.apply_stroke(self.pos, self.pos)
        elif event == cv2.EVENT_MOUSEMOVE and self.cut:
            self.cut["p1"] = self.pos
        elif event == cv2.EVENT_LBUTTONUP and self.cut:
            p0, self.cut = self.cut["p0"], None
            dragged = np.hypot((p0[0] - self.pos[0]) * self.scale, (p0[1] - self.pos[1]) * self.scale) > POLY_CLOSE_PX
            if dragged and not self.poly:
                self.split([p0, self.pos])   # a drag is a straight cut
            else:
                self.poly_click()            # a click adds a point to the path
        elif event == cv2.EVENT_MOUSEMOVE and self.stroke:
            self.apply_stroke(self.stroke["last"], self.pos)
            self.stroke["last"] = self.pos
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.stroke:
            self.stroke = None
            self.prune()

    # ---------- drawing ----------
    def build_overlay(self):
        base = self.view_base
        self.small = cv2.resize(self.labels, (self.vw, self.vh), interpolation=cv2.INTER_NEAREST)
        lut = np.zeros((int(self.labels.max()) + 1, 3), np.uint8)
        for obj_id, cls in self.classes.items():
            if obj_id < len(lut):
                lut[obj_id] = PALETTE[cls % len(PALETTE)]
        col = lut[self.small]
        m = self.small > 0
        out = base.copy()
        m_tint = m
        if self.incomplete or self.covered:
            # uncompleted objects are shown hatched (diagonal stripes) instead of evenly tinted, covered ones with
            # stripes in the other direction
            yy, xx = np.indices(m.shape)
            if self.incomplete:
                m_tint = m_tint & ~(np.isin(self.small, list(self.incomplete)) & ((xx + yy) % 16 >= 8))
            if self.covered:
                m_tint = m_tint & ~(np.isin(self.small, list(self.covered)) & ((xx - yy) % 16 >= 8))
        out[m_tint] = (base[m_tint] * 0.55 + col[m_tint] * 0.45).astype(np.uint8)
        edge =np.zeros(m.shape, bool)
        edge[:, 1:] |= self.small[:, 1:] != self.small[:, :-1]
        edge[:, :-1] |= self.small[:, 1:] != self.small[:, :-1]
        edge[1:, :] |= self.small[1:] != self.small[:-1]
        edge[:-1, :] |= self.small[1:] != self.small[:-1]
        edge &= m
        out[edge] = col[edge]
        self.letter_pos = {}                 # where to draw each object's letter: the middle of the object
        for obj_id in self.letters:
            ys, xs = np.nonzero(self.small == obj_id)
            if len(xs):
                self.letter_pos[obj_id] = (int(xs.mean()), int(ys.mean()))
        self.overlay = out
        self.overlay_dirty = False

    def draw_sidebar(self, hover, height):
        """All the text lives here: a panel to the right of the image."""
        panel = np.full((height, SIDEBAR_W, 3), (38, 38, 38), np.uint8)
        font, pad, line_h, scale = cv2.FONT_HERSHEY_SIMPLEX, 14, 18, 0.5
        y = 6

        def text(s, color=(235, 235, 235), indent=0):
            nonlocal y
            for line in wrap_text(s, SIDEBAR_W - 2 * pad - indent, scale):
                y += line_h
                cv2.putText(panel, line, (pad + indent, y), font, scale, color, 1, cv2.LINE_AA)

        def heading(s):
            nonlocal y
            y += 8
            text(s.upper(), (150, 150, 150))

        GREEN, ORANGE, YELLOW = (90, 200, 90), (0, 165, 255), (0, 220, 255)
        w = self.worker
        heading("image")
        text(self.files[self.index].parent.parent.name)           # the test folder of this image
        text(f"{self.index + 1} / {len(self.files)}   " + ("UNSAVED" if self.changed else "saved"),
             ORANGE if self.changed else GREEN)
        text(self.files[self.index].parent.name, (180, 180, 180))

        heading("notes")
        notes = []
        if self.awaiting_draft and not self.message:
            notes.append("Making a draft in the background...")
        if self.message:
            notes.append(self.message)
        if self.merge_sel is not None:
            notes.append(f"COMBINE: object {self.merge_sel} is picked. Hover another object and press C "
                         f"(C on empty space cancels).")
        if w.fatal:
            notes.append(f"Draft worker stopped: {w.fatal}")
        elif w.current:
            notes.append(f"Drafts: working on ...{w.current.parent.name[-6:]}, {w.left} left")
        elif w.left < 0:
            notes.append("Drafts: starting...")
        else:
            notes.append("Drafts: all done")
        notes.append(f"Model: {effective_model()}")
        if self.letter_target is not None:
            notes.append(f"LETTER for object {self.letter_target}: press a letter or digit "
                         f"(Backspace removes it, Esc cancels).")
        for n in notes:
            text(n, YELLOW if n.startswith(("COMBINE", "LETTER", "Draft worker", "Could not", "Nothing", "Hover"))
                 else (200, 200, 200))

        if self.job is not None:
            heading("dry run" if self.job.dry_run else "redraft")
            if self.job.finished:
                text(self.job.summary or f"Stopped (exit code {self.job.code}), see the log.", GREEN)
                text(f"Details: {REDRAFT_LOG.name}", (170, 170, 170))
            else:
                text(self.job.last, (200, 200, 200))

        if self.cover_job is not None:
            heading("covered marking")
            text(self.cover_job.summary() if self.cover_job.finished else self.cover_job.summary() + " ...",
                 GREEN if self.cover_job.finished and not self.cover_job.error else (200, 200, 200))

        heading("mode")
        text(self.mode.upper(), (255, 255, 255))
        text({"brush": f"Brush size {self.brush}px (mouse wheel)",
              "fill": f"Tolerance {self.tol} (mouse wheel)",
              "split": f"Cut width {SPLIT_THICKNESS}px, {len(self.poly)} points",
              "polygon": f"{len(self.poly)} points"}[self.mode], (200, 200, 200))
        for line in {"brush": ("Left: paint", "Right: erase"),
                     "polygon": ("Left: add a point (click the first point to close)", "Right / Backspace: remove last point",
                                 "Enter: close the polygon   Esc: cancel"),
                     "fill": ("Left: fill the object under the mouse", "Right: delete the object"),
                     "split": ("Left drag: straight cut across an object",
                               "Left click: add a point to a bent cut (Enter: cut; "
                               "click the first point: closed cut around a part)",
                               "Right / Backspace: remove last point   Esc: cancel")}[self.mode]:
            text(line, (200, 200, 200))

        heading("classes")
        hover_cls = self.classes.get(hover, 0) % len(CLASSES) if hover else None
        for i, name in enumerate(CLASSES):
            y += line_h
            cv2.rectangle(panel, (pad, y - 11), (pad + 12, y + 1), PALETTE[i % len(PALETTE)], cv2.FILLED)
            tags = ("   <- new objects" if i == self.default_class else "") + \
                   ("   <- hovered" if i == hover_cls else "")
            cv2.putText(panel, f"{i + 1}  {name}{tags}", (pad + 22, y), font, scale,
                        (255, 255, 255) if tags else (200, 200, 200), 1, cv2.LINE_AA)

        heading("objects in this image")
        counts = [[0, 0, 0] for _ in CLASSES]            # per class: all, uncompleted, covered
        for obj_id, cls in self.classes.items():
            row = counts[cls % len(CLASSES)]
            row[0] += 1
            row[1] += obj_id in self.incomplete
            row[2] += obj_id in self.covered
        columns = (pad + 22, 250, 295, 340)              # x of the class name and of the three number columns
        y += line_h
        for x, head in zip(columns[1:], ("all", "unc", "cov")):
            cv2.putText(panel, head, (x, y), font, scale, (150, 150, 150), 1, cv2.LINE_AA)
        for name, row in zip(CLASSES + ["total"], counts + [[sum(c) for c in zip(*counts)]]):
            y += line_h
            shade = (200, 200, 200) if row[0] else (110, 110, 110)
            cv2.putText(panel, name, (columns[0], y), font, scale, shade, 1, cv2.LINE_AA)
            for x, n in zip(columns[1:], row):
                cv2.putText(panel, str(n), (x, y), font, scale, shade if n else (110, 110, 110), 1, cv2.LINE_AA)

        heading("hover")
        if hover:
            text(f"Object {hover}: {CLASSES[hover_cls]}" + ("  (UNCOMPLETED)" if hover in self.incomplete else "")
                 + ("  (COVERED)" if hover in self.covered else ""))
            text(f"Letter: {self.letters[hover]}" if hover in self.letters else "No letter", (200, 200, 200))
        else:
            text("-", (150, 150, 150))

        heading("keys")
        for key, what in (("N / B", "next / previous image"), ("O", "folders, model, redraft..."),
                          ("1-9", "set class (hovered object, else new)"), ("Space", "next class"),
                          ("F", "fill mode (again: brush)"), ("S", "split mode (again: brush)"),
                          ("P", "polygon mode (again: brush)"),
                          ("C", "combine two objects"), ("U", "uncompleted on / off"),
                          ("V", "covered on / off"), ("A", "auto-mark covered objects"),
                          ("L", "set the letter on an object"),
                          ("Q / Esc", "save and quit")):
            y += line_h
            cv2.putText(panel, key, (pad, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, what, (pad + 80, y), font, scale, (170, 170, 170), 1, cv2.LINE_AA)
        return panel

    def render(self):
        """The whole window: image with overlay on the left, sidebar on the right."""
        if self.overlay_dirty:
            self.build_overlay()
        image = self.overlay.copy()
        hover = self.hover_obj()
        if hover:
            contours = find_contours((self.small == hover).astype(np.uint8))
            cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
        if self.merge_sel is not None:
            contours = find_contours((self.small == self.merge_sel).astype(np.uint8))
            cv2.drawContours(image, contours, -1, (0, 255, 255), 4)
        if self.letter_target is not None:       # the object that is waiting for its letter
            contours = find_contours((self.small == self.letter_target).astype(np.uint8))
            cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
        for obj_id, (cx, cy) in self.letter_pos.items():
            letter = self.letters[obj_id]
            width, height = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)[0]
            origin = (cx - width // 2, cy + height // 2)
            cv2.putText(image, letter, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(image, letter, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        if self.inside:
            vx, vy = self.view_pos
            if self.mode == "brush":
                cv2.circle(image, (vx, vy), max(1, int(self.brush * self.scale)), (255, 255, 255), 1)
            else:
                cv2.drawMarker(image, (vx, vy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        if self.poly:
            pts = [(int(x * self.scale), int(y * self.scale)) for x, y in self.poly]
            if self.inside:
                pts.append(self.view_pos)     # the edge being drawn follows the mouse
            if len(pts) >= 3 and self.mode == "polygon":
                shade = image.copy()
                cv2.fillPoly(shade, [np.array(pts, np.int32).reshape(-1, 1, 2)], PALETTE[self.default_class % len(PALETTE)])
                image = cv2.addWeighted(shade, 0.3, image, 0.7, 0)
            cv2.polylines(image, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(image, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
            for i, p in enumerate(pts[:len(self.poly)]):
                cv2.circle(image, p, POLY_CLOSE_PX if i == 0 and len(self.poly) >= 3 else 4, (0, 0, 0), 3)
                cv2.circle(image, p, POLY_CLOSE_PX if i == 0 and len(self.poly) >= 3 else 4, (255, 255, 255), 1)
        if self.cut:
            a = (int(self.cut["p0"][0] * self.scale), int(self.cut["p0"][1] * self.scale))
            b = (int(self.cut["p1"][0] * self.scale), int(self.cut["p1"][1] * self.scale))
            cv2.line(image, a, b, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(image, a, b, (0, 255, 255), 2, cv2.LINE_AA)

        height = max(self.vh, SIDEBAR_MIN_H)
        frame = np.full((height, self.vw + SIDEBAR_W, 3), (38, 38, 38), np.uint8)
        frame[:self.vh, :self.vw] = image
        frame[:, self.vw:] = self.draw_sidebar(hover, height)
        return frame

    def draw(self):
        cv2.imshow(WINDOW, self.render())
        self.redraw = False

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while not self.quit:
            self.poll_draft()
            if self.redraw:
                self.draw()
            raw = cv2.waitKey(15)
            key = raw & 0xFF
            if self.letter_target is not None:               # an object waits for its letter: keys only go to it
                if raw != -1:
                    self.letter_key(raw, key)
            elif key in (ord("l"), ord("L")):
                self.start_letter()
            elif key in (ord("n"), ord("N")):
                self.goto(self.index + 1)
            elif key in (ord("b"), ord("B")):
                self.goto(self.index - 1)
            elif key in (ord("f"), ord("F")):
                self.set_mode("fill")
            elif key in (ord("s"), ord("S")):
                self.set_mode("split")
            elif key in (ord("p"), ord("P")):
                self.set_mode("polygon")
            elif key == 13 and self.poly:                    # Enter closes the polygon / makes the split
                self.finish_points()
            elif key == 8 and self.poly:                     # Backspace removes the last point
                self.poly.pop()
                self.redraw = True
            elif key == 27 and self.poly:                    # Esc cancels the polygon instead of quitting
                self.poly = []
                self.redraw = True
            elif key == 32:
                self.set_class()
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(CLASSES):
                self.set_class(key - ord("1"))
            elif key in (ord("c"), ord("C")):
                self.combine()
            elif key in (ord("u"), ord("U")):
                self.toggle_incomplete()
            elif key in (ord("v"), ord("V")):
                self.toggle_covered()
            elif key in (ord("a"), ord("A")):
                self.auto_covered()
            elif key in (ord("o"), ord("O")):
                self.open_dialog()
            elif key in (ord("q"), ord("Q"), 27):
                self.quit = True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True
        if self.job is not None:
            self.job.stop()          # a redraft started from the window ends with the window
        if self.cover_job is not None:
            self.cover_job.stop()
            self.cover_job.join(5)   # it may be in the middle of writing a file
        self.worker.stop()           # also kills the rembg child process, so quitting is instant
        self.save()
        cv2.destroyAllWindows()


def fix_quoted_args(argv):
    """cmd and PowerShell turn  "folder\\"  into  folder" --headless  : the backslash escapes the closing quote, so
    the rest of the command line ends up inside the folder argument. Split such an argument again."""
    fixed = []
    for arg in argv:
        if '"' in arg:
            head, _, rest = arg.partition('"')
            if head:
                fixed.append(head)
            fixed += [token.replace('"', "") for token in shlex.split(rest, posix=False)]
        else:
            fixed.append(arg)
    return fixed


def expand_folders(args):
    """Folder arguments as Paths. Wildcards (data\\out\\test*) are expanded here because PowerShell and cmd
    pass them on as they are."""
    folders = []
    for arg in args:
        arg = arg.strip().strip('"')
        if any(c in arg for c in "*?[") and not Path(arg).exists():
            matches = sorted(Path(p) for p in glob.glob(arg) if Path(p).is_dir())
            if not matches:
                print(f"Warning: nothing matches {arg}")
            folders += matches
        else:
            folders.append(Path(arg))
    return folders


def run_headless(folders, start=0, redraft=False, dry_run=False):
    """No window: one folder after the other, make the rembg draft for every image (from number start on) that
    has no truth.json yet. With redraft, untouched drafts made by another model are replaced too (see
    draft_state); anything you made or edited is never touched. With dry_run nothing is written.
    Returns the exit code: 1 if a folder had no images, rembg kept failing or Ctrl+C."""
    target = effective_model()
    try:
        from rembg.sessions import sessions_names
        if target not in sessions_names:
            print(f"This rembg does not know the model '{target}'. Available: {', '.join(sessions_names)}")
            return 1
    except (ImportError, AttributeError):
        pass
    try:
        rembg_version = importlib.metadata.version("rembg")
    except importlib.metadata.PackageNotFoundError:
        rembg_version = "?"
    print(f"rembg model: {target} (rembg {rembg_version}, Python {sys.version.split()[0]})"
          + ("  [dry run: nothing is written]" if dry_run else ""), flush=True)

    code, fails = 0, 0
    total = {"drafted": 0, "replaced": 0, "yours": 0, "current": 0}
    try:
        for k, folder in enumerate(folders, 1):
            files = find_images(folder)
            print(f"=== [{k}/{len(folders)}] {folder}", flush=True)
            if not files:
                print("No org.png images found, skipping this folder", flush=True)
                code = 1
                continue
            images = files[start:]
            print(f"{len(files)} images (from image {start + 1} on)", flush=True)
            counts = dict.fromkeys(total, 0)
            for n, path in enumerate(images, 1):
                final = path.parent / "truth.json"
                stamp = None
                if redraft:
                    state, stamp = draft_state(path, target)
                    if n % 100 == 0:
                        print(f"  ...checked {n}/{len(images)}", flush=True)
                else:
                    state = "missing" if not final.exists() else "yours"   # without redraft nothing existing is touched
                if state in ("yours", "current"):
                    counts[state] += 1
                    continue
                label = "drafted" if state == "missing" else "replaced"
                if dry_run:
                    counts[label] += 1
                    print(f"[{n}/{len(images)}] {path.parent.name}: would be {label}", flush=True)
                    continue
                tmp = path.parent / "truth.json.draft"
                try:
                    objects, incomplete = draft_to_file(path, tmp)
                except Exception as e:
                    fails += 1
                    print(f"[{n}/{len(images)}] {path.parent.name}: FAILED ({type(e).__name__}: {e})", flush=True)
                    if fails >= DraftWorker.MAX_FAILS_IN_A_ROW:
                        print(f"Stopping after {fails} failures in a row.", flush=True)
                        return 1
                    continue
                fails = 0
                saved = commit_draft(tmp, final, stamp)
                counts[label] += saved
                print(f"[{n}/{len(images)}] {path.parent.name}: {label if saved else 'skipped'}, {objects} objects, "
                      f"{incomplete} uncompleted" + ("" if saved else " (the file changed meanwhile, left alone)"),
                      flush=True)
            print(f"  {counts['drafted']} drafted, {counts['replaced']} replaced, {counts['current']} already made "
                  f"with {target}, {counts['yours']} left alone (existing truth.json)", flush=True)
            for key in total:
                total[key] += counts[key]
    except KeyboardInterrupt:
        print(f"Interrupted. {total['drafted'] + total['replaced']} drafts made so far are kept.", flush=True)
        return 1
    print(f"Done: {total['drafted']} drafted, {total['replaced']} replaced, {total['current']} already made with "
          f"{target}, {total['yours']} left alone, in {len(folders)} folder(s).", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description="Paint ground truth (truth.json) on the org.png images of a test folder.")
    parser.add_argument("folders", nargs="*",
                        help="test folder(s) or a parent such as \"data\\out\" (all test folders in it); wildcards "
                             "like \"data\\out\\test*\" work. Without any, the window asks for them")
    parser.add_argument("--start", "-s", type=int, default=1, metavar="N",
                        help="number of the image to start at (default 1). Headless: single test folder only")
    parser.add_argument("--headless", action="store_true",
                        help="no window: make the rembg drafts for all images without a truth.json in every given "
                             "folder, one after the other, then exit (needs at least one folder)")
    parser.add_argument("--model", metavar="NAME", help="rembg model to use (default: REMBG_MODEL in this file, "
                                                        "else rembg's own default), e.g. bria-rmbg or u2net")
    parser.add_argument("--redraft", action="store_true",
                        help="also replace drafts that nobody has edited and that were made by another model than "
                             "the one in use. Anything you made or edited is never touched. With the window this "
                             "runs in the background")
    parser.add_argument("--auto-covered", action="store_true",
                        help="in the window: mark the covered objects (every object that touches another one) in "
                             "all images, in the background. Only adds marks")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --redraft: only list what would be drafted / replaced, write nothing")
    args = parser.parse_args(fix_quoted_args(sys.argv[1:]))
    global REMBG_MODEL
    if args.model:
        REMBG_MODEL = args.model
    if args.dry_run and not args.redraft:
        parser.error("--dry-run needs --redraft")
    if len(args.folders) == 2 and args.folders[1].isdigit() and not Path(args.folders[1]).exists():
        args.start = int(args.folders.pop())    # old form: groundTruth.py <folder> <start image number>
    tests = resolve_folders(args.folders)

    if args.headless:
        if not args.folders:
            parser.error("--headless needs at least one folder")
        if not tests:
            sys.exit(1)
        if args.start != 1 and len(tests) > 1:
            parser.error("--start only works with a single test folder when headless")
        sys.exit(run_headless(tests, max(0, args.start - 1), args.redraft, args.dry_run))

    start, redraft, dry_run, cover = args.start, args.redraft, args.dry_run, args.auto_covered
    while not tests:                 # nothing (usable) given: ask in the window until there is, or it is cancelled
        options = options_dialog(args.folders, start, REMBG_MODEL, redraft, dry_run, cover)
        if options is None:
            return
        REMBG_MODEL = options["model"]
        start, redraft, dry_run, cover = options["start"], options["redraft"], options["dry_run"], options["cover"]
        args.folders = [str(f) for f in options["folders"]]
        tests = resolve_folders(args.folders)
    files = images_in(tests)
    print(f"Found {len(files)} images in {len(tests)} test folder(s)")
    editor = Editor(tests, files, max(0, min(len(files) - 1, start - 1)))
    if redraft:
        editor.start_redraft(dry_run)
    if cover:
        editor.start_cover_all()
    editor.run()


if __name__ == "__main__":
    main()
