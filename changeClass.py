"""Change the class of objects in the truth.json files made by groundTruth.py: every object of one class becomes
another class.

    python changeClass.py "data\\out\\test2 - 2x4 brick" --r --i=0 --o=2
    python changeClass.py "data\\out" --r --i="2x4 brick" --o="gray rock" --dry-run

A class is given by its number (its position in CLASSES in groundTruth.py, counting from 0) or by its name.
Without --r only the truth.json directly in the folder is changed; with --r every truth.json in the folder and in
all folders below it (so the test folders and image folders of data\\out).

Close the groundTruth window first: it keeps its own copy of the image it shows and would write that copy back over
the change when it saves.
"""
import argparse
import ast
import glob
import json
import os
import shlex
import sys
from pathlib import Path


def fix_quoted_args(argv):
    """cmd and PowerShell turn  "folder\\"  into  folder" --r --i=0  : the backslash escapes the closing quote, so the
    rest of the command line ends up inside the folder argument. Split such an argument again."""
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


def load_classes():
    """CLASSES from groundTruth.py (read from the source, so it is always the list you have there)."""
    source_file = Path(__file__).with_name("groundTruth - Tobias.py")
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except OSError as e:
        sys.exit(f"Could not read {source_file}: {e}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    sys.exit(f"Could not find CLASSES in {source_file}")


def parse_class(text, classes, option):
    """A class number or name (any case) as a class number. Exits with a message if it is neither."""
    text = text.strip()
    if text.isdigit() and int(text) < len(classes):
        return int(text)
    for number, name in enumerate(classes):
        if name.lower() == text.lower():
            return number
    listing = ", ".join(f"{number} = {name}" for number, name in enumerate(classes))
    sys.exit(f"{option}: '{text}' is not a class. The classes are: {listing}")


def find_truth_files(folders, recursive):
    files = []
    for folder in folders:
        found = sorted(folder.rglob("truth.json")) if recursive else [p for p in [folder / "truth.json"] if p.exists()]
        files += [p for p in found if p not in files]
    return files


def change_file(path, source, target, classes, dry_run):
    """Change class source to target in one truth.json. Returns how many objects were changed."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for obj in data.get("objects", []):
        if obj.get("class") == source:
            obj["class"] = target
            obj["class_name"] = classes[target]
            changed += 1
    if changed and not dry_run:
        # The file is no longer an untouched automatic draft, so groundTruth.py --redraft must leave it alone.
        data.pop("draft", None)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)            # atomic: a half-written file is never seen
    return changed


def main():
    classes = load_classes()
    parser = argparse.ArgumentParser(
        description="Change all objects of one class to another class in the truth.json files.",
        epilog="Classes: " + ", ".join(f"{number} = {name}" for number, name in enumerate(classes)))
    parser.add_argument("folders", nargs="+", help="folder(s) to change; wildcards like \"data\\out\\test*\" work")
    parser.add_argument("--r", "--recursive", dest="recursive", action="store_true",
                        help="also change the truth.json files in all folders below the given ones")
    parser.add_argument("--i", "--input", dest="source", required=True, metavar="CLASS",
                        help="the class to change from (number or name)")
    parser.add_argument("--o", "--output", dest="target", required=True, metavar="CLASS",
                        help="the class to change to (number or name)")
    parser.add_argument("--dry-run", action="store_true", help="only report what would change, write nothing")
    parser.add_argument("--verbose", "-v", action="store_true", help="list every file that is changed")
    args = parser.parse_args(fix_quoted_args(sys.argv[1:]))

    source = parse_class(args.source, classes, "--i")
    target = parse_class(args.target, classes, "--o")
    if source == target:
        sys.exit(f"--i and --o are both class {source} ({classes[source]}): nothing to change")

    folders = []
    for arg in args.folders:
        arg = arg.strip().strip('"')         # a path like "data\out\" reaches us with a stray quote at the end
        wildcard = any(c in arg for c in "*?[") and not Path(arg).exists()
        matches = [Path(p) for p in sorted(glob.glob(arg))] if wildcard else [Path(arg)]
        if not matches:
            print(f"Warning: nothing matches {arg}")
        for m in matches:
            if m.is_dir():
                folders.append(m)
            else:
                print(f"Warning: {m} is not a folder, skipped")
    files = find_truth_files(folders, args.recursive)
    if not files:
        sys.exit("No truth.json found" + ("" if args.recursive else " (use --r to look in the folders below too)"))

    print(f"Class {source} ({classes[source]}) -> class {target} ({classes[target]}) in {len(files)} truth.json file(s)"
          + ("  [dry run: nothing is written]" if args.dry_run else ""))
    files_changed = objects_changed = unreadable = 0
    for n, path in enumerate(files, 1):
        try:
            count = change_file(path, source, target, classes, args.dry_run)
        except (OSError, ValueError) as e:
            unreadable += 1
            print(f"Warning: could not change {path}: {e}")
            continue
        if count:
            files_changed += 1
            objects_changed += count
            if args.verbose:
                print(f"  {path}: {count} object(s)")
        if n % 500 == 0:
            print(f"  ...{n}/{len(files)} files", flush=True)
    print(f"{'Would change' if args.dry_run else 'Changed'} {objects_changed} object(s) in {files_changed} of "
          f"{len(files)} file(s)." + (f" {unreadable} file(s) could not be read." if unreadable else ""))
    sys.exit(1 if unreadable else 0)


if __name__ == "__main__":
    main()
