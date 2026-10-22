"""Split a dataset into training data and test data: random image folders are moved to the test folder, all the
others to the training folder.

    python splitDataset.py "data\\out" "data\\train" "data\\test" 100
    python splitDataset.py "data\\out" "data\\train" "data\\test" 20% --seed 1 --dry-run

A data point is an image folder: a folder that contains an org.png (with its truth.json and everything else in it).
The folders are found anywhere below the source folder, so both a folder of image folders and data\\out with its test
folders work. Each image folder keeps its path relative to the source folder (test1 - 2x4 brick\\image_0001), so image
folders with the same name from different test folders never collide. The count is a number of image folders or a
percentage like 20%. It counts all image folders together; with --per-folder it is applied to every test folder
separately (20% of each one, or that many from each one).

The image folders are moved, not copied (use --copy to keep the source). Nothing is moved if a target folder already
exists. Close the groundTruth window first.
"""
import argparse
import os
import random
import shlex
import shutil
import sys
from pathlib import Path

IMAGE_FILE = "org.png"


def fix_quoted_args(argv):
    """cmd and PowerShell turn  "folder\\"  into  folder" --seed 1  : the backslash escapes the closing quote, so the
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


def clean_path(text):
    return Path(text.strip().strip('"')).resolve()       # a path like "data\out\" reaches us with a stray quote


def is_inside(path, folder):
    return path == folder or folder in path.parents


def find_data_points(source, skip):
    """Every folder below source that contains an org.png, sorted. Folders in skip (the train and test folders) are
    not looked into. Without any org.png the folders directly in source are the data points."""
    found = []
    for root, dirs, files in os.walk(source):
        root = Path(root)
        if IMAGE_FILE in files and root != source:
            found.append(root)
            dirs[:] = []                                 # an image folder is one data point, do not go deeper
        else:
            dirs[:] = sorted(d for d in dirs if not any(is_inside(root / d, s) for s in skip))
    if not found:
        found = sorted(p for p in source.iterdir() if p.is_dir() and not any(is_inside(p, s) for s in skip))
    return sorted(found)


def parse_count(text, total, where=""):
    """'100' or '20%' as a number of data points. Exits with a message if it is neither or too large."""
    text = text.strip()
    try:
        if text.endswith("%"):
            percent = float(text[:-1])
            if not 0 <= percent <= 100:
                raise ValueError
            return round(total * percent / 100)
        count = int(text)
        if count < 0:
            raise ValueError
    except ValueError:
        sys.exit(f"count: '{text}' is not a number of data points (use for example 100 or 20%)")
    if count > total:
        sys.exit(f"count: {count} is more than the {total} data point(s) found{where}")
    return count


def transfer(points, source, target, copy):
    """Move (or copy) every data point to target, keeping its path relative to source. Returns the number that failed."""
    failed = 0
    for n, point in enumerate(points, 1):
        destination = target / point.relative_to(source)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copytree(point, destination)
            else:
                shutil.move(str(point), str(destination))
        except OSError as e:
            failed += 1
            print(f"Warning: could not {'copy' if copy else 'move'} {point}: {e}")
        if n % 500 == 0:
            print(f"  ...{n}/{len(points)} to {target.name}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Split a dataset into training data and test data by moving random image folders.")
    parser.add_argument("source", help="folder with all the data (image folders, each with an org.png)")
    parser.add_argument("train", help="folder the training data goes to")
    parser.add_argument("test", help="folder the test data goes to")
    parser.add_argument("count", help="how many image folders go to the test data: a number or a percentage (20%%)")
    parser.add_argument("--per-folder", action="store_true",
                        help="apply the count to every folder of image folders (every test folder) separately")
    parser.add_argument("--seed", type=int, help="random seed, to get the same split again (default: random, printed)")
    parser.add_argument("--copy", action="store_true", help="copy the image folders instead of moving them")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be moved, write nothing")
    args = parser.parse_args(fix_quoted_args(sys.argv[1:]))

    source, train, test = clean_path(args.source), clean_path(args.train), clean_path(args.test)
    if not source.is_dir():
        sys.exit(f"{source} is not a folder")
    if train == test or is_inside(train, test) or is_inside(test, train):
        sys.exit("The train and test folders must be two separate folders, not one inside the other")
    if is_inside(source, train) or is_inside(source, test):
        sys.exit("The source folder cannot be (inside) the train or test folder")

    points = find_data_points(source, [train, test])
    if not points:
        sys.exit(f"No image folders found in {source}")
    # Without --per-folder all data points form one group; with it every folder that holds image folders is a group.
    groups = {}
    for point in points:
        groups.setdefault(point.parent if args.per_folder else source, []).append(point)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
    rng = random.Random(seed)
    test_points = []
    for folder, group in sorted(groups.items()):
        where = f" in {folder}" if args.per_folder else ""
        test_points += rng.sample(group, parse_count(args.count, len(group), where))
    test_set = set(test_points)
    test_points = sorted(test_points)
    train_points = [p for p in points if p not in test_set]
    test_count = len(test_points)

    clashes = [t / p.relative_to(source) for t, group in ((train, train_points), (test, test_points)) for p in group
               if (t / p.relative_to(source)).exists()]
    if clashes:
        shown = "\n  ".join(str(c) for c in clashes[:5])
        sys.exit(f"{len(clashes)} target folder(s) already exist, nothing was moved. For example:\n  {shown}")

    verb = "copy" if args.copy else "move"
    print(f"{len(points)} image folder(s) in {source}: {len(train_points)} to train, {test_count} to test "
          f"(seed {seed})" + ("  [dry run: nothing is written]" if args.dry_run else ""))
    if args.dry_run:
        print(f"Would {verb} {len(train_points)} folder(s) to {train} and {len(test_points)} to {test}.")
        return
    failed = transfer(train_points, source, train, args.copy) + transfer(test_points, source, test, args.copy)
    print(f"{'Copied' if args.copy else 'Moved'} {len(train_points) + len(test_points) - failed} folder(s): "
          f"{len(train_points)} to {train}, {len(test_points)} to {test}." +(f" {failed} failed." if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
