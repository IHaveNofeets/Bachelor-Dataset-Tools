# Bachelor Dataset Tools

Tools for building the ground truth of an image dataset. They work on folders of `org.png` images and write one `truth.json` per image, holding the labelled objects as polygons.

| File | What it does |
| --- | --- |
| `groundTruth - Tobias.py` | Annotation tool (OpenCV window), with a *covered* mark, a bent/closed split tool and an **Other** class |
| `groundTruth - Claus.py` | Annotation tool (OpenCV window), with a straight-line split, `--redraft-all` and `--default-class` |
| `changeClass.py` | Batch script that changes every object of one class to another class in many `truth.json` files |

The two `groundTruth` files are two variants of the same tool. See [Differences between the two variants](#differences-between-the-two-variants).

## How it works

1. [rembg](https://github.com/danielgatis/rembg) removes the background from each image.
2. The foreground is split into objects (connected components, minimum area 5000 px) and saved as a **draft** `truth.json`. Objects touching the top or bottom edge are marked *uncompleted*.
3. You correct the draft in the window: paint, fill, cut, merge, set classes and flags.
4. Drafts are made in a background process, from the current image to the last one, so the window never freezes.

Anything you save is never overwritten by a redraft. A draft carries a `"draft": {"model": ...}` field that is removed as soon as you edit the file, and only files that still have it are replaced.

## Requirements

- Python 3
- `opencv-python`, `numpy`, `rembg` (plus an `onnxruntime` build for it)
- `tkinter` for the options dialog (included with the standard Windows Python installer)

```
pip install opencv-python numpy "rembg[cpu]"
```

Use `rembg[gpu]` for CUDA. Only the Claus variant explicitly asks for the CUDA provider first (falling back to CPU). rembg downloads its model on first use.

The scripts are written for Windows (`data\out` paths, `CREATE_NO_WINDOW`).

## Data layout

A **test folder** holds one folder per image, each with an `org.png`. The tool writes `truth.json` next to it.

```
data\out\
  test1 - 2x4 brick\
    image_0001\
      org.png          <- input
      truth.json       <- written by the tool
    image_0002\
      ...
```

You can pass a test folder, a parent such as `data\out` (every test folder below it is used) or a wildcard such as `data\out\test*`. A folder that directly contains an `org.png` also counts as a test folder.

## Annotation tool

```
python "groundTruth - Tobias.py" "data\out\test1 - 2x4 brick"
python "groundTruth - Tobias.py" data\out --start 25
python "groundTruth - Tobias.py"                          # no folder: a dialog asks for the options
```

### Options

| Option | Meaning |
| --- | --- |
| `folders` | Test folder(s), a parent folder or wildcards. Without any, a dialog asks. |
| `--start N`, `-s N` | Image number to start at (default 1) |
| `--model NAME` | rembg model, for example `u2net` or `bria-rmbg`. The default is rembg's own default. |
| `--headless` | No window. Make drafts for every image that has no `truth.json`, then exit. |
| `--redraft` | Also replace untouched drafts made by a *different* model. Your own work is never touched. |
| `--dry-run` | With `--redraft`: only report what would change |

Variant-specific options (`--auto-covered`, `--redraft-all`, `--default-class`) are listed below.

Press **O** in the window for the same options as a dialog. Redrafting started from the window runs in a separate process and logs to `groundTruth_redraft.log`.

### Keys and mouse

| Input | Action |
| --- | --- |
| **N** / **B** | Next / previous image (the current one is saved first) |
| **1**-**9** | Set the class of the hovered object, or the class for new objects when over empty space |
| **Space** | Cycle to the next class |
| **F** | Fill mode: left click fills the region under the mouse (colour tolerance, mouse wheel), right click deletes the object |
| **S** | Split mode: cut an object into pieces |
| **P** | Polygon mode: left click adds a point, click the first point or press Enter to close, right click or Backspace removes a point, Esc cancels |
| **C** | Combine: press over one object, then over another to merge them |
| **U** | Toggle *uncompleted* (object is cut off by the image edge) |
| **L** | Give the hovered object a letter. Press a letter or digit, Backspace removes it, Esc cancels. |
| **O** | Open the options dialog (folders, start image, model, redraft) |
| **Q** / **Esc** | Save and quit |

In the default **brush** mode, left drag paints, right drag erases, and the mouse wheel changes the brush size (Claus variant: hold Ctrl). Press **F**, **S** or **P** again to return to the brush.

### `truth.json`

```json
{
  "image": "org.png",
  "width": 1920,
  "height": 1080,
  "objects": [
    {
      "id": 1,
      "class": 0,
      "class_name": "2x4 brick",
      "uncompleted": false,
      "covered": false,
      "letter": "A",
      "bbox": {"x": 10, "y": 20, "w": 300, "h": 150},
      "polygons": [[[10, 20], [310, 20], [310, 170]]]
    }
  ]
}
```

`class` is the position of the class in the `CLASSES` list at the top of the script (counting from 0). `letter` only exists on objects that were given one. `covered` only exists in the Tobias variant. Saving is atomic (a `.tmp` file is renamed), so a crash never leaves a half-written file.

## Changing classes in bulk

`changeClass.py` turns every object of one class into another class across many `truth.json` files.

```
python changeClass.py "data\out\test2 - 2x4 brick" --r --i=0 --o=2
python changeClass.py "data\out" --r --i="2x4 brick" --o="gray rock" --dry-run
```

| Option | Meaning |
| --- | --- |
| `folders` | Folder(s) to change. Wildcards work. |
| `--r`, `--recursive` | Also change every `truth.json` in folders below. Without it only the `truth.json` directly in the folder is changed. |
| `--i CLASS` | Class to change from, as a number or a name (any case) |
| `--o CLASS` | Class to change to |
| `--dry-run` | Only report what would change |
| `-v`, `--verbose` | List every changed file |

Close the annotation window before running it. The window keeps its own copy of the open image and would write it back over the change.

Changed files lose their `draft` marker, so a later `--redraft` leaves them alone.

## Differences between the two variants

| | Tobias | Claus |
| --- | --- | --- |
| Classes | 2x4 brick, Slide, Brown rock, gray rock, 2x2 brick, 2x2 brick with letter, **Other** | the same without *Other* |
| *Covered* mark (**V**; **A** marks every object touching another) | yes, plus `--auto-covered` to mark all images in the background | no |
| Split | straight drag, or a bent or closed path (click points, Enter to cut) | straight line, **Shift** + drag |
| Brush size | mouse wheel | **Ctrl** + mouse wheel |
| `--redraft-all`, `--default-class` | no | yes: overwrite *every* `truth.json` (headless only, use with care) |
| GPU | rembg's default provider | asks for CUDA first, then CPU |

## Customising

Constants are at the top of each script: `CLASSES` (the class names, so adding a class means appending to the list), `PALETTE`, `BRUSH_DEFAULT`, `TOLERANCE_DEFAULT`, `DRAFT_MIN_AREA`, `DRAFT_THRESHOLD` and `REMBG_MODEL`. Append new classes to the end of `CLASSES` so existing `truth.json` files keep their meaning.

## License

Apache License 2.0, see [LICENSE](LICENSE).
