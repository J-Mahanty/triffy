# Starting Triffy

## Double-click

| System | Double-click |
|---|---|
| Windows | `start.bat` |
| Mac | `start.command` (the first time, right-click it and choose **Open**) |

It asks one question, how many live London traffic cameras to watch, starts
everything, and opens the website in your browser. Close the window (or press
Ctrl+C) to stop it.

The same from a terminal, in the `triffy` folder:

```
python start.py                  asks how many cameras
python start.py --cameras 30     no questions
python start.py --cameras 0      no cameras: replay recorded traffic
```

It needs the packages from `requirements.txt` (see the README). Live cameras
also need `requirements-vision.txt`; without them it replays recorded traffic.

## How many cameras?

Without a GPU, reading one camera (download its latest clip, count and track the
vehicles in it) costs about **34 CPU-seconds**, and each camera is read every 5
minutes. So the CPU cores kept busy are about `cameras × 34 ÷ 300`:

| Cameras | Cores busy | Good for |
|---|---|---|
| 0 | none | any computer: replays Saturday 19 September, 17:30 |
| **30** | **~3.4** | **recommended**: a normal 4-core-or-more laptop |
| 80 | ~9 | a large server |
| 172 | ~19.5 | a GPU |

The launcher refuses a number that would keep more than three-quarters of this
computer's cores busy, because the readings would then fall behind.

## Why 30, and why these 30

Spread evenly across London, a few cameras mostly watch roads that no trip we
plan ever uses. So the collector's `--near-routes` option picks the cameras
closest to the routes of the demo trips (`DEMO_TRIPS` in
`route_engine/coverage.py`), alternatives included. The 30 it picks all sit
within about 10 m of one of those routes.

Replaying one recorded moment with only some cameras reporting, 30 cameras
chosen this way kept journey times within about 7% (median) of the answer from
all 172. Fewer is not simply cheaper: with 5, a single camera stuck at a jam
pulls whole routes off course, and the answers were worse than with no cameras
at all.

The first readings arrive a few minutes after starting. Until then, and for any
road without a camera, Triffy uses the usual traffic for that time of day.
Progress is written to `data/collector.log`.
