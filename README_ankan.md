# Data

Everything Triffy's route engine reads at runtime lives in this folder. It is
committed on purpose: the road networks take minutes to download, and the
camera readings were recorded over real hours and cannot be collected again.

For coverage statistics and charts of the recorded camera data, run:

```
python scripts/data_summary.py
```

That writes `data/charts/summary.md` (tables), `coverage.svg` (readings per hour
across the whole recording) and `traffic_by_hour.svg` (average vehicles per
reading by hour of day). It uses only the standard library.

## What each file is

| File | What it holds | Where it comes from |
|---|---|---|
| `live_observations.jsonl` | One line per camera reading: vehicle counts and movement from a real London traffic camera | Recorded by `route_engine/collector.py` (see below) |
| `city_kol.json` | Road network for central Kolkata (Esplanade, Park Street, BBD Bagh) | OpenStreetMap, via `route_engine/osm_import.py` |
| `city_lon.json` | Road network for central London (Zone 1) | OpenStreetMap |
| `city_blr.json` | Road network for central Bengaluru (Indiranagar, Domlur, MG Road); not used by default | OpenStreetMap |
| `landmarks_kol.json`, `landmarks_lon.json` | Named places (stations, landmarks) with alternative spellings, including Bengali for Kolkata, so users can type "Howrah Station" instead of a street; matched before street names | Hand-curated for the project |
| `collector_ids_A.txt`, `collector_ids_B.txt` | Which cameras each of the two collectors watches | Chosen for the project |
| `validation.json` | Forecast accuracy measured on the recorded London data | `python -m route_engine.validate` |
| `benchmark.json`, `bench_final.txt` | Route-time accuracy on simulated Kolkata trips | `python -m route_engine.benchmark` |

The active city is Kolkata unless the `TRIFFY_CITY` environment variable says
otherwise (`kol`, `lon` or `blr`).

## How the camera data was collected

Kolkata has no public traffic-camera feed, so the real-camera data comes from
London's **TfL JamCams**: public roadside cameras that publish a short video
clip every few minutes, with no API key needed. The camera list comes from
`https://api.tfl.gov.uk/Place/Type/JamCam`.

For each camera, every 6 minutes, the collector:

1. downloads the camera's latest clip (about 10 seconds of video, typically around 100 KB);
2. takes 40 frames from it, using every 2nd frame;
3. runs the **YOLO11n** object detector on each frame (confidence 0.25, image
   size 640), keeping only road vehicles: bicycle, car, motorcycle, bus, truck;
4. links detections across frames into individual vehicles with Ultralytics'
   default tracker (**TrackTrack** in Ultralytics 8.4.137, the version used for
   this recording);
5. writes one line to `live_observations.jsonl`.

The model file itself (`yolo11n.pt`, 5.6 MB) is not stored in the repo.
Ultralytics downloads it automatically the first time the collector runs,
which needs an internet connection. Route planning never needs it: it only
reads the numbers the model already produced.

A vehicle counts as **moving** if its box travelled more than 1.2% of the frame
width (at least 2 pixels) over the clip. Measuring movement relative to frame
width means no camera calibration is needed.

Two collectors ran side by side, one on a GPU and one on the CPU, each watching
its own list of cameras (`collector_ids_A.txt`, `collector_ids_B.txt`) and
appending to the same file.

## Record format: `live_observations.jsonl`

One JSON object per line. Every record has these 14 fields:

| Field | Meaning |
|---|---|
| `camera_id` | TfL camera id, e.g. `00001.09747` |
| `name` | Camera location, e.g. `Edgware Way / Broadfields Ave` |
| `lat`, `lon` | Camera position |
| `t_wall` | When the reading was taken (Unix time, seconds, UTC) |
| `feed_age_s` | How old TfL's clip already was when it was downloaded, in seconds |
| `frames` | Frames analysed (normally 40) |
| `count_mean` | Average number of vehicles visible per frame |
| `count_max` | Most vehicles visible in any single frame |
| `classes` | Average per frame for each vehicle type, e.g. `{"car": 4.62, "truck": 0.07}` |
| `tracks` | Number of distinct vehicles tracked across the clip |
| `moving_frac` | Share of tracked vehicles that were moving (0 to 1) |
| `width`, `height` | Frame size of the clip in pixels |

Example:

```json
{"camera_id": "00001.09747", "name": "Edgware Way / Broadfields Ave", "lat": 51.6216, "lon": -0.27384,
 "t_wall": 1789813644.7, "feed_age_s": 206.5, "frames": 40, "count_mean": 4.7, "count_max": 14,
 "classes": {"car": 4.62, "truck": 0.07}, "tracks": 5, "moving_frac": 1.0, "width": 352, "height": 288}
```

## How the readings feed the route engine

The engine (`route_engine/live_engine.py`) turns each reading into a
congestion level for the road the camera watches:

1. **Each camera is its own yardstick.** From a camera's history (at least 6
   readings), its 20th-percentile vehicle count counts as "quiet here" and its
   85th percentile as "busy here". This way a busy junction and a quiet side
   street are both judged against their own normal traffic.
2. **Count and movement are combined.** When at least 2 vehicles were tracked
   and at least half a vehicle was visible per frame on average, congestion = 0.45 x (how busy, from step 1) + 0.55 x (share of vehicles
   stopped). Movement weighs more because a jam is many vehicles, few moving.
   With too few vehicles to judge movement, only the count is used; with no
   usable information at all, the reading is skipped rather than guessed.
3. **Old readings are ignored.** Readings older than 25 minutes don't count as
   current traffic.
4. **Congestion becomes speed:** speed = free-flow speed x (1 - 0.8 x congestion),
   and the engine spreads each camera's information along the road network to
   roads without cameras.

## Known limitations

- **Gaps in collection.** Collection only happens while a collector runs; the
  coverage chart shows exactly when data exists. Earlier gaps came from the
  collecting laptop sleeping and from collectors being stopped.
- **Weekend-heavy history.** Readings before 25 September are from a Saturday
  and a Sunday only. Weekday readings start on 25 September.
- **A few unreadable lines.** A reading being written at the moment a
  collector stops can be cut short. Both the engine and `data_summary.py` skip
  such lines; the summary reports how many.
- **London only.** The real-camera data is London. Kolkata routes use a traffic
  simulation because there is no open camera feed for Kolkata.

## Sources and attribution

- Camera data: Powered by TfL Open Data.
- Road networks: © OpenStreetMap contributors, available under the Open Database
