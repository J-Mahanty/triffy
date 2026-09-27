# Triffy

**Routes that tell you how sure they are.** Triffy plans a drive and predicts
not just how long it will take, but how *reliable* that time is (the best and
worst of 1-in-10 trips), using live traffic measured from real traffic cameras.

Team Veridien G1T2 · theme CIT-01, *Analysis of traffic flow and best route prediction*.

- **Try it:** https://triffy.vargoseus.com
- **Chat to it on Discord:** [join the server](https://discord.gg/rEpvnmGtRV); the bot will message you.
- **Run it on your own computer:** [below](#run-it-on-your-computer), about 5 minutes, no GPU needed.

## What it does

- **Two cities.** *Kolkata:* real roads with simulated traffic (its camera
  feeds are not public), running on Kolkata's real clock. *London:* real roads
  and real traffic, measured by our own computer vision on Transport for
  London's traffic cameras.
- **Risk-aware routes.** Each route comes with a typical time, a range, and how
  predictable it is, and alternatives are offered only when they are worth it
  (quicker but less certain, more predictable, shorter...).
- **Traffic along the route.** The chosen route is coloured by the traffic
  expected *when you reach each part of it*, not the traffic there now.
- **Three ways in:** a website (laptop and phone), a Discord bot, and a
  terminal chat.

## Run it on your computer

You need **Python 3.10 to 3.13** ([python.org](https://www.python.org/downloads/);
on Windows, tick *"Add python.exe to PATH"* while installing) and an internet
connection for the map and the first setup. No GPU is needed.

### 1. Get the code

With git:

```
git clone https://github.com/J-Mahanty/triffy.git
cd triffy
```

Or without git: on this page, **Code → Download ZIP**, unzip it, and open a
terminal in the unzipped `triffy` folder.

### 2. Install (once)

This makes a private Python environment in a `.venv` folder, so nothing is
installed system-wide. Takes a few minutes.

**Windows** (PowerShell or Command Prompt):

```
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Mac or Linux** (Terminal):

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Start it

**Double-click** `start.bat` (Windows) or `start.command` (Mac; the first time,
right-click it and choose **Open**). Or from the terminal:

```
.venv\Scripts\python start.py      (Windows)
.venv/bin/python start.py          (Mac or Linux)
```

It asks one question, **how many live London cameras to watch**:

- **0** needs nothing else: London replays real traffic recorded on Saturday
  19 September at 17:30. **Choose this unless you have set up the live
  cameras** (below).
- **30** (recommended for live) watches the cameras along the demo routes; see
  [Live London](#live-london).

Your browser then opens the site (http://127.0.0.1:8000, or the next free
port). Close the window, or press Ctrl+C, to stop.

### 4. Use it

Pick **Kolkata** or **London** at the top, type where from and where to (for
example *Park Circus → BBD Bagh*, or *King's Cross → Tower Bridge*), and press
**Find routes**. On a phone the plan is a sheet you can swipe up and down.

## Live London

Live London needs a **collector**: a program that downloads each camera's
latest clip every 5 minutes and counts and tracks the vehicles in it with
YOLO11. It needs the vision packages (about 2 GB, once):

```
.venv\Scripts\python -m pip install -r requirements-vision.txt      (Windows)
.venv/bin/python -m pip install -r requirements-vision.txt          (Mac or Linux)
```

Then start Triffy and answer **30** (or any number up to 172). Without a GPU
each camera costs about 34 CPU-seconds per reading, so 30 cameras keep about
3.4 cores busy; the launcher refuses a number your computer cannot keep up
with. The first readings arrive after a few minutes. Details, and why 30:
[docs/starting.md](docs/starting.md).

*Intel Macs:* use Python 3.12 for the vision packages (PyTorch has no Intel-Mac
build for 3.13).

To replay any recorded moment instead: [docs/replay.md](docs/replay.md).

## Other ways to use it

**Terminal chat**, with no browser:

```
.venv/bin/python -m route_engine.cli "Park Circus to BBD Bagh"
.venv/bin/python -m route_engine.cli --clock 09:00 "Sealdah to Esplanade"
.venv/bin/python -m route_engine.cli --live --replay "2026-09-19 17:30" "Victoria to Waterloo"
```

Leave out the message for an interactive chat (on Windows, use
`.venv\Scripts\python`).

**Discord bot**: create a `.env` file in the `triffy` folder with your bot's
token and the channel it should announce itself in, then run it:

```
SERVER_BOT_TOKEN=your-bot-token
CHANNEL_ID=123456789012345678
```

```
.venv/bin/python discordBot/main.py
```

Type `!route` in Discord and answer its two questions. Never commit `.env`
(it is in `.gitignore`).

**On your phone**, on the same Wi-Fi as the computer running Triffy: start the
website open to the network, then open the address it prints on the phone.

```
set TRIFFY_HOST=0.0.0.0                  (Windows Command Prompt)
$env:TRIFFY_HOST = "0.0.0.0"             (Windows PowerShell)
export TRIFFY_HOST=0.0.0.0               (Mac or Linux)
python -m route_engine.api
```

Anyone on that network can then reach it; add `TRIFFY_READONLY=1` so they
cannot change the demo's settings.

## Settings

All optional, set as environment variables:

| Setting | Default | What it does |
|---|---|---|
| `TRIFFY_PORT` | `8000` | Port the website listens on |
| `TRIFFY_HOST` | local only | `0.0.0.0` to open the website to your network or a server |
| `TRIFFY_READONLY` | off | `1`: visitors cannot change the clock, incidents or feedback (use for any public server) |
| `TRIFFY_SIM_CLOCK` | real time | e.g. `18:30`: pin Kolkata's clock, to show the evening rush at any hour |
| `TRIFFY_REPLAY` | live | e.g. `2026-09-19 17:30`: replay recorded London traffic from that moment (London time) |
| `TRIFFY_MAP_BASE` | `http://127.0.0.1:8000` | The website address the chat's map links point to |
| `TRIFFY_DEVICE` | `cpu` | Launcher only: `cuda` to run the collector's vision on a GPU |

## Run it on a server (Docker)

`docker-compose.yml` runs the website, the Discord bot and Redis:

```
docker compose up -d --build
```

Put the secrets and settings in a `.env` file next to it (`SERVER_BOT_TOKEN`,
`CHANNEL_ID`, `APP_PORT`, and any of the settings above; see
[CONTRIBUTING.md](CONTRIBUTING.md)). The website is served on `APP_PORT`
(default `42069`). For a public site, set `TRIFFY_READONLY=1`. The compose file
does not run a collector, so London shows live data only with `TRIFFY_REPLAY`
set, or with a collector writing to the same `data/` folder.

## Tests

```
.venv/bin/python -m pytest tests -q
```

The vision tests are skipped unless the vision packages are installed.

## If something goes wrong

| Problem | Fix |
|---|---|
| `python` or `py` not found, or the Microsoft Store opens | Install Python from python.org and tick *Add python.exe to PATH*, then open a new terminal |
| `No module named ...` | The install step did not finish, or was run with a different Python; rerun step 2 |
| The page looks unstyled | Open it through the server (http://127.0.0.1:8000), not by double-clicking `index.html` |
| London shows no traffic | No collector is running and no replay is set: start with 0 cameras (replay), or set up [Live London](#live-london) |
| "Nothing could listen on port 8000" | Something else uses it; `start.py` picks the next free port, or set `TRIFFY_PORT=8001` |
| A phone cannot open it | Same Wi-Fi? `TRIFFY_HOST=0.0.0.0` set? Allow Python through the firewall when Windows asks |
| Installing OpenCV builds from source for ages (old Macs) | Update pip (`python -m pip install -U pip`) and rerun step 2; ready-made builds are preferred |

## What is where

| Path | What |
|---|---|
| `route_engine/` | The engine: road network, traffic model, forecasting, routing, chat, the website's API, and the camera collector |
| `web/` | The website (served by `route_engine/api.py`) |
| `discordBot/` | The Discord bot |
| `data/` | Road networks, landmarks, and the recorded camera readings; see [README_ankan.md](README_ankan.md) |
| `docs/` | Starting Triffy and choosing cameras; replaying recorded traffic |
| `tests/` | Automated tests |
| `start.py`, `start.bat`, `start.command` | The launcher |

## Team

| Role | Member |
|---|---|
| Integration, deployment and the Discord bot | [@J-Mahanty](https://github.com/J-Mahanty) |
| Computation and analysis: the route engine | [@A-Martyr](https://github.com/A-Martyr) |
| User interface | [@rupkathadm2007-sys](https://github.com/rupkathadm2007-sys) |
| Data | [@ItzzAnkan](https://github.com/ItzzAnkan) |

## Credits and licence

Triffy is free software under the [GNU GPL v3](LICENSE). Map data ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL).
Traffic camera feeds: Transport for London open data, powered by TfL Open Data.
Vehicle detection: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
