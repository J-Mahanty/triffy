# How Triffy works

Every diagram here reads on two levels. The **bold line** in each box says what
happens in plain words; the small text underneath is the technical detail, for
anyone who wants to check it against the code. GitHub draws the diagrams from
the text below (Mermaid), and every number in them is the one the code uses.

| # | Diagram | In one line |
|---|---|---|
| 0 | [The 30-second version](#0-the-30-second-version) | the whole idea in five steps |
| 1 | [The system](#1-the-system) | the parts, and what flows between them |
| 2 | [From a camera to a route](#2-from-a-camera-to-a-route) | how a video clip becomes a prediction |
| 3 | [Inside a collector](#3-inside-a-collector) | how cameras are read, over and over |
| 4 | [Inside the router](#4-inside-the-router) | how the best routes are found |
| 5 | [From a name to a place](#5-from-a-name-to-a-place) | how "Parkk Sirkus" becomes Park Circus |
| 6 | [One request, end to end](#6-one-request-end-to-end) | what happens when you press Find routes |
| 7 | [Two cities, one engine](#7-two-cities-one-engine) | what is real and what is simulated |
| 8 | [Deployment](#8-deployment) | how it runs on the server, and how that is tested |
| 9 | [Where the code lives](#9-where-the-code-lives) | which file does what |

Colours used throughout: **green** open data, **blue** sensing, **grey** stored
data, **orange** prediction, **purple** what people use, **yellow** a decision,
**red** something set aside.

---

## 0. The 30-second version

**In plain words:** Triffy watches London's public traffic cameras, counts the
cars itself, works out the traffic on every road, predicts how it will change,
and suggests routes with an honest "how sure are we".

```mermaid
flowchart LR
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef ui fill:#ece2fb,stroke:#6a3fb5,color:#1f1233

    A["📷 <b>Cameras watch the roads</b><br/><small>172 public TfL cameras in central London</small>"]:::sense
    B["🤖 <b>AI counts the cars</b><br/><small>YOLO11 finds and follows each vehicle</small>"]:::sense
    C["🗺️ <b>Traffic on every road</b><br/><small>from a few cameras to 45,000 road segments</small>"]:::predict
    D["🔮 <b>Predicts the next hour</b><br/><small>today's jams fade back toward normal</small>"]:::predict
    E["🧭 <b>Suggests routes</b><br/><small>'22 min, worst case 23, 96% predictable'</small>"]:::ui

    A --> B --> C --> D --> E
```

---

## 1. The system

**In plain words:** one part of Triffy watches cameras and writes down what it
sees; another part reads those notes and answers people's questions. They share
a notebook (a data file), so either can restart without breaking the other.

```mermaid
flowchart TB
    classDef src fill:#d9f2e3,stroke:#1e7d4a,color:#10261a
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef store fill:#eceff3,stroke:#6b7785,color:#1b2128
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef ui fill:#ece2fb,stroke:#6a3fb5,color:#1f1233

    subgraph S["Where the information comes from"]
        TFL["📷 <b>Public traffic cameras</b><br/><small>TfL JamCams: ~890 cameras,<br/>a fresh ~10 s clip every few minutes</small>"]:::src
        OSM["🗺️ <b>A free map of every road</b><br/><small>OpenStreetMap: roads, junctions, names</small>"]:::src
    end

    subgraph C["Watching: runs separately from the website"]
        COL["👀 <b>Two camera readers</b><br/><small>collectors A and B, 172 cameras,<br/>every 6 minutes</small>"]:::sense
        VIS["🤖 <b>AI counts and follows vehicles</b><br/><small>YOLO11 + tracker: count,<br/>and how many are moving</small>"]:::sense
        COL --> VIS
    end

    subgraph D["What is written down"]
        OBS[("📝 <b>The readings notebook</b><br/><small>data/live_observations.jsonl,<br/>one line per camera per round</small>")]:::store
        GRAPH[("🗺️ <b>Road maps of both cities</b><br/><small>data/city_lon.json, city_kol.json</small>")]:::store
    end

    subgraph E["Thinking: route_engine/"]
        LIVE["🏙️ <b>London: real traffic</b><br/><small>LiveEngine, from the readings</small>"]:::predict
        SIM["🏙️ <b>Kolkata: simulated traffic</b><br/><small>TriffyEngine, real road map</small>"]:::predict
        CORE["🧠 <b>Estimate, predict, plan</b><br/><small>nowcast, forecast, risk-aware router,<br/>personal profiles</small>"]:::predict
        LIVE --> CORE
        SIM --> CORE
    end

    subgraph U["What people use"]
        API["📱 <b>Website, laptop and phone</b><br/><small>route_engine/api.py + web/</small>"]:::ui
        CHAT["💬 <b>Chat that understands 'A to B'</b><br/><small>route_engine/chat.py</small>"]:::ui
        BOT["💬 <b>Discord bot</b>"]:::ui
        CLI["⌨️ <b>Terminal chat</b>"]:::ui
        CHAT --> BOT
        CHAT --> CLI
    end

    TFL -- "download the latest clip" --> COL
    VIS -- "write one line per camera" --> OBS
    OSM -- "imported once" --> GRAPH
    OBS -- "each camera's latest reading,<br/>if under 25 min old" --> LIVE
    GRAPH --> LIVE
    GRAPH --> SIM
    CORE -- "routes with times,<br/>ranges and colours" --> API
    CORE --> CHAT
```

---

## 2. From a camera to a route

**In plain words:** a short video becomes a count of cars; the count becomes
"how jammed is this spot"; that spreads to nearby roads; then Triffy predicts
ahead and plans. The yellow diamonds are checks that stop bad or old data
from fooling it.

```mermaid
flowchart TD
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef decide fill:#fff6c9,stroke:#a08400,color:#2a2300
    classDef out fill:#ece2fb,stroke:#6a3fb5,color:#1f1233
    classDef drop fill:#f6d6d6,stroke:#a33,color:#300

    R["📝 <b>A camera's latest reading</b><br/><small>vehicle count, share moving, tracks, time</small>"]:::sense
    R --> FRESH{"⏱️ <b>Recent enough?</b><br/><small>under 25 minutes old</small>"}:::decide
    FRESH -- "no" --> STALE["🗑️ <b>Too old: ignored</b><br/><small>it no longer describes now</small>"]:::drop
    FRESH -- "yes" --> TRACKS{"🚗 <b>Enough cars seen<br/>to judge the flow?</b><br/><small>at least 2 tracked vehicles</small>"}:::decide

    TRACKS -- "yes" --> BOTH["🚦 <b>How jammed is it here?</b><br/><small>0.45 x how busy + 0.55 x how stopped</small>"]:::predict
    TRACKS -- "no: an empty road and a jam<br/>both show 0% moving" --> ONLY{"📚 <b>Do we know this<br/>camera's normal?</b><br/><small>6+ past readings</small>"}:::decide
    ONLY -- "yes" --> COUNT["🚦 <b>Judge from the count alone</b>"]:::predict
    ONLY -- "no" --> ABSTAIN["🤷 <b>Say 'don't know'</b><br/><small>use the usual traffic instead</small>"]:::drop

    Q["📏 <b>Each camera is its own ruler</b><br/><small>its 20th percentile of past counts = quiet here,<br/>its 85th = busy here</small>"]:::predict
    Q -.-> BOTH
    Q -.-> COUNT

    BOTH --> NOW
    COUNT --> NOW
    NOW["🗺️ <b>Fill in roads with no camera</b><br/><small>nowcast: spread each camera's 'slower than usual'<br/>along its own road, 1.4 km scale, 4.2 km max;<br/>turning off counts as +550 m;<br/>roads feeding a jam weigh about twice as much</small>"]:::predict
    NOW --> CONF["🎯 <b>Every road: a speed and a confidence</b><br/><small>far from any camera: the usual traffic, low confidence;<br/>speed = free-flow x (1 - 0.8 x jam level)</small>"]:::predict
    CONF --> FC["🔮 <b>Predict each road for any future minute</b><br/><small>usual(t) x [1 + (today's anomaly - 1) x e^(-dt / 20 min)];<br/>uncertainty 6% + 5.5% per 10 min ahead</small>"]:::predict
    FC --> ROUTE["🧭 <b>Plan the routes</b><br/><small>time-dependent, risk-aware A*, see diagram 4</small>"]:::predict
    ROUTE --> ANS["✅ <b>Three routes, each saying how sure it is</b><br/><small>typical time, best and worst of 1-in-10, % predictable,<br/>honest label, colours for traffic when you get there</small>"]:::out
```

---

## 3. Inside a collector

**In plain words:** a camera reader goes round its list of cameras again and
again: download the latest clip, let the AI count and follow the cars, write
one line, move to the next. Two readers split the cameras between them.

```mermaid
flowchart TD
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef store fill:#eceff3,stroke:#6b7785,color:#1b2128
    classDef decide fill:#fff6c9,stroke:#a08400,color:#2a2300
    classDef drop fill:#f6d6d6,stroke:#a33,color:#300

    START(["▶️ <b>Start the reader</b><br/><small>python -m route_engine.collector<br/>--ids data/collector_ids_A.txt --cameras 92</small>"]) --> REG
    REG["📋 <b>Get TfL's list of cameras</b><br/><small>cached for a day, written atomically</small>"]:::sense --> PICK
    PICK["✂️ <b>Keep only this reader's cameras</b><br/><small>A: 92, B: 80, no overlap</small>"]:::sense --> YLOAD
    YLOAD["🤖 <b>Load the AI once</b><br/><small>YOLO11n, on the CPU or a GPU</small>"]:::sense --> CYCLE

    CYCLE(["🔁 <b>Start a round</b>"]) --> NEXT{"📷 <b>Another camera<br/>this round?</b>"}:::decide
    NEXT -- "yes" --> DL["⬇️ <b>Download its latest clip</b><br/><small>about 10 s of 352 x 288 video; note its age</small>"]:::sense
    DL --> OK{"🎞️ <b>Got a usable clip?</b>"}:::decide
    OK -- "no" --> SKIP["⏭️ <b>Skip it: 'no feed'</b>"]:::drop
    SKIP --> NEXT
    OK -- "yes" --> DET["🔍 <b>Find the vehicles</b><br/><small>YOLO11 on 40 frames, every 2nd frame;<br/>bicycle, car, motorcycle, bus, truck</small>"]:::sense
    DET --> TRK["🏷️ <b>Follow each one across frames</b><br/><small>tracker gives every vehicle an ID</small>"]:::sense
    TRK --> MEAS["📐 <b>Measure the scene</b><br/><small>average count, share of vehicles that moved,<br/>number of tracks</small>"]:::sense
    MEAS --> WRITE[("📝 <b>Write one line, straight away</b><br/><small>append + flush, so two readers can share the file</small>")]:::store
    WRITE --> NEXT

    NEXT -- "no" --> WAIT["😴 <b>Rest until the next round</b><br/><small>rounds every 6 minutes</small>"]
    WAIT --> CYCLE
```

---

## 4. Inside the router

**In plain words:** like a sat-nav, it keeps a list of "most promising ways so
far" and extends the best one until it reaches you; but it prices every road at
the time you will actually drive it, adds a cost for being unsure, then looks
again for genuinely different alternatives.

```mermaid
flowchart TD
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef decide fill:#fff6c9,stroke:#a08400,color:#2a2300
    classDef out fill:#ece2fb,stroke:#6a3fb5,color:#1f1233
    classDef drop fill:#f6d6d6,stroke:#a33,color:#300

    IN(["🧭 <b>From, to, when, and how much<br/>you hate being late</b><br/><small>origin, destination, departure, lambda</small>"]) --> PQ
    PQ["📋 <b>Keep a list of partial routes,<br/>most promising first</b><br/><small>A*: cost so far + a guess of the rest;<br/>guess = straight-line distance at top speed,<br/>never too high, so the best route is found</small>"]:::predict
    PQ --> POP["👉 <b>Take the most promising</b>"]:::predict
    POP --> DONE{"🏁 <b>Reached the destination?</b>"}:::decide
    DONE -- "yes" --> PATH["🧵 <b>That's a route</b><br/><small>walk back through each step's parent</small>"]:::predict
    DONE -- "no" --> EXP["↪️ <b>Try every road leaving here</b><br/><small>except straight back the way it came</small>"]:::predict
    EXP --> COST["💰 <b>Price that road for when you'd get there</b><br/><small>forecast time at your arrival x your road preferences<br/>x diversity penalty + turn: 4 s, sharp 9 s,<br/>right across traffic x1.8, U-turn +25 s<br/>+ lambda x its uncertainty</small>"]:::predict
    COST --> BETTER{"📉 <b>Cheaper than any way<br/>found there before?</b>"}:::decide
    BETTER -- "yes" --> PUSH["➕ <b>Remember it and add it to the list</b><br/><small>record parent, time, variance</small>"]:::predict
    BETTER -- "no" --> POP
    PUSH --> POP

    PATH --> ALT{"🔁 <b>Mostly the same roads as a<br/>route we already have?</b><br/><small>over 72% overlap</small>"}:::decide
    ALT -- "yes" --> REJ["🗑️ <b>Drop it</b>"]:::drop
    ALT -- "no" --> KEEP["✅ <b>Keep it</b><br/><small>re-priced with no penalty</small>"]:::predict
    REJ --> PEN
    KEEP --> PEN["💸 <b>Make its roads pricier and search again</b><br/><small>x1.7, until 3 routes</small>"]:::predict
    PEN -.-> PQ

    KEEP --> RANGE["📊 <b>Work out the range</b><br/><small>late is possible, very early is not: log-normal;<br/>spread x1.28 as nearby roads jam together;<br/>median, best and worst of 1-in-10,<br/>% predictable = median / worst</small>"]:::predict
    RANGE --> LABEL["🏷️ <b>Label it, only if true</b><br/><small>Recommended, More predictable, Quicker less certain,<br/>Shorter slower, Wider roads</small>"]:::out
    LABEL --> COLOUR["🎨 <b>Colour each stretch</b><br/><small>by traffic expected when you get there:<br/>flowing, slowing, congested, near gridlock</small>"]:::out
    COLOUR --> STEPS["🗒️ <b>Write the directions</b><br/><small>tiny hops and repeated roads merged,<br/>'to stay on' when a road turns</small>"]:::out
```

---

## 5. From a name to a place

**In plain words:** people type places messily. Triffy tries the strictest
match first and only loosens up if needed, and if nothing fits it says so
instead of guessing.

```mermaid
flowchart TD
    classDef decide fill:#fff6c9,stroke:#a08400,color:#2a2300
    classDef ok fill:#d9f2e3,stroke:#1e7d4a,color:#10261a
    classDef drop fill:#f6d6d6,stroke:#a33,color:#300

    T(["⌨️ <b>What was typed</b><br/><small>e.g. 'parkk sirkus'</small>"]) --> C1{"📍 <b>Map coordinates?</b><br/><small>like 51.50, -0.12</small>"}:::decide
    C1 -- "yes" --> P1["✅ <b>Nearest junction</b>"]:::ok
    C1 -- "no" --> C2{"🎯 <b>Exactly a known place?</b><br/><small>landmark, nickname or street</small>"}:::decide
    C2 -- "yes" --> P2["✅ <b>That place</b><br/><small>exact</small>"]:::ok
    C2 -- "no" --> C3{"🔎 <b>A landmark inside a sentence?</b><br/><small>'near Park Circus'</small>"}:::decide
    C3 -- "yes" --> P3["✅ <b>The landmark</b><br/><small>phrase</small>"]:::ok
    C3 -- "no" --> C4{"✏️ <b>A landmark spelt nearly right?</b><br/><small>similarity at least 0.8</small>"}:::decide
    C4 -- "yes" --> P4["✅ <b>The landmark</b><br/><small>typo</small>"]:::ok
    C4 -- "no" --> C5{"🗣️ <b>A landmark that sounds the same?</b><br/><small>c/k/s, double letters, silent h, spaces;<br/>similarity at least 0.9</small>"}:::decide
    C5 -- "yes" --> P5["✅ <b>The landmark</b><br/><small>'parkk sirkus' = 'park circus'</small>"]:::ok
    C5 -- "no" --> C6{"🛣️ <b>Part of a street name?</b><br/><small>or enough shared words</small>"}:::decide
    C6 -- "yes" --> P6["✅ <b>The street</b><br/><small>partial</small>"]:::ok
    C6 -- "no" --> C7{"✏️ <b>A street spelt nearly right?</b>"}:::decide
    C7 -- "yes" --> P7["✅ <b>The street</b><br/><small>typo</small>"]:::ok
    C7 -- "no" --> NONE["❌ <b>'I could not find X on the map'</b><br/><small>no route is guessed;<br/>the chat also suggests near matches</small>"]:::drop
```

---

## 6. One request, end to end

**In plain words:** you ask for a route; the website asks the engine; the engine
refreshes its picture of the traffic if it is more than 2 minutes old, finds
both places, and plans. If a place can't be found, you are told, never given a
made-up route.

```mermaid
sequenceDiagram
    actor V as 🙂 You
    participant W as 📱 Website
    participant A as 🔌 API (api.py)
    participant L as 🏙️ London engine
    participant F as 📝 Readings
    participant R as 🧭 Router

    V->>W: King's Cross to Tower Bridge, Find routes
    W->>A: POST /api/live/plan
    A->>L: plan(origin, destination)
    alt its traffic picture is over 2 minutes old
        L->>F: latest reading per camera
        F-->>L: readings under 25 minutes old
        L->>L: jam level per camera, fill in all roads, predict ahead
    end
    L->>L: find both places (diagram 5)
    alt a place is not on the map
        L-->>A: "I could not find X on the map"
        A-->>W: 400 with that message
        W-->>V: the message, never an invented route
    else both places found
        L->>R: search three times for different routes (diagram 4)
        R-->>L: routes with times, ranges, labels, colours
        L-->>A: the plan
        A-->>W: JSON
        W-->>V: coloured route on the map, answer in the panel
    end
    Note over W,V: every 30 s the page also refreshes<br/>the clock and the traffic figures
```

---

## 7. Two cities, one engine

**In plain words:** London's traffic is measured by Triffy from real cameras.
Kolkata's cameras aren't public, so its traffic is simulated on its real road
map. After that first step, both cities go through exactly the same code.

```mermaid
flowchart LR
    classDef src fill:#d9f2e3,stroke:#1e7d4a,color:#10261a
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef sim fill:#f1e4d3,stroke:#8a6a3c,color:#2a1d0c
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef out fill:#ece2fb,stroke:#6a3fb5,color:#1f1233

    subgraph LON["🇬🇧 London: measured"]
        LCAM["📷 <b>172 real cameras</b><br/><small>TfL JamCams</small>"]:::src --> LYOLO["🤖 <b>Our AI reads them</b><br/><small>YOLO11, every 6 minutes</small>"]:::sense
        LYOLO --> LCONG["🚦 <b>How jammed each spot is</b><br/><small>per-camera rulers</small>"]:::sense
    end

    subgraph KOL["🇮🇳 Kolkata: simulated"]
        KWORLD["🎲 <b>A simulated city's traffic</b><br/><small>base x time of day x incidents;<br/>morning and evening peaks;<br/>crashes and breakdowns that clear</small>"]:::sim
        KWORLD --> KCAM["📷 <b>400 simulated cameras</b><br/><small>see it with realistic noise</small>"]:::sim
    end

    SHARED["🧠 <b>Same code for both</b><br/><small>fill in, predict, plan, personas,<br/>directions, colours</small>"]:::predict
    LCONG --> SHARED
    KCAM --> SHARED
    SHARED --> ANS["🧭 <b>Routes with honest ranges</b><br/><small>on the real road maps of both cities</small>"]:::out
```

---

## 8. Deployment

**In plain words:** everything runs on one server, in separate boxes
(containers). Visitors reach it through Cloudflare. Before any change goes
live, GitHub builds and tries the same setup on a spare machine.

```mermaid
flowchart LR
    classDef ui fill:#ece2fb,stroke:#6a3fb5,color:#1f1233
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef store fill:#eceff3,stroke:#6b7785,color:#1b2128
    classDef src fill:#d9f2e3,stroke:#1e7d4a,color:#10261a
    classDef check fill:#fff6c9,stroke:#a08400,color:#2a2300

    V["🙂 <b>Visitors</b><br/><small>phone or laptop</small>"]:::ui -- "HTTPS" --> CF["☁️ <b>Cloudflare</b><br/><small>public address</small>"]
    CF -- "private tunnel,<br/>no open port" --> WEB

    subgraph SRV["🖥️ The server: docker compose, 28 cores"]
        WEB["📱 <b>Website</b><br/><small>web: API + site, read-only, port 42069</small>"]:::ui
        CA["👀 <b>Camera reader A</b><br/><small>collector-a: 92 cameras, 10 threads</small>"]:::sense
        CB["👀 <b>Camera reader B</b><br/><small>collector-b: 80 cameras, 10 threads</small>"]:::sense
        BOTC["💬 <b>Discord bot</b><br/><small>bot</small>"]:::ui
        RED["🗄️ <b>Short-term store</b><br/><small>redis</small>"]
        VOL[("📝 <b>Shared notebook</b><br/><small>triffy-data volume: maps + readings</small>")]:::store
        CA -- "write" --> VOL
        CB -- "write" --> VOL
        VOL -- "read" --> WEB
        BOTC --> RED
        WEB --> RED
    end

    TFL["📷 <b>TfL cameras</b>"]:::src --> CA
    TFL --> CB

    subgraph GH["✅ GitHub check, on changes to the engine or containers"]
        G1["🔨 <b>Build both</b>"]:::check --> G2["▶️ <b>Start them</b>"]:::check
        G2 --> G3["⏱️ <b>Wait for fresh readings<br/>to reach the website</b>"]:::check
        G3 --> G4["🧭 <b>Plan a live route;<br/>visitors can't change settings</b>"]:::check
    end
```

The camera readers keep about 16 of the server's 28 cores busy (each camera
costs about 34 CPU-seconds a reading) and run at a quarter of the website's
CPU priority, so pages stay quick.

---

## 9. Where the code lives

**In plain words:** each file has one job. Arrows show which file hands its
results to which; dotted lines are looser links (through a data file, or
settings).

```mermaid
flowchart TB
    classDef src fill:#d9f2e3,stroke:#1e7d4a,color:#10261a
    classDef sense fill:#dbe8ff,stroke:#2f5fb3,color:#0f1f3d
    classDef predict fill:#ffe9d1,stroke:#c26a12,color:#2e1a05
    classDef ui fill:#ece2fb,stroke:#6a3fb5,color:#1f1233
    classDef check fill:#fff6c9,stroke:#a08400,color:#2a2300

    subgraph MAP["🗺️ Roads and places"]
        osm_import["<b>osm_import.py</b><br/><small>turns OpenStreetMap into a road graph</small>"]:::src
        network["<b>network.py</b><br/><small>fast road lookup; names, typos, sound-alikes</small>"]:::src
        config["<b>config.py</b><br/><small>every tunable number</small>"]:::src
    end
    subgraph SEE["👀 Seeing"]
        livecams["<b>livecams.py</b><br/><small>TfL cameras, YOLO11 readings</small>"]:::sense
        collector["<b>collector.py</b><br/><small>the reader's endless loop</small>"]:::sense
        coverage["<b>coverage.py</b><br/><small>cameras nearest the demo routes</small>"]:::sense
        cams["<b>cams.py</b><br/><small>simulated cameras</small>"]:::sense
        vision["<b>vision.py</b><br/><small>detection and tracking toolkit</small>"]:::sense
    end
    subgraph THINK["🧠 Understanding and predicting"]
        live_engine["<b>live_engine.py</b><br/><small>London: rulers, jam levels, replay</small>"]:::predict
        engine["<b>engine.py</b><br/><small>Kolkata: plan, leave-by, map state</small>"]:::predict
        simulator["<b>simulator.py</b><br/><small>Kolkata's simulated traffic</small>"]:::predict
        nowcast["<b>nowcast.py</b><br/><small>fills in roads with no camera</small>"]:::predict
        forecast["<b>forecast.py</b><br/><small>traffic at future minutes</small>"]:::predict
    end
    subgraph DECIDE["🧭 Deciding"]
        router["<b>router.py</b><br/><small>routes, alternatives, ranges, directions</small>"]:::predict
        personalize["<b>personalize.py</b><br/><small>personas, vehicles, feedback</small>"]:::predict
    end
    subgraph TALK["📱 Talking to people"]
        api["<b>api.py</b><br/><small>website server, read-only guard</small>"]:::ui
        chat["<b>chat.py + live_chat.py</b><br/><small>one rule-based language brain</small>"]:::ui
        cli["<b>cli.py</b><br/><small>terminal chat</small>"]:::ui
        webdir["<b>web/</b><br/><small>the map; swipe-up panel on phones</small>"]:::ui
        bot["<b>discordBot/</b><br/><small>Discord bot</small>"]:::ui
        pics["<b>mapimg.py, routemap.py</b><br/><small>route pictures for chat</small>"]:::ui
    end
    subgraph PROVE["✅ Proving it works"]
        validate["<b>validate.py</b><br/><small>predictions vs what cameras saw later</small>"]:::check
        benchmark["<b>benchmark.py</b><br/><small>three route planners, one test world</small>"]:::check
        tests["<b>tests/</b><br/><small>a test for every real bug</small>"]:::check
    end

    osm_import --> network
    config -.-> network
    network --> live_engine
    network --> engine
    vision -.-> livecams
    livecams --> collector
    coverage -.-> collector
    collector -. "readings file" .-> live_engine
    cams --> engine
    simulator --> engine
    live_engine --> nowcast
    engine --> nowcast
    nowcast --> forecast
    forecast --> router
    personalize --> router
    router --> api
    router --> chat
    router --> pics
    api --> webdir
    chat --> bot
    chat --> cli
    live_engine -.-> validate
    engine -.-> benchmark
```

See also: [starting Triffy](starting.md), [replay mode](replay.md), and the
[data guide](../README_ankan.md).
