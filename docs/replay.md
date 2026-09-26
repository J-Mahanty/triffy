# Replay mode

Replay makes London mode use **recorded** camera readings from a chosen
moment instead of live ones. Use it when no collector is running: a demo on a
laptop without a GPU, a room without reliable internet, or any time you want
the same result on every run.

## Running it

Pick a moment covered by `data/live_observations.jsonl`, in **London time**.
At the time of writing, the recording covers Saturday 19 September
(11:11 to 20:38) and Sunday 20 September (06:58 to 18:56); later recordings
extend this. Saturday's early evening is a good busy moment:

```
python -m route_engine.api --replay "2026-09-19 17:30"
```

or, for the terminal chat:

```
python -m route_engine.cli --live --replay "2026-09-19 17:30"
```

You can also set the environment variable `TRIFFY_REPLAY` to the same value.
Kolkata mode is unaffected; it is simulated either way.

## What happens

- The engine's clock starts at the replay moment and runs forward in real
  time, so the recording plays like a film: readings "arrive" every few
  minutes, as they did on the day.
- Only readings recorded up to that clock are used. The rest of the recording
  has not happened yet.
- Everything else (routing, forecasts, route colours) works exactly as live.
- The API reports a `replay` field (`from`, `now`, `now_s`) so the interface
  can label the result.

## Presenting it honestly

Replay data is real: every reading was measured by the vehicle-detection
pipeline on real Transport for London cameras. What is not live is the timing.
Label it as recorded, for example "Replaying recorded traffic from Saturday
19 September, 17:30 London time", and never as live.
