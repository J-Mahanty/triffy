"""Natural-language layer shared by every conversational front-end.

The Discord bot, the Telegram bot and the terminal client all route through
``respond()`` (or ``handle()``, its plain-text form), so there is exactly one
place where language is understood and one place where replies are worded. A
second parser living inside a bot would drift within a week.

Parsing is deliberately rule-based rather than model-backed. For a commuting
assistant the intent space is small and highly patterned ("A to B", "get me to X
by 9:30"), and a deterministic parser has three properties that matter for a
live demo: it cannot hallucinate a destination, it runs in microseconds, and it
behaves identically every time you show it to someone.

The same brain serves simulated Kolkata (``TriffyEngine``) and live London
(``live_chat.LiveChatEngine``, which offers the same surface). The few answers
that genuinely differ - there is no clock to move in London, and its
"incidents" are what the cameras see - branch on ``self.live``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from .config import ACTIVE_CITY, DATA, MAP_BASE
from .engine import TriffyEngine
from .network import _norm
from .simulator import fmt_clock

_LOG = logging.getLogger(__name__)

# Longest message we will parse. Nineteen matchers, most of them regexes, run
# over every message; an arbitrarily long one from a public Telegram bot is
# work someone else chose for us. Real commuter questions are far shorter, so
# this only ever truncates abuse or a paste accident.
MAX_MESSAGE_CHARS = 500

# Where unparsed messages go. This file is the next iteration's to-do list.
MISS_LOG = DATA / "chat_misses.jsonl"

# Example places for the help text, so London users see London examples.
EXAMPLES = {
    "kol": ("Park Circus", "BBD Bagh", "Alipore", "Esplanade", "Sealdah", "Park Street"),
    "lon": ("Waterloo", "Bank", "Victoria", "King's Cross", "Oxford Circus",
            "Trafalgar Square"),
}

HELP = """**Triffy** — your commute, predicted.

**Plan a trip**
• `{a} to {b}`
• `{c} to {d} at 8:15`
• `get me to {e} by 19:30 from {f}`

**Follow up**
• `what about at 9?` · `15 min later` · `way back`

**Decide**
• `should I go now?` — leaving now against waiting, by arrival time
• `which is faster, {c} or {e}?`
• `what time should I leave for {b}?`
• `get me home` · `take me to work`

**Live conditions**
• `traffic` — how the network looks right now
• `traffic on {f}` — one road
• `incidents` — what is disrupting things

**Make it yours**
• `I ride a motorcycle` (or car / auto / taxi)
• `set home {a}` · `set work {b}`
• `commute` — your usual trip, right now
• `I hate being late` / `I want the fastest`
• `good route` / `bad route` — teach me after a trip
{local}
**Demo controls**
• `time 18:45` — move the network clock
• `whoami` — what I know about you
"""

LOCAL_HELP = """
**Your language**
• `Park Circus theke Sealdah` · `Park Circus se Sealdah`
• `পার্ক সার্কাস থেকে শিয়ালদহ` · `पार्क सर्कस से सियालदह`
"""

TIME_RE = r"(\d{1,2}(?::|\.|h)?\d{0,2}\s*(?:am|pm)?)"
COORD_RE = re.compile(r"-?\d+\.\d+, -?\d+\.\d+")

GREETINGS = {"help", "?", "hi", "hello", "hey", "start", "/help", "namaste",
             "namaskar", "nomoskar", "নমস্কার", "नमस्ते"}

# Closing pleasantries. Without these a "thanks" fell through every matcher and
# was answered with "I did not catch a route in that" - the bot telling someone
# who was being polite that they had got it wrong.
#
# Deliberately NOT here: "good", "great", "nice", "bad", "terrible" - those are
# _cmd_feedback's thumbs up/down on the last route, and swallowing them here
# would quietly disable the personalisation the whole profile system is built
# on. Anything added to this set must not appear in that matcher's verdicts.
THANKS = {"thanks", "thank you", "thanks!", "thank you!", "thx", "ty", "tnx",
          "cheers", "got it", "ok", "okay", "kk",
          "dhonnobad", "dhanyabad", "ধন্যবাদ",
          "shukriya", "धन्यवाद"}

# "office kotokkhon lagbe" - Bengali for "how long does the office take". These
# trail the place instead of leading it, which is why the FILLER lead-ins miss
# them; ROUTE_WORDS only covers the two-place "X theke Y" form.
LOCAL_ASK_TAILS = ("kotokkhon lagbe", "koto khon lagbe", "kotokhon lagbe",
                   "কতক্ষণ লাগবে",
                   "kotokkhon", "kotodur", "koto dur", "kitna time lagega",
                   "kitni der")

# "what time should I leave for work?" - a deadline question with no deadline.
LEAVE_TIME_RE = re.compile(
    r"^\s*(?:(?:what time|when|how early)\s+(?:should|shall|do|must)\s+i"
    r"|(?:what(?:'|’)?s|whats)\s+(?:the\s+)?(?:best|right|ideal|safest)"
    r"\s+time\s+(?:for me\s+)?to)\s+"
    r"(?:leave|start|set off|head out|head off|go)\s*"
    r"(?:for|to|towards)?\s*(.*?)\s*\??$", re.I)

# "should I go now?" - the question this whole project is about.
WHEN_TO_GO_RE = re.compile(
    r"^\s*(?:"
    r"should i (?:go|leave|set off|head out|start)"
    r"|is (?:it|now|this) (?:a )?(?:good|bad|better|the right|right) (?:time|idea)"
    r"|is it better to (?:go|leave|wait)"
    r"|(?:is it |would it be )?(?:worth|better) (?:waiting|to wait)"
    r"|will it be (?:faster|quicker|better|worse|easier)"
    r"|(?:go|leave) now or (?:later|wait)"
    r"|now or later"
    r"|am i (?:going to be|gonna be|likely to be) late"
    r"|will i be late"
    r"|do i have time"
    r")\b.*$", re.I)

# How far ahead "or later" looks, in minutes. An hour: past that the forecast
# is mostly the historical profile and the advice stops being live.
WAIT_OFFSETS_MIN = (0, 15, 30, 45, 60)

# "which is faster, Esplanade or Sealdah?" - two destinations, one origin.
COMPARE_WORDS = ("faster", "quicker", "fastest", "quickest", "better", "best",
                 "sooner", "closer", "nearer")
COMPARE_LEAD_RE = re.compile(
    r"^(?:which|what)(?:'|’)?s?\s+(?:is\s+|are\s+|would be\s+)?"
    r"(?:the\s+)?(?:%s)(?:\s+(?:one|option|route|way|to go to))?\s*[,:]?\s*"
    % "|".join(COMPARE_WORDS), re.I)
COMPARE_TAIL_RE = re.compile(
    r"\s*[,:]?\s*(?:which|what)?\s*(?:is|are)?\s*(?:%s)\s*$"
    % "|".join(COMPARE_WORDS), re.I)
MAX_COMPARE = 4

# "I'm at Sealdah, how long home" - the origin stated conversationally.
AT_ORIGIN_RE = re.compile(
    r"^\s*(?:i(?:'|’)?m|im|i am|currently|standing)\s+(?:at|in|near|on)\s+"
    r"(.+?)\s*[,;]\s*(.+)$", re.I)
AT_ORIGIN_ASK_RE = re.compile(
    r"^\s*(?:how (?:long|far)|what(?:'|’)?s the (?:time|eta))\s*"
    r"(?:is it\s+)?(?:to|until|till|for)?\s*", re.I)

# "X theke Y" (Bengali) and "X se Y" (Hindi) both mean "from X to Y", and the
# sentence usually ends in a verb ("jabo", "jana hai") that is not a place.
ROUTE_WORDS = {"theke", "থেকে", "se", "से"}
TRAILING_WORDS = {"jabo", "jaabo", "jaabo?", "jabo?", "যাব", "যাবো", "যাব?", "যাবো?",
                  "jete", "যেতে", "chai", "চাই", "tak", "तक", "jana", "jaana",
                  "जाना", "hai", "है", "hai?", "है?"}

# Follow-ups, matched after trailing spaces and question marks are stripped.
FOLLOW_TIME = re.compile(r"^(?:and |what about |how about |what if |and what if )?"
                         r"(?:i )?(?:if i )?"
                         r"(?:leave |leaving |left |go |going |set off )?at "
                         + TIME_RE + r"$")
FOLLOW_ABOUT_TIME = re.compile(r"^(?:what|how) about " + TIME_RE + r"$")
FOLLOW_LATER = re.compile(r"^(?:and |what if i )?(?:leave |leaving |go )?"
                          r"(?:in )?(\d{1,3}) ?(?:min|mins|minutes)(?: later)?$")
FOLLOW_BACK = {"way back", "the way back", "and back", "and the way back", "return",
               "return trip", "return journey", "reverse", "going back", "back"}
FOLLOW_PLACE = re.compile(r"^(?:what|how) about (to|from) (\S.*)$")
DEFAULT_LATER_MIN = 15


@dataclass
class Reply:
    """A chat answer, and the plan behind it when there is one.

    Text-only front-ends read ``text``. The Telegram bot also reads ``plan`` to
    draw the route on a map and offer buttons for the alternatives.
    """
    text: str
    plan: object = None          # engine.Plan for a route or leave-by answer
    kind: str = "text"           # "text" | "route" | "leave_by"


class ChatBrain:
    def __init__(self, engine: TriffyEngine | None = None,
                 remote: bool = False):
        self.eng = engine or TriffyEngine()
        self.live = bool(getattr(self.eng, "is_live", False))
        self.city = getattr(self.eng, "city", ACTIVE_CITY)
        # True when the person reading the answer is NOT at this machine -
        # Telegram, Discord. It decides whether a map link is worth sending;
        # see map_link.
        self.remote = remote
        # Unparsed messages this process has seen, for a quick in-session
        # read without opening the file.
        self.misses: list = []
        # user_id -> (origin, destination, depart_s) of their last trip, for
        # feedback and follow-ups like "what about at 9?".
        self.last_plan: dict = {}

    # -- entry points ---------------------------------------------------------

    def handle(self, text: str, user_id: str = "guest",
               display_name: str = "Commuter") -> str:
        return self.respond(text, user_id, display_name).text

    def respond(self, text: str, user_id: str = "guest",
                display_name: str = "Commuter") -> Reply:
        t = localise((text or "").strip())
        # Punctuation-only messages ("...", "??", "!") are a stray tap or a
        # thinking-out-loud pause, not a question. Treat them as empty rather
        # than telling the person they failed to name a route.
        if not t or not re.search(r"[^\W_]", t, re.UNICODE):
            return Reply(self.help_text())
        t = t[:MAX_MESSAGE_CHARS]
        low = t.lower()
        self.eng.profiles.get(user_id, display_name)

        try:
            return self._dispatch(t, low, user_id)
        except Exception:
            # Every message must produce a reply. route() catches ValueError,
            # but a RuntimeError from the router or a KeyError from a corrupt
            # spatial index escaped the brain entirely - and on Telegram a
            # handler that raises sends *nothing*, so the person sees their
            # message delivered and then silence, unable to tell whether we are
            # thinking, broken, or ignoring them. Silence is the worst failure
            # a chat interface has. One boundary here is worth more than a
            # try/except in each of nineteen matchers.
            _LOG.exception("chat dispatch failed on %r (user %s)", t[:120], user_id)
            return Reply(
                "Something went wrong at my end working that out - it is not "
                "something you typed. Try again, or ask a simpler question like "
                "`%s to %s`." % EXAMPLES.get(self.city, EXAMPLES["kol"])[:2])

    def _dispatch(self, t: str, low: str, user_id: str) -> Reply:
        """Run the matcher chain. Order is load-bearing; see the notes inline."""
        for matcher in (self._cmd_help, self._cmd_whoami, self._cmd_vehicle,
                        self._cmd_setplace, self._cmd_risk, self._cmd_feedback,
                        # After _cmd_feedback, so a one-word verdict on the last
                        # route is still learned from rather than read as small talk.
                        self._cmd_ack,
                        # Next to _cmd_ack: both are small talk, but an
                        # affirmation needs the last trip for context.
                        self._cmd_affirm,
                        self._cmd_time, self._cmd_incidents, self._cmd_traffic,
                        self._cmd_commute, self._cmd_arrive_by, self._cmd_leave_time,
                        # Before _cmd_route: "is it better to go now or in an
                        # hour" contains " to " and would otherwise be read as
                        # a trip from a place called "is it better".
                        self._cmd_when_to_go,
                        # After _cmd_when_to_go, which owns the "now or later"
                        # reading of the same "X or Y" shape.
                        self._cmd_compare,
                        self._cmd_followup, self._cmd_at_origin,
                        self._cmd_destination_only, self._cmd_route,
                        # Last two, in this order: a message that is only a
                        # place name (anything with structure has already had
                        # its turn, so this cannot steal "A to B"), and then
                        # the limits, which only see what nothing could answer.
                        self._cmd_bare_place, self._cmd_scope):
            out = matcher(t, low, user_id)
            if out is not None:
                return out if isinstance(out, Reply) else Reply(out)

        # Nothing understood it. Keep the message: the list of things we could
        # not parse IS the roadmap for the next iteration, and it is the one
        # artefact that cannot be reconstructed afterwards. Every NLU fix so far
        # came from someone guessing phrasings; real users produce ones nobody
        # imagined, and without this they vanish when the session ends.
        self.record_miss(t, user_id)
        a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]
        return Reply("I did not catch a route in that. Try `%s to %s`, "
                     "or send `help` for everything I can do." % (a, b))

    def record_miss(self, text: str, user_id: str) -> None:
        """Append an unparsed message to the miss log, best effort.

        Never raises: a bot that crashes because it could not write a log file
        is worse than one that loses a log line. Only the message and a hashed
        user id are stored - enough to spot repeat askers without keeping who
        they are.
        """
        self.misses.append(text)
        try:
            import hashlib
            import json
            with MISS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "text": text,
                    "user": hashlib.sha256(str(user_id).encode()).hexdigest()[:12],
                    "city": self.city,
                    "live": self.live,
                }) + "\n")
        except Exception:                      # pragma: no cover - disk issues
            _LOG.debug("could not append to the miss log", exc_info=True)

    def help_text(self) -> str:
        a, b, c, d, e, f = EXAMPLES.get(self.city, EXAMPLES["kol"])
        return HELP.format(a=a, b=b, c=c, d=d, e=e, f=f,
                           local=LOCAL_HELP if self.city == "kol" else "")

    # -- simple commands ----------------------------------------------------

    def _cmd_help(self, t, low, uid):
        if low in GREETINGS:
            return self.help_text()
        return None

    def _cmd_ack(self, t, low, uid):
        """"thanks" is not a routing request, and should not be answered as a
        failed one. The old reply was "I did not catch a route in that", which
        made the polite end of a working conversation look like a crash."""
        if low.strip().rstrip("!. ") not in THANKS:
            return None
        return ("Any time. Ask me again whenever you are about to set off — "
                "the answer changes with the traffic.")

    def _cmd_affirm(self, t, low, uid):
        """A bare "yes" when nothing was asked.

        The brain has no notion of a pending question, so an affirmation is
        genuinely ambiguous. Rather than invent a confirmation state machine,
        read it the way the context suggests: with a previous trip, "yes" almost
        certainly means "yes, that one", so offer the obvious next step. With no
        context, say plainly that nothing was asked - the old reply told the
        person they had failed to name a route, which is a strange answer to
        agreement.
        """
        if low.strip().rstrip("!. ") not in AFFIRM:
            return None
        last = self.last_plan.get(uid)
        if last:
            origin, dest, _depart = last
            return ("I had not asked anything, so I am not sure what to say yes "
                    "to. If you want %s to %s again with the latest traffic, "
                    "just say `again`." % (_display(origin), _display(dest)))
        a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]
        return ("I had not asked anything — tell me where you are going and I "
                "will work it out. For example `%s to %s`." % (a, b))

    def _cmd_whoami(self, t, low, uid):
        # SELF_KNOWLEDGE is the same question asked in words. "what do you know
        # about me" was answered with "I did not catch a route in that", which
        # is a strange reply to a question about the profile we are keeping on
        # them - and the honest answer is a demo asset, not a liability.
        q = low.strip().rstrip("?!. ")
        if q not in ("whoami", "me", "profile", "/profile") and q not in SELF_KNOWLEDGE:
            return None
        p = self.eng.profiles.get(uid)
        if p.risk_aversion > 1.2:
            style = "avoids risk, prefers predictable routes"
        elif p.risk_aversion > 0.5:
            style = "balanced"
        else:
            style = "chases the fastest option"
        home, work = self._home_work(p)
        lines = ["**%s**" % p.name,
                 "Vehicle: %s" % p.vehicle,
                 "Style: %s" % style,
                 "Risk setting (lambda): %.2f" % p.risk_aversion,
                 "Home: %s" % (home or "not set"),
                 "Work: %s" % (work or "not set"),
                 "Trips learned from: %d" % p.trips_logged]
        if p.road_bias:
            top = sorted(p.road_bias.items(), key=lambda kv: -abs(kv[1] - 1.0))[:4]
            lines.append("Learned road preferences: " +
                         ", ".join("%s %s" % (k, "avoid" if v > 1 else "prefer")
                                   for k, v in top))
        return "\n".join(lines)

    def _cmd_vehicle(self, t, low, uid):
        m = re.search(r"\b(?:i (?:ride|drive|use|am on|take)|vehicle|switch to)\b.*?"
                      r"\b(motorcycle|motorbike|bike|scooter|car|auto|rickshaw|taxi|cab)\b",
                      low)
        if not m:
            return None
        raw = m.group(1)
        veh = _VEHICLE_WORDS.get(raw, "car")
        self.eng.profiles.set_field(uid, "vehicle", veh)
        extra = ""
        if veh == "motorcycle":
            extra = ("\nI will now assume you can filter through stopped traffic, "
                     "so your ETAs in heavy congestion will drop noticeably.")
        elif veh == "car":
            extra = "\nI will keep you off the narrowest lanes where I can."
        return "Got it — you are on a **%s**.%s" % (veh, extra)

    def _cmd_setplace(self, t, low, uid):
        m = re.match(r"^set +(home|work) +(?:to +)?(\S.*)$", t.strip(), re.I)
        if not m:
            return None
        field, value = m.group(1).lower(), m.group(2).strip()
        place = self.eng.resolve(value)
        if place is None:
            return self._not_found(value)
        self._save_place(uid, field, place.name)
        return "Saved. Your **%s** is %s." % (field, place.name)

    def _cmd_risk(self, t, low, uid):
        if re.search(r"hate being late|must not be late|don'?t be late|"
                     r"predictable|reliable|on time", low):
            self.eng.profiles.set_field(uid, "risk_aversion", 1.9)
            return ("Understood — I will favour routes that are **rarely late**, "
                    "even when a gamble looks faster on paper.")
        if re.search(r"fastest|quickest|just get me there|speed", low) and \
                not re.search(r"\bto\b", low):
            self.eng.profiles.set_field(uid, "risk_aversion", 0.25)
            return ("Done — I will chase the **fastest** option and accept a "
                    "wider spread of arrival times.")
        return None

    def _cmd_feedback(self, t, low, uid):
        m = re.match(r"^\s*(good|bad|great|terrible|awful|nice)\s*"
                     r"(route|trip|one)?\s*$", low)
        if not m:
            return None
        verdict = "up" if m.group(1) in ("good", "great", "nice") else "down"
        last = self.last_plan.get(uid)
        if not last:
            return "Plan a trip with me first, then tell me how it went."
        try:
            plan = self.eng.plan(last[0], last[1], user_id=uid, k=1,
                                 with_baseline=False)
        except ValueError as exc:
            return str(exc)
        return self.rate(uid, plan.best, verdict)

    def rate(self, uid: str, route, verdict: str) -> str:
        """Learn from a thumbs up ("up") or down ("down") on one route."""
        p = self.eng.profiles.get(uid)
        changed = p.record_feedback(self.eng.net, route, verdict)
        self.eng.profiles.save()
        note = (", ".join(list(changed)[:3]) if changed else "your general preferences")
        return ("Noted. I have adjusted %s and set your risk dial to %.2f. "
                "Future routes for you will lean %s."
                % (note, p.risk_aversion,
                   "safer and more predictable" if verdict == "down"
                   else "a little more adventurous"))

    def _cmd_time(self, t, low, uid):
        m = re.match(r"^\s*(?:time|clock|set time)\s+" + TIME_RE + r"\s*$", low)
        if not m:
            return None
        if self.live:
            return ("Live London runs on the real clock, so there is nothing to "
                    "move. It is %s there now." % self.eng.clock)
        self.eng.set_clock(m.group(1))
        s = self.eng.network_stats()
        return ("Network clock set to **%s**. Arterials averaging %s km/h, "
                "%s%% of roads congested, %d incidents live."
                % (self.eng.clock, s["arterial_kph"], s["congested_pct"],
                   s["incidents_active"]))

    def _cmd_incidents(self, t, low, uid):
        if not re.search(r"\bincident|disruption|what'?s happening|jam|blocked\b", low):
            return None
        if self.live:
            return self._hotspots()
        acts = self.eng.sim.active_incidents(self.eng.now_s)
        if not acts:
            return "Nothing significant on the network right now."
        lines = ["**%d disruptions live at %s**" % (len(acts), self.eng.clock)]
        for i in sorted(acts, key=lambda x: -x.factor(self.eng.now_s))[:6]:
            lines.append("• %s — since %s, easing by %s (intensity %d%%)"
                         % (i.label, fmt_clock(i.start_s),
                            fmt_clock(i.start_s + i.duration_s),
                            round(i.factor(self.eng.now_s) * 100)))
        return "\n".join(lines)

    def _hotspots(self) -> str:
        """London has no incident feed; say what the cameras see instead."""
        spots = self.eng.hotspots()
        if not spots:
            return ("No camera readings are fresh enough to judge right now, so "
                    "I will not guess. Try again in a few minutes.")
        lines = ["**Busiest camera-watched roads in London, %s**" % self.eng.clock]
        for label, cong, age_s in spots:
            lines.append("• %s — %s (%d%% congested, %d min ago)"
                         % (label, _mood(1.0 - cong), round(cong * 100),
                            max(1, round(age_s / 60))))
        lines.append("_London publishes no incident feed, so this is what the "
                     "cameras measure._")
        return "\n".join(lines)

    def _cmd_traffic(self, t, low, uid):
        # "whats traffic like", "how's the traffic", "traffic?" - all the same
        # question, and none of them matched the original stricter pattern.
        if re.match(r"^(?:what'?s|whats|hows|how is|how'?s) +(?:the +)?"
                    r"traffic(?: +like)? *\??$", low.strip()):
            low = "traffic"
        m = re.match(r"^ *(?:traffic|conditions|how is|how'?s) *"
                     r"(?:(on|at|in|near|around) +|the +)?(.*)$", low)
        if not m:
            return None
        locative, target = m.group(1), m.group(2).strip(" ?")
        if not target or target in ("it", "things", "now", "traffic"):
            return self._network_summary()

        place = self.eng.resolve(target)
        if place is None:
            # "traffic update" is not a question about a place called "update",
            # but this matcher used to answer it with "I could not find
            # **update**" - the same blame-the-user bug as _cmd_route, from the
            # same cause: a capture treated as a place without being checked.
            #
            # Only a locative ("traffic ON x") is the user claiming x is a
            # place. The geocoder forgives typos - "Park Streat" resolves - so
            # a bare word it cannot resolve was never a place at all, and the
            # honest answer is the general one they asked for.
            if locative:
                return self._not_found(target)
            return self._network_summary()
        return self._road_report(place)

    def _network_summary(self) -> str:
        s = self.eng.network_stats()
        if self.live:
            age = s.get("data_age_s")
            return ("**London live at %s**\n"
                    "• Arterials averaging **%s km/h**\n"
                    "• %s%% of roads congested\n"
                    "• %d of %d cameras reporting%s\n"
                    "• %s%% of the network inferred from them"
                    % (self.eng.clock, s["arterial_kph"], s["congested_pct"],
                       s["cameras_online"], s["cameras_total"],
                       " (readings about %d min old)" % max(1, round(age / 60)) if age else "",
                       s["inferred_pct"]))
        return ("**Network at %s**\n"
                "• Arterials averaging **%s km/h**\n"
                "• %s%% of roads congested\n"
                "• %d incidents live\n"
                "• %d of %d cameras reporting; %s%% of the network inferred from them"
                % (self.eng.clock, s["arterial_kph"], s["congested_pct"],
                   s["incidents_active"], s["cameras_online"],
                   s["cameras_total"], s["inferred_pct"]))

    def _road_report(self, place) -> str:
        """Summarise conditions on the edges around a resolved place."""
        eng, net, st = self.eng, self.eng.net, self.eng.state
        eids = [int(e) for e in net.out_edge_ids(place.node)]
        eids += [int(e) for e in net.in_edge_ids(place.node)]
        if not eids:
            return "I have no live picture for %s right now." % place.name
        import numpy as np
        kph = float(np.mean([st.kph[e] for e in eids]))
        free = float(np.mean([net.ekph[e] for e in eids]))
        ratio = kph / max(free, 1.0)
        watched = sum(1 for e in eids if st.observed[e])
        mood = _mood(ratio)
        out = ["**%s** at %s — %s" % (place.name, eng.clock, mood),
               "• Measured **%.0f km/h** against a free-flow %.0f km/h" % (kph, free),
               "• %s" % ("directly watched by %d camera(s)" % watched if watched
                         else "inferred from nearby cameras, not directly watched")]
        if self.live:
            return "\n".join(out)
        hit = [i for i in eng.sim.active_incidents(eng.now_s)
               if set(int(x) for x in i.edges) & set(eids)]
        if hit:
            out.append("• ⚠ %s, easing by %s"
                       % (hit[0].label, fmt_clock(hit[0].start_s + hit[0].duration_s)))
        return "\n".join(out)

    def _cmd_commute(self, t, low, uid):
        if low.strip() not in ("commute", "my commute", "go to work", "home"):
            return None
        home, work = self._home_work(self.eng.profiles.get(uid))
        if low.strip() == "home":
            o, d = work, home
        else:
            o, d = home, work
        if not o or not d:
            a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]
            return ("Tell me where you travel first: `set home %s` "
                    "and `set work %s`." % (a, b))
        return self.route(o, d, None, uid)

    # -- saved places ---------------------------------------------------------

    def _home_work(self, p):
        """(home, work) for this brain's city. The profile's own ``home`` and
        ``work`` belong to the default city; other cities keep theirs in
        ``places`` so a London commute never overwrites a Kolkata one."""
        if self.city == ACTIVE_CITY:
            return p.home, p.work
        saved = p.places.get(self.city, {})
        return saved.get("home", ""), saved.get("work", "")

    def _save_place(self, uid: str, field: str, name: str) -> None:
        if self.city == ACTIVE_CITY:
            self.eng.profiles.set_field(uid, field, name)
            return
        p = self.eng.profiles.get(uid)
        p.places.setdefault(self.city, {})[field] = name
        self.eng.profiles.save()

    def save_commute(self, uid: str, origin: str, destination: str) -> str:
        """Remember a trip as this user's home -> work commute."""
        self._save_place(uid, "home", origin)
        self._save_place(uid, "work", destination)
        return ("Saved. Home is **%s** and work is **%s**. Send `commute` any "
                "time for the trip right now, or `home` for the way back."
                % (_display(origin), _display(destination)))

    # -- routing ------------------------------------------------------------

    def _parse_arrive_by(self, t, uid):
        """(destination, time, origin) from a deadline phrase, or None."""
        m = re.search(r"(?:reach|get to|get me to|get me|be at|arrive at|arrive in)"
                      r"\s+(.+?)\s+by\s+" + TIME_RE, t, re.I)
        if not m:
            m = re.search(r"(.+?)\s+by\s+" + TIME_RE + r"\s+from\s+(.+)$", t, re.I)
            if not m:
                return None
            return m.group(1), m.group(2), m.group(3)
        dest, when = m.group(1), m.group(2)
        # Split "<destination> from <origin>" with plain string operations
        # rather than a regex. SonarQube flagged the regex version for
        # polynomial backtracking (python:S5852), and for splitting on a fixed
        # delimiter a regex buys nothing over str.rfind, which is obviously
        # linear and easier to read.
        stated_origin, dest = _split_on_from(dest, t)
        home, _ = self._home_work(self.eng.profiles.get(uid))
        return dest, when, stated_origin or home

    def _cmd_arrive_by(self, t, low, uid):
        parsed = self._parse_arrive_by(t, uid)
        if parsed is None:
            return None
        dest, when, origin = parsed
        home, work = self._home_work(self.eng.profiles.get(uid))
        dest = _named_place(dest, home, work)
        problem, origin = _check_trip_ends(origin, dest, work)
        if problem:
            return problem

        try:
            best, deadline = self.eng.leave_by(origin.strip(), dest.strip(),
                                               self._disambiguate_hour(when),
                                               user_id=uid)
        except ValueError as exc:
            return self._explain_failure(exc, origin, dest)
        if not best:
            return ("Even leaving right now you would probably miss **%s**. "
                    "Conditions are bad — consider the metro." % fmt_clock(deadline))
        depart_s, plan = best
        r = plan.best
        self.last_plan[uid] = (plan.origin, plan.destination, depart_s)
        text = ("🕒 **Leave by %s** to reach %s by %s.\n"
                "• Typical run **%d min**, but budget **%d min** to be 90%% safe\n"
                "• Via %s\n"
                "• %s"
                % (fmt_clock(depart_s), _display(plan.destination), fmt_clock(deadline),
                   round(r.median_s / 60), round(r.percentile_s(0.9) / 60),
                   ", ".join(_roads(self.eng.net, r)[:3]),
                   plan.advisory or "Conditions look normal on this corridor."))
        notes = self._match_notes(origin, dest)
        if notes:
            text += "\n\n" + notes
        text = self._with_map(text, plan.origin, plan.destination, depart_s)
        return Reply(text, plan, "leave_by")

    def _cmd_leave_time(self, t, low, uid):
        """"what time should I leave for work?" - the right question, asked
        without the one fact needed to answer it.

        We can answer it exactly, but only backwards from an arrival deadline:
        that is what ``leave_by`` does. Guessing a deadline would be inventing
        the premise of the answer, so ask for it, and offer the journey time as
        the thing we *can* say without one.
        """
        m = LEAVE_TIME_RE.match(t)
        if not m:
            return None
        raw = m.group(1).strip(" ?.,")
        home, work = self._home_work(self.eng.profiles.get(uid))
        named = _named_place(raw, home, work)
        if raw and not named:
            # They said "work" and have not saved one.
            return ("I do not know where that is yet. Set it with "
                    "`set work <place>`, then ask me again.")
        place = self.eng.resolve(named) if named else None
        if place is None:
            return None          # not a place - let the other matchers try
        name = _display(place.name)
        return ("When do you need to be at **%s**? Say `get me to %s by 9:30` "
                "and I will work back from it, with enough slack to be on time "
                "9 days in 10.\n"
                "If you only want the journey as it is now, ask "
                "`how long to %s`." % (name, name, name))

    def _cmd_when_to_go(self, t, low, uid):
        """"Should I go now?" - answered by pricing the same trip at several
        departure times and showing when you would actually *arrive*.

        The honest part is the arrival column. Waiting almost always shortens
        the drive during a clearing peak, and a router that stopped there would
        be telling people to sit at home to make a number look better. You still
        arrive later. Both facts go in the answer, and the recommendation is
        made on arrival time unless waiting genuinely wins.
        """
        if not WHEN_TO_GO_RE.match(t):
            return None

        trip = self.last_plan.get(uid)
        if trip:
            origin, dest = trip[0], trip[1]
        else:
            home, work = self._home_work(self.eng.profiles.get(uid))
            origin, dest = home, work
        if not origin or not dest:
            a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]
            return ("Which trip? Ask me for one first — `%s to %s` — and then "
                    "`should I go now?` compares leaving now against waiting."
                    % (a, b))

        now_s = self.eng.now_s
        rows = []
        for mins in WAIT_OFFSETS_MIN:
            depart = now_s + mins * 60.0
            try:
                plan = self.eng.plan(origin, dest, depart, user_id=uid, k=1,
                                     with_baseline=False)
            except ValueError as exc:
                return self._explain_failure(exc, origin, dest)
            r = plan.best
            rows.append((mins, depart, r.median_s, r.percentile_s(0.9),
                         depart + r.median_s))
        if not rows:
            return None

        quickest = min(rows, key=lambda x: x[2])      # shortest drive
        earliest = min(rows, key=lambda x: x[4])      # earliest arrival
        now_row = rows[0]

        lines = ["🕐 **Go now, or wait?** — %s → %s"
                 % (_display(origin), _display(dest)), ""]
        for mins, depart, med, p90, arrive in rows:
            when = "now" if mins == 0 else "in %d min" % mins
            mark = ""
            if (mins, depart) == (earliest[0], earliest[1]):
                mark = "  ← arrives first"
            lines.append("• leave %-10s (%s) · drive **%d min** · arrive **%s**%s"
                         % (when, fmt_clock(depart), round(med / 60),
                            fmt_clock(arrive), mark))
        lines.append("")

        saving_min = round((now_row[2] - quickest[2]) / 60)
        later_min = round((quickest[4] - now_row[4]) / 60)
        if earliest[0] == 0:
            if saving_min >= 2:
                # The case worth stating plainly: waiting IS a shorter drive
                # and still a later arrival. Saying only the first half would
                # be true and misleading.
                lines.append("**Leave now.** Waiting does shorten the drive — "
                             "%d min now against %d min if you leave in %d — but "
                             "you still arrive about %d min later, so it only "
                             "wins if you would rather spend the time here than "
                             "in traffic."
                             % (round(now_row[2] / 60), round(quickest[2] / 60),
                                quickest[0], later_min))
                lines.append("")
                lines.append("_Waiting gets you there sooner only when the road "
                             "ahead clears by more than you wait, which on a "
                             "%d-minute trip essentially means an incident "
                             "lifting. Most 'traffic is improving' advice quietly "
                             "ignores this._" % round(now_row[2] / 60))
            else:
                lines.append("**Leave now.** Waiting does not buy you a shorter "
                             "drive either — conditions are not improving on "
                             "this corridor.")
        else:
            lines.append("**Wait %d minutes.** Leaving at %s gets you there at "
                         "%s, which is earlier than leaving right now (%s) — "
                         "the corridor is clearing faster than the delay costs "
                         "you."
                         % (earliest[0], fmt_clock(earliest[1]),
                            fmt_clock(earliest[4]), fmt_clock(now_row[4])))
        lines.append("")
        lines.append("_Budget %d min to be on time 9 days in 10 if you go now._"
                     % round(now_row[3] / 60))
        return "\n".join(lines)

    def _cmd_compare(self, t, low, uid):
        """"Which is faster, Esplanade or Sealdah?" - one origin, two ends.

        Runs after _cmd_when_to_go, which owns the "now or later" sense of the
        same shape ("is it better to go now or in an hour" is a question about
        departure times, not destinations).

        Both ends are resolved before anything is planned; if fewer than two
        are places we were wrong about the intent and hand back rather than
        guess at one.
        """
        q = low.strip().rstrip("?!. ")
        if " or " not in q or not any(w in q for w in COMPARE_WORDS):
            return None

        body = COMPARE_TAIL_RE.sub("", COMPARE_LEAD_RE.sub("", q)).strip()
        home, work = self._home_work(self.eng.profiles.get(uid))
        named = [_named_place(p.strip(" ,.?"), home, work)
                 for p in body.split(" or ")]
        places, seen = [], set()
        for n in named:
            got = self.eng.resolve(n) if n else None
            if got is not None and got.name not in seen:
                seen.add(got.name)
                places.append(got)
        if len(places) < 2:
            return None
        places = places[:MAX_COMPARE]

        trip = self.last_plan.get(uid)
        origin = (trip[0] if trip else home) or home
        if not origin:
            return ("From where? Save a starting point with `set home <place>`, "
                    "then ask me again.")

        scored = []
        for p in places:
            try:
                plan = self.eng.plan(origin, p.name, user_id=uid, k=1,
                                     with_baseline=False)
            except ValueError:
                continue        # same place as the origin, or unroutable
            scored.append((p.name, plan.best))
        if len(scored) < 2:
            return None
        scored.sort(key=lambda x: x[1].median_s)

        win_name, win = scored[0]
        # Remember the winner, so the natural next question works: "which is
        # faster, A or B?" then "should I go now?" / "and back" / "what about
        # at 9" should all be about the one they were just told to pick.
        self.last_plan[uid] = (origin, win_name, self.eng.now_s)

        lines = ["⚖️ **%s** from %s, leaving %s"
                 % (" vs ".join(n for n, _ in scored), _display(origin),
                    fmt_clock(self.eng.now_s)), ""]
        for i, (name, r) in enumerate(scored):
            lines.append("• %s%s%s — **%d min** (%.1f km) · usually %d–%d min"
                         % ("**" if i == 0 else "", name, "**" if i == 0 else "",
                            round(r.median_s / 60), r.distance_m / 1000,
                            round(r.p10_s / 60), round(r.p90_s / 60)))
        lines.append("")

        second_name, second = scored[1]
        gap_min = round((second.median_s - win.median_s) / 60)
        if gap_min < 2:
            lines.append("**Too close to call** — %d min between them, which is "
                         "inside the spread on either. Pick on something else."
                         % gap_min)
        else:
            lines.append("**%s is about %d min quicker.**" % (win_name, gap_min))
            # The point of the whole project, when it happens to show up here.
            if win.distance_m > second.distance_m:
                lines.append("")
                lines.append("_Note it is also %.1f km **further**. Distance and "
                             "time come apart in traffic, which is why picking "
                             "the shortest route is not the same as picking the "
                             "quickest one._"
                             % ((win.distance_m - second.distance_m) / 1000))
        return "\n".join(lines)

    def _cmd_at_origin(self, t, low, uid):
        """"I'm at Sealdah, how long home" - origin stated, destination asked.

        The strict "A to B" form assumes the user writes the origin first and
        bare; in practice they say where they are in a clause of their own. The
        comma is the split, and both halves are checked against the map before
        anything is planned, for the reason in _cmd_route.
        """
        m = AT_ORIGIN_RE.match(t)
        if not m:
            return None
        origin_raw, rest = m.group(1).strip(" ?.,"), m.group(2).strip()
        home, work = self._home_work(self.eng.profiles.get(uid))

        origin = _named_place(origin_raw, home, work)
        o_place = self.eng.resolve(origin) if origin else None
        if o_place is None:
            return None

        # "how long home" / "how far to work" - strip the question, keep the place.
        rest = AT_ORIGIN_ASK_RE.sub("", rest).strip(" ?.,")
        dest = self._strip_filler(rest) or rest
        dest = _named_place(dest, home, work)
        d_place = self.eng.resolve(dest) if dest else None
        if d_place is None:
            return None
        return self.route(o_place.name, d_place.name, None, uid)

    def _cmd_bare_place(self, t, low, uid):
        """A message that is nothing but a place: "Esplanade", "Sealdah please".

        Runs last, after every structured matcher has declined, so it cannot
        take "A to B" from the router. A person who types one word is asking to
        go there from where they usually start, which is what the saved home is
        for.
        """
        q = low.strip().rstrip("?!. ").strip()
        for tail in (" please", " pls", " plz", " thanks"):
            if q.endswith(tail):
                q = q[: -len(tail)].strip()
        if not q or " to " in " %s " % q or len(q.split()) > 4:
            return None
        home, work = self._home_work(self.eng.profiles.get(uid))
        named = _named_place(q, home, work)
        if not named:
            return None
        place = self.eng.resolve(named)
        if place is None:
            return None
        return self._destination_only(place.name, uid)

    def _cmd_scope(self, t, low, uid):
        """Name the limit, for questions this assistant genuinely cannot answer.

        Runs LAST, so anything that could have been routed already was. A
        message reaching here is one no intent could handle, and the useful
        reply is which of our limits it ran into - not "I did not catch a route
        in that", which blames the user for a boundary we chose.

        The public-transport answer in particular is a limitation `README.md`
        already states. A bot that cannot say its own documented limits out
        loud is hiding them.
        """
        q = low.strip().rstrip("?!. ")
        if not q:
            return None
        a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]

        if q in IDENTITY:
            return ("I am Triffy — a commute assistant, and a student "
                    "prototype, not a product. I predict what the roads will "
                    "be like *when you get there* rather than what they are "
                    "like now, and I try to tell you when I am unsure.\n"
                    "Send `help` for everything I can do, or try `%s to %s`."
                    % (a, b))
        if q in CAPABILITY:
            return self.help_text()
        if q in ("where am i", "where am i now", "do you know where i am",
                 "can you see where i am"):
            # Worth answering straight: we do not track anyone. On Telegram a
            # shared pin arrives as a one-off origin and is not stored.
            home, _work = self._home_work(self.eng.profiles.get(uid))
            extra = (" I assume you are starting from **%s** unless you say "
                     "otherwise." % home) if home else ""
            return ("I do not know where you are — I have no location on you "
                    "unless you send one, and I do not keep it.%s" % extra)

        # Substring match, because these arrive inside sentences ("is the metro
        # running", "what about the bus"). Word-boundary checked so "bus" does
        # not fire on "business" or a road with it in the name.
        words = set(re.findall(r"[a-z]+", q))
        for triggers, reply in OUT_OF_SCOPE:
            for trig in triggers:
                hit = (trig in words if " " not in trig
                       else trig in q)
                if hit:
                    return reply % (a, b)
        return None

    def _cmd_route(self, t, low, uid):
        # "A to B", optionally "... at 18:45"
        m = re.search(r"^\s*(?:route\s+|from\s+)?(.+?)\s+(?:to|->|→)\s+(.+?)"
                      r"(?:\s+(?:at|leaving at|departing)\s+" + TIME_RE + r")?\s*$",
                      t, re.I)
        if not m:
            return None
        origin, dest, when = m.group(1).strip(), m.group(2).strip(), m.group(3)

        # Check the captured origin is really a place before committing to it.
        #
        # This regex takes everything before " to " as an origin, so "i want to
        # go to Esplanade" yielded origin="i want" and the user was told
        # "I could not find **i want** inside the mapped area" - the bot blaming
        # them for words it had invented. The FILLER list guards the phrasings
        # we thought of; this guards the ones we did not, which was four of the
        # six failures found by testing real phrasing.
        #
        # The geocoder already knows what is and is not a place, so ask it. If
        # the origin does not resolve but the destination does, there was no
        # origin - fall through and let _cmd_destination_only use the saved home.
        resolved_dest = self.eng.resolve(dest)
        resolved_origin = self.eng.resolve(origin)

        # When BOTH ends fail to resolve, whether to report the miss depends on
        # whether the user was naming places at all.
        #
        # "Narnia to Mordor" is two noun phrases: the user asserted they are
        # places, so say we could not find them, with suggestions. But
        # "am i going to be late" also contains " to ", and answering it with
        # "I could not find **am i going**" is the blame-the-user bug again -
        # words we cut out of their sentence, quoted back as their mistake.
        #
        # The discriminator is grammatical, not a list of phrasings: a sentence
        # opening with an auxiliary verb or a wh-word is a QUESTION, not a place
        # pair. That matters because it is a *closed* class - English has a
        # fixed set of auxiliaries and wh-words - so unlike the FILLER lead-ins
        # it cannot quietly go out of date as people phrase things differently.
        if resolved_origin is None and resolved_dest is None and _is_question(t):
            return None

        if resolved_origin is None and resolved_dest is not None:
            # Hand on the canonical name, not the raw capture. The regex is
            # non-greedy and splits on the FIRST " to ", so "i want to go to
            # Esplanade" leaves dest="go to esplanade"; the fuzzy matcher still
            # finds it, but echoing that back ("Say `<place> to go to
            # esplanade`") reads like the bot is malfunctioning.
            return self._destination_only(resolved_dest.name, uid, when)

        return self.route(origin, dest, when, uid)

    def route(self, origin: str, dest: str, when, uid: str) -> Reply:
        """Plan a trip and word it. ``when`` is a clock string, seconds, or None."""
        try:
            plan = self.eng.plan(origin, dest, when, user_id=uid, k=3)
        except ValueError as exc:
            return Reply(self._explain_failure(exc, origin, dest))
        self.last_plan[uid] = (plan.origin, plan.destination, plan.depart_s)
        text = self.route_card(plan)
        notes = self._match_notes(origin, dest)
        if notes:
            text += "\n\n" + notes
        text = self._with_map(text, plan.origin, plan.destination, plan.depart_s)
        return Reply(text, plan, "route")

    def map_link(self, origin: str, dest: str, depart_s=None) -> str:
        """A deep link that reopens this exact trip on our own dashboard.

        Deliberately our map and not a Google Maps directions link. A
        `maps.google.com/dir/` URL renders *Google's* route, which may take
        different roads than the one we just recommended - so the picture would
        contradict the answer it is attached to, and there would be no way to
        see what our router actually chose. The whole separation between what
        is ours and what is not is the honesty mechanism here (§5), so the
        picture has to be ours too.

        Returns "" when MAP_BASE is empty, which is how you turn this off when
        nothing is serving the dashboard.

        Also returns "" when the reader is somewhere else and MAP_BASE is a
        loopback address. **127.0.0.1 on someone's phone means their phone.**
        A Telegram user who taps that link gets "127.0.0.1 refused to connect"
        and reasonably concludes the bot is broken — which is exactly what
        happened the first time this shipped. Telegram gets the drawn PNG
        instead, which needs no network of ours at all.

        To give remote users a working link, serve the dashboard on the network
        (`TRIFFY_HOST=0.0.0.0`) and set `TRIFFY_MAP_BASE` to that address.
        """
        if not MAP_BASE or not origin or not dest:
            return ""
        if self.remote and _is_loopback(MAP_BASE):
            return ""
        q = {"from": origin, "to": dest}
        if depart_s is not None:
            q["at"] = fmt_clock(depart_s)
        if self.live:
            q["mode"] = "live"
        return "%s/?%s" % (MAP_BASE, urlencode(q))

    def _with_map(self, text: str, origin: str, dest: str, depart_s=None) -> str:
        link = self.map_link(origin, dest, depart_s)
        return "%s\n\n🗺 See it on the map: %s" % (text, link) if link else text

    def route_card(self, plan, index: int = 0) -> str:
        """The words for one route of a plan: the recommendation (index 0),
        with the alternatives listed, or one alternative on its own."""
        net = self.eng.net
        r = plan.routes[index]
        lines = ["🚦 **%s → %s**, leaving %s"
                 % (_display(plan.origin), _display(plan.destination),
                    fmt_clock(plan.depart_s))]
        lines.append("")
        label = r.label if index == 0 else "Route %d of %d" % (index + 1, len(plan.routes))
        lines.append("**%s · %d min** (%.1f km)"
                     % (label, round(r.median_s / 60), r.distance_m / 1000))
        lines.append("Usually %d–%d min · %d%% predictable"
                     % (round(r.p10_s / 60), round(r.p90_s / 60),
                        round(r.reliability * 100)))
        lines.append("Via %s" % ", ".join(_roads(net, r)[:3]))

        if index == 0:
            lines += self._recommendation_extras(plan)
        else:
            best = plan.best
            lines.append("")
            lines.append("_%s than the recommended route, which is usually %d min._"
                         % ("Slower" if r.median_s > best.median_s else "No slower",
                            round(best.median_s / 60)))

        lines.append("")
        lines.append("**Directions**")
        for i, s in enumerate(r.steps[:6], start=1):
            lines.append("%d. %s — %d m" % (i, s.instruction, s.distance_m))
        if len(r.steps) > 6:
            lines.append("…and %d more turns." % (len(r.steps) - 6))
        return "\n".join(lines)

    @staticmethod
    def _recommendation_extras(plan) -> list:
        lines = []
        if plan.advisory:
            lines += ["", "💡 %s" % plan.advisory]

        if len(plan.routes) > 1:
            lines += ["", "**Alternatives**"]
            for alt in plan.routes[1:]:
                lines.append("• %s — %d min (p90 %d), %.1f km"
                             % (alt.label, round(alt.median_s / 60),
                                round(alt.p90_s / 60), alt.distance_m / 1000))

        b = plan.baseline
        if b and b.claimed_s:
            err = b.mean_s - b.claimed_s
            if abs(err) > 60:
                lines += ["", "_A snapshot-based app would have promised %d min "
                              "on its route and delivered about %d._"
                              % (round(b.claimed_s / 60), round(b.mean_s / 60))]
        return lines

    def _disambiguate_hour(self, when: str) -> str:
        """Resolve a bare hour to whichever of am/pm comes round next.

        "get me home by 6" said at 18:30 parsed to 06:00, which is in the past,
        rolled to tomorrow morning, and answered a question nobody asked. A bare
        hour is genuinely ambiguous, and the tie-break people actually mean is
        "the next time it is 6 o’clock". At 14:00 that gives 18:00 today; at
        18:30 it correctly gives 06:00 tomorrow.

        Only bare hours are touched: "6:30" and "6pm" already say what they mean.
        """
        txt = (when or "").strip().lower()
        if not re.fullmatch(r"\d{1,2}", txt):
            return when
        hh = int(txt)
        if not 1 <= hh <= 11:
            return when
        now = self.eng.now_s % 86400.0
        best, best_wait = when, None
        for cand in (hh, hh + 12):
            secs = cand * 3600.0
            wait = (secs - now) % 86400.0
            if best_wait is None or wait < best_wait:
                best_wait, best = wait, "%02d:00" % cand
        return best

    # -- follow-ups -----------------------------------------------------------

    def _cmd_followup(self, t, low, uid):
        """Questions about the last trip: "what about at 9?", "15 min later",
        "way back", "what about to Sealdah?". People ask these as a
        conversation, not as a fresh query."""
        change = _followup_change(low)
        if change is None:
            return None
        last = self.last_plan.get(uid)
        if not last:
            a, b = EXAMPLES.get(self.city, EXAMPLES["kol"])[:2]
            return "Plan a trip first, for example `%s to %s`." % (a, b)
        origin, dest, depart_s = last
        kind, value = change
        if kind == "at":
            return self.route(origin, dest, self._disambiguate_hour(value), uid)
        if kind == "later":
            return self.route(origin, dest, depart_s + 60.0 * value, uid)
        if kind == "again":
            return self.route(origin, dest, None, uid)
        if kind == "back":
            return self.route(dest, origin, None, uid)
        if kind == "to":
            return self.route(origin, value, None, uid)
        return self.route(value, dest, None, uid)

    # -- natural phrasing ----------------------------------------------------

    # People do not type "A to B". They type "how long to Sealdah" while
    # standing at the origin, and expect us to know where they are. Every one of
    # these lead-ins was collected from a real phrasing that the parser choked
    # on, treating "fastest way" as a place name and failing to find it.
    FILLER = (
        r"how (?:long|far) (?:is it |does it take )?(?:to|until)",
        r"(?:what(?:'|’)?s|whats) the (?:fastest|quickest|best) (?:way|route) to",
        r"(?:fastest|quickest|best|shortest) (?:way|route) to",
        r"should i leave (?:now )?for",
        # "to" optional: "get me home" and "take me home" are the forms the
        # code below has always claimed to support, and neither parsed at all -
        # the pattern demanded a "to" that nobody says before "home".
        r"(?:can you )?(?:get|take|route|direct|bring) me(?: to)?",
        r"(?:give me )?(?:directions|route|way) to",
        r"i(?:'|’)?m going to",
        r"i need to (?:get|be) (?:to|at)",
        r"take me to",
        r"head(?:ing)? to",
        r"going to",
        # Round two, from testing real phrasing again after the "i want to go
        # to X" fix. Each of these reached the user as a parse failure.
        r"(?:can you )?drop me (?:at|off at|off near|to|near)",
        r"leav(?:e|ing) (?:now |soon )?for",
        r"(?:i(?:'|’)?m )?off to",
        r"(?:i )?want to (?:go|get) to",
    )

    def _strip_filler(self, text: str):
        """Pull a bare destination out of a conversational sentence.

        Returns the destination, or None if nothing matched.
        """
        low = text.strip().lower()
        for pat in self.FILLER:
            m = re.match(r"^\s*" + pat + r"\s+(.+?)\s*\??$", low, re.I)
            if m:
                return m.group(1).strip(" ?.")
        # The same question in Bengali or Hindi puts the verb last, so the
        # place leads and no lead-in pattern can see it.
        for tail in LOCAL_ASK_TAILS:
            if low.rstrip(" ?.").endswith(tail):
                rest = low.rstrip(" ?.")[: -len(tail)].strip(" ?.,")
                if rest:
                    return rest

        # "ghore jabo" - the sentence ends in the verb, which TRAILING_WORDS
        # already enumerates for localise(). Reused rather than duplicated so
        # the two paths cannot drift apart.
        #
        # Only returns when a verb was ACTUALLY stripped. That guard is
        # load-bearing: this matcher runs before _cmd_route, so returning the
        # whole message on no match would hijack every "A to B" ever sent.
        words = low.rstrip(" ?.").split()
        kept = list(words)
        while kept and kept[-1] in TRAILING_WORDS:
            kept.pop()
        if kept and len(kept) < len(words):
            return " ".join(kept)
        return None

    def _cmd_destination_only(self, t, low, uid):
        """Handle 'how long to X' - a destination with no stated origin.

        Falls back to the user's saved home, which is the whole reason `set
        home` exists. If they have not set one, ask for it once rather than
        failing with a syntax hint, because the user is not writing a query
        language - they are asking a question.
        """
        dest = self._strip_filler(t)
        if not dest:
            return None
        return self._destination_only(dest, uid)

    def _destination_only(self, dest: str, uid: str, when=None):
        """Route to ``dest`` from the user's saved home.

        Shared by the filler-stripping matcher and by _cmd_route, which falls
        back here when the text before " to " turns out not to be a place.
        """
        # "get me home" / "take me to work" resolve against the profile. This
        # must happen before the geocoder sees the word: "office" fuzzy-matches
        # a real street named Officers Colony, so an unsaved "office" used to
        # route somewhere plausible and wrong rather than asking.
        home, work = self._home_work(self.eng.profiles.get(uid))
        dest = _named_place(dest, home, work)
        if not dest:
            return ("I do not know where that is yet. Tell me with "
                    "`set home <place>` or `set work <place>`, then ask again.")
        place = self.eng.resolve(dest)
        if place is None:
            return self._not_found(dest)

        origin = home
        # "get me home" / "ghore jabo" asked FROM home is not a trip. They mean
        # from wherever they spend the day, and work is the only other place we
        # know about - the same reasoning _check_trip_ends uses, and the same
        # reading _cmd_commute already gives a bare "home". Without this the
        # worded forms answered "Origin and destination are the same place",
        # which is true and useless.
        if home and place.name.strip().lower() == home.strip().lower():
            origin = work

        if not origin:
            # Name the place the way the map does. Whichever matcher got here
            # first, the user sees "Say `<place> to Esplanade`" - never the raw
            # lower-cased fragment we happened to cut out of their sentence,
            # which reads like the bot is quoting its own confusion back.
            # Advise the field they have NOT filled. Telling someone to
            # `set home` when home is exactly what they just asked for reads
            # like the bot has not been listening.
            missing = "set work" if home else "set home"
            return ("Where are you starting from? Say `<place> to %s`, or save "
                    "it with `%s <place>`." % (_display(place.name), missing))
        return self.route(origin, dest, when, uid)

    def trip_places(self, text: str) -> list:
        """The place names a message asks about, without planning anything.

        The Telegram bot uses this to notice a London trip sent while in
        Kolkata mode (or the reverse) before answering it in the wrong city.
        """
        t = localise((text or "").strip())
        m = re.search(r"^\s*(?:route\s+|from\s+)?(.+?)\s+(?:to|->|→)\s+(.+?)"
                      r"(?:\s+(?:at|leaving at|departing)\s+" + TIME_RE + r")?\s*$",
                      t, re.I)
        if m and not re.search(r"\bby\s+\d", t, re.I):
            return [m.group(1).strip(), m.group(2).strip()]
        dest = self._strip_filler(t)
        return [dest] if dest else []

    # -- helpers ------------------------------------------------------------

    def _match_notes(self, *texts) -> str:
        """Say so when a place was read loosely: "I read X as Y"."""
        notes = []
        for text in texts:
            if not text or COORD_RE.fullmatch(text.strip()):
                continue
            place, how = self.eng.net.match(text)
            if place is None or how == "exact" or _norm(text) == _norm(place.name):
                continue
            notes.append("_I read “%s” as %s._" % (text.strip(), place.name))
        return "\n".join(notes)

    def _explain_failure(self, exc: ValueError, *texts) -> str:
        """A planning error in words that help: which place was not found,
        and what nearby names we do know."""
        for text in texts:
            if text and self.eng.resolve(text) is None:
                return self._not_found(text)
        return str(exc)

    def _not_found(self, text: str) -> str:
        sug = self.eng.net.suggest(text, limit=5)
        if sug:
            return ("I could not find **%s**. Did you mean: %s?"
                    % (text, ", ".join(sug)))
        return ("I could not find **%s** inside the mapped area (%s). "
                "Try a nearby landmark or a street name."
                % (text, self.eng.net.meta.get("label", "")))


_FROM = " from "


def localise(text: str) -> str:
    """Rewrite "X theke Y (jabo)" and "X se Y (tak jana hai)" as "X to Y".

    Word-based rather than a regex, for the same reason as ``_split_on_from``:
    a fixed set of words needs no backtracking. Only a message with exactly one
    route word and no English "to" is touched, so an English sentence that
    happens to contain "se" is left alone.
    """
    words = text.split()
    if " to " in " %s " % text.lower():
        return text
    hits = [i for i, w in enumerate(words) if w.lower() in ROUTE_WORDS]
    if len(hits) != 1 or hits[0] == 0:
        return text
    i = hits[0]
    right = words[i + 1:]
    while right and right[-1].lower() in TRAILING_WORDS:
        right.pop()
    if not right:
        return text
    return "%s to %s" % (" ".join(words[:i]), " ".join(right))


FOLLOW_AGAIN = {"again", "recheck", "re-check", "check again", "refresh",
                "same again", "and now", "now", "what about now", "update"}


def _followup_change(low: str):
    """What a follow-up asks to change: ("at", clock), ("later", minutes),
    ("back", None), ("to"/"from", place) - or None if it is not one."""
    q = low.strip().rstrip("? ").strip()
    # "again" re-runs the same trip against the traffic as it is NOW, which is
    # the whole point of a live router: the answer changes while you stand
    # there deciding. Offered by name in the affirmation reply, so it has to work.
    if q in FOLLOW_AGAIN:
        return "again", None
    if q in FOLLOW_BACK:
        return "back", None
    m = FOLLOW_TIME.match(q) or FOLLOW_ABOUT_TIME.match(q)
    if m:
        return "at", m.group(1).strip()
    m = FOLLOW_LATER.match(q)
    if m:
        return "later", int(m.group(1))
    if q in ("later", "leave later", "leaving later", "go later"):
        return "later", DEFAULT_LATER_MIN
    m = FOLLOW_PLACE.match(q)
    if m:
        return m.group(1), m.group(2)
    return None


# Things people reasonably ask a commuting assistant that this one does not do.
#
# Answering these with "I did not catch a route in that" is wrong twice over:
# it implies the user mis-typed, and it hides a limitation the README states
# openly. Each entry names the limit instead. Keep the wording honest - these
# are things we do not do, not things we are about to do.
OUT_OF_SCOPE = (
    (("metro", "underground", "tube", "bus", "train", "tram", "ferry",
      "local train", "public transport"),
     # Deliberately city-neutral: this same brain answers for London, where
     # "no Underground" is an even more pointed admission than in Kolkata.
     "I only know road routes — no metro, bus or rail. That is a real gap, not "
     "an oversight: the honest answer to a commute is often a mixed one, and "
     "this prototype cannot give it. Road trips I can do: try `%s to %s`."),
    (("uber", "ola", "rapido", "book me", "book a", "call me a", "call an",
      "hail"),
     "I cannot book anything — I have no connection to any cab service. I can "
     "tell you how long the drive will take, which is the part that is usually "
     "wrong: try `%s to %s`."),
    (("weather", "rain", "raining", "storm", "flood"),
     "I do not have a weather feed. Worth saying that rain is exactly when "
     "these predictions matter most, and exactly when this prototype is "
     "weakest. For the drive itself: `%s to %s`."),
    (("fare", "cost", "price", "how much will it cost", "cheaper", "cheapest"),
     "I do not know fares or fuel costs — I optimise for arriving on time, not "
     "for what it costs. For the timing: `%s to %s`."),
    # NOT a bare "park": this city has Park Street and Park Circus, and the
    # single-word triggers are matched against whole words. "parking" is safe,
    # "park" would fire on two of the demo's own landmarks.
    # NB no bare "to park" here either - it is a substring of "how long to
    # park street", which must keep routing.
    (("parking", "car park", "park my car", "park the car", "where to park",
      "can i park"),
     "I do not know anything about parking. I stop at the destination road. "
     "For the drive there: `%s to %s`."),
)

# "who are you" / "what can you do". Answered honestly - it IS a prototype, and
# saying so costs nothing and buys credibility.
IDENTITY = ("who are you", "what are you", "are you a bot", "are you real",
            "are you human", "are you an ai", "what is this", "whats this",
            "how does this work", "how do you work", "what do you do")
CAPABILITY = ("what can you do", "what can i ask", "what can you help with",
              "what do you know", "commands", "options", "menu")
# Bare affirmations. Not thanks - "ok" can close a conversation, but "yes"
# answers a question, and we never asked one. Logged by three separate users
# in the first minutes of miss logging, which is why this exists.
AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "y", "haan", "han", "hyan",
          "হ্যাঁ", "हाँ", "please", "pls", "do it", "go ahead"}

SELF_KNOWLEDGE = ("what do you know about me", "what do you know about me?",
                  "what have you learned about me", "my profile", "about me")

# English auxiliaries and wh-words. A closed grammatical class, deliberately:
# unlike a list of phrasings it does not go stale as people say things
# differently, because the language does not grow new auxiliaries.
QUESTION_OPENERS = frozenset("""
    is are am was were do does did will would shall should can could may might
    have has had what which who whom whose when where why how
""".split())


def _is_question(text: str) -> bool:
    """Does this open like a question rather than like a place name?

    Used to decide whether an unresolvable "X to Y" was someone naming two
    places badly ("Narnia to Mordor" - tell them) or asking something that
    merely contains the word "to" ("am i going to be late" - do not accuse them
    of inventing a place called "am i going").
    """
    words = (text or "").strip().lower().lstrip("¿").split()
    if not words:
        return False
    if words[0] in QUESTION_OPENERS:
        return True
    # "i" + auxiliary covers "i should leave", "i am going", "i can make it".
    return len(words) > 1 and words[0] == "i" and words[1] in QUESTION_OPENERS


LOOPBACK_HOSTS = frozenset(("127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"))


def _is_loopback(base: str) -> bool:
    """Is this URL only reachable from the machine serving it?

    "0.0.0.0" counts: it is a valid bind address but not a valid destination -
    a browser handed http://0.0.0.0:8000 has nowhere to go.
    """
    host = urlsplit(base).hostname or ""
    return host.lower() in LOOPBACK_HOSTS or host.startswith("127.")


def _display(name: str) -> str:
    """A shared location reaches us as "lat, lon"; people call it that."""
    return "your location" if COORD_RE.fullmatch((name or "").strip()) else name


def _named_place(dest, home, work):
    """Resolve "home" and "work" to the places this user saved."""
    d_low = (dest or "").strip().lower()
    # The Bengali words are here for the same reason ROUTE_WORDS are: the
    # product is for Kolkata, and "ghore jabo" is how someone says "I am going
    # home". They resolve against the saved profile, never against the
    # geocoder - see the Officers Colony trap below.
    if d_low in ("home", "my home", "my place", "back home",
                 "ghor", "ghore", "bari", "barite", "ঘরে", "বাড়ি"):
        return home
    if d_low in ("work", "my work", "office", "the office", "my office",
                 "kaaj", "kaj", "অফিস", "দপ্তর"):
        return work
    return dest


def _check_trip_ends(origin, dest, work):
    """(problem message or "", origin to use). A trip to itself is taken to
    start from work, the only other place we know about."""
    if not dest:
        return ("I do not know where that is yet. Set it with "
                "`set home <place>` or `set work <place>`."), origin
    if not origin:
        return ("Where are you starting from? Either add `from <place>` or "
                "save one with `set home <place>`."), origin
    if origin.strip().lower() == dest.strip().lower():
        # "get me home by 6" from home: they mean from wherever they are.
        if not work:
            return ("You are asking me to route from %s to itself. Tell me "
                    "where you are starting with `<place> to %s`, or save it "
                    "with `set work <place>`." % (dest, dest)), origin
        return "", work
    return "", origin

# What people call their vehicle, mapped to the profile's vehicle types.
_VEHICLE_WORDS = {
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle",
    "scooter": "motorcycle", "auto": "auto", "rickshaw": "auto",
    "taxi": "taxi", "cab": "taxi",
}


def _mood(ratio: float) -> str:
    """Words for a road running at `ratio` of its free-flow speed."""
    if ratio > 0.75:
        return "flowing well"
    if ratio > 0.55:
        return "a bit slow"
    if ratio > 0.35:
        return "congested"
    return "close to gridlock"


def _split_on_from(dest: str, whole: str):
    """Split a phrase on its last case-insensitive " from ".

    Returns (origin, destination). Either may be empty.

    Deliberately not a regex. The patterns this replaced were flagged for
    polynomial backtracking, and a fixed-delimiter split needs nothing more
    than str.rfind - which is linear, obvious, and cannot be made to
    misbehave by a long input.
    """
    src = whole if _FROM in whole.lower() else dest
    low = src.lower()
    idx = low.rfind(_FROM)
    if idx < 0:
        return "", dest.strip()
    origin = src[idx + len(_FROM):].strip()

    # The destination is whatever precedes " from ", but only if that split
    # point falls inside the destination phrase we were given.
    d_low = dest.lower()
    d_idx = d_low.rfind(_FROM)
    clean_dest = dest[:d_idx].strip() if d_idx >= 0 else dest.strip()
    return origin, clean_dest


def _roads(net, route) -> list:
    acc: dict = {}
    for e in route.edges:
        nm = net.ename[e]
        if nm:
            acc[nm] = acc.get(nm, 0.0) + float(net.elen[e])
    out = [k for k, _ in sorted(acc.items(), key=lambda kv: -kv[1])]
    return out or ["local streets"]
