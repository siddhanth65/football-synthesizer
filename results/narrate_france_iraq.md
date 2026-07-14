# France 3–0 Iraq — pundit narration (LLM layer, grounded)

*Example output of `report/narrate.py` on the France-Iraq evidence bundle. Every claim below traces to a
CV metric or an official FIFA number in that bundle — the SYSTEM prompt forbids anything else. This is the
hand-verified target; with `ANTHROPIC_API_KEY` set the module produces it automatically for any match.*

---

France set up in a 4-3-3 and, true to type, they came to take the game to Iraq. Our tracking has their
defensive line sitting around 52 metres up the pitch with a build-up height near 58 — this is a side that
squeezes the game high and lives in the opposition half, with roughly a third of their shape (32%)
permanently camped in the attacking third. FIFA's own numbers back the picture up: 39% of the match spent
in build-up, another 19% in controlled progression. France want the ball at their feet and they want to
carry it forward.

The problem they were handed is the one every big side meets at a tournament — a bus to break down. Iraq
spent nearly half the game, 46%, in a low block, and another 22% in a mid block, sitting deep and daring
France to find a way through. That is precisely the puzzle France's front line is built for. With Mbappé
leading it and Olise and Barcola stretching the width, they kept working the ball into the channels and
around the edges of that block — and it told. France completed 117 line breaks to Iraq's 89, and the
quality was there, not just the volume: 2.3 expected goals against 0.7, Mbappé the focal point of almost
all of it.

What's striking is that this wasn't a possession stranglehold — the ball split almost evenly, 49% to
France. This was a clinical performance, not a suffocating one. And it fits a pattern we can now measure
across France's group stage: the deeper an opponent sits, the further forward France commit. Against
Iraq's deep block they pushed higher and harder than against sides that come out to play.

Without the ball, France don't sit off for a second. They defended mostly in a mid block (28% of the game)
and pressed high (8%), dropping into a low block only 7% of the time — a genuine front-foot posture that
turned Iraq over 42 times. This is a team comfortable defending forwards, confident it can win the ball
back before the opponent settles.

One honest note on the read: the shape numbers — line height, width, attacking-third share — are measured
directly from the broadcast and are reliable. The counting stats (line breaks, possession, turnovers) come
from FIFA's official report. Where our computer vision estimates possession itself, treat it as indicative;
the structural story is the solid ground.
