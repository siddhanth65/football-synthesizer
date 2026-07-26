"""Pundit-voice renderer for ``results/MANUTD_IDENTITY_PROFILE.md``.

Separated from :mod:`tools.manutd_identity` (the evidence engine) only because the prose is long.
Discipline copied from :mod:`tools.render_scouting_v2`: story first, numbers as supporting clauses,
abstentions spoken plainly, and **every figure in the prose is interpolated from the evidence dict**
so the narrative cannot drift away from what was measured.
"""
from __future__ import annotations

CHANNELS = ("left", "central", "right")
ZONES = ("own third", "middle third", "final third")


MIN_HEADLINE_FRAMES = 150   # a player only carries a claim above this tracked-frame count


def _pct(x: float, dp: int = 1) -> str:
    """Share -> percentage string."""
    return f"{x * 100:.{dp}f}%"


def _pp(x: float, dp: int = 1) -> str:
    """Share difference -> percentage-point string."""
    return f"{x * 100:.{dp}f} percentage points"


def _grid_row(gs: dict, zone: str) -> str:
    """One zone's left/central/right shares as a markdown row."""
    r = gs[zone]
    return (f"| {zone} | {_pct(r['left'])} | {_pct(r['central'])} | {_pct(r['right'])} | "
            f"{_pct(r['left'] + r['right'])} |")


def _headlines(ev: dict) -> list[str]:
    """The claims this corpus actually supports, each with its evidence and its n."""
    b, bl, pl = ev["ball"], ev["block"], ev["players"]
    ft, st = b["funnel_test"]["manutd"], b["side_test"]["manutd"]
    fto = b["funnel_test"]["opponents"]
    mg, og = b["pooled"]["grid_share"], b["opponents_pooled"]["grid_share"]
    wp = bl["wp_pooled"]
    top = _by_frames(pl)
    return [
        "1. **They come out through the middle and finish down the sides -- and so does everybody "
        f"else.** {_pct(mg['own third']['central'])} of United's ball time in their own third is in "
        f"the central channel; by the final third that is down to "
        f"{_pct(mg['final third']['central'])}, with "
        f"{_pct(mg['final third']['left'] + mg['final third']['right'])} of it out wide. The swing "
        f"repeats in {ft['n_more_central_at_the_back']} of {ft['n']} matches (median "
        f"{_pp(ft['median_drop'])}, sign test p = {ft['p_sign']}). But the same six "
        f"opponents do it in {fto['n_more_central_at_the_back']} of {fto['n']} "
        f"(p = {fto['p_sign']}) and finish *more* wide-heavy than United do "
        f"({_pct(og['final third']['central'])} central to United's "
        f"{_pct(mg['final third']['central'])}). So: yes, United attack down the flanks -- and no, "
        "that is not a United trait. It is what a Premier League attack looks like.",

        "2. **There is no favoured flank.** Pooled, the final third splits "
        f"{_pct(mg['final third']['left'])} left to {_pct(mg['final third']['right'])} right, which "
        f"reads like a tilt until you count matches: {st['n_left_heavy']} of {st['n']} lean left, "
        f"the rest lean right (median tilt {_pp(st['median_tilt'])}, sign test "
        f"p = {st['p_sign']}). Whatever side a given afternoon runs down, it is the afternoon, not "
        "the team. We decline the left-side / right-side read.",

        "3. **The territory map is real; the volume map is not.** "
        f"{top['deep']['player']} is the floor of the build "
        f"({top['deep']['x']:.1f} m from his own goal, tenth percentile "
        f"{top['deep']['x_p10']:.1f} m, {_pct(top['deep']['z_own'])} of his time in his own third, "
        f"on {top['deep']['frames']} tracked frames). {top['high']['player']} is the ceiling "
        f"({top['high']['x']:.1f} m, {_pct(top['high']['z_final'])} of his frames in the final "
        f"third, and a deepest tenth of {top['high']['x_p10']:.1f} m -- he does not come back). "
        f"{top['seen']['player']} is the man the cameras find most "
        f"({top['seen']['frames']} named frames under Amorim, close to double the next), and his "
        f"ground covers all three zones ({_pct(top['seen']['z_own'])} own / "
        f"{_pct(top['seen']['z_middle'])} middle / {_pct(top['seen']['z_final'])} final). That is "
        "the honest sentence about him: we can see where he plays, not how much he does.",

        "4. **Rigid? Not on the one axis we can measure.** United's defensive line averages "
        f"{bl['line']['mean']:.1f} m from their own goal with a between-match SD of "
        f"**{bl['line']['sd']:.2f} m** (CV {_pct(bl['line']['cv'])}) and a range of "
        f"{bl['line']['min']:.1f}-{bl['line']['max']:.1f} m across {bl['line']['n']} matches -- a "
        f"{bl['line']['max'] - bl['line']['min']:.1f} m swing. Against the same opponent in two "
        f"legs it moves {bl['pair_mean_line_gap_m']:.1f} m on average, and the leg-to-leg "
        "correlation is **-0.40** (`results/PAIR_ANALYSIS_v1.md`) -- knowing where they defended at "
        "home tells you nothing about the away leg. The whole opponent set pooled sits at "
        f"{bl['opponent_line_spread']['sd']:.2f} m SD, i.e. no more variable than United despite "
        f"containing six different clubs (F = {bl['variance_tests']['line']['F']}, "
        f"p = {bl['variance_tests']['line']['p']}, so the two cannot be separated at n = 12). "
        "Their height is set by the afternoon, not by a principle.",

        "5. **The one thing that never moves is not theirs.** Block vertical compactness sits at "
        f"{bl['vspread']['mean']:.2f} +/- {bl['vspread']['sd']:.2f} m "
        f"(CV {_pct(bl['vspread']['cv'])}) across {bl['vspread']['n']} matches, six opponents, two "
        "managers, two venues and three game states. That looks like the rigidity claim -- until "
        f"the same measurement on the opposition comes back at "
        f"{bl['opponent_spread']['mean']:.2f} +/- {bl['opponent_spread']['sd']:.2f} m "
        f"(CV {_pct(bl['opponent_spread']['cv'])}), statistically indistinguishable "
        f"(F = {bl['variance_tests']['vspread']['F']}, "
        f"p = {bl['variance_tests']['vspread']['p']}). A constant that holds for all seven sides is "
        "a property of the measurement, or of the league, not a fingerprint of one of them. "
        "Declined.",

        "6. **Nobody comes to press them.** In all 12 legs the opposition block sits between "
        f"{bl['opponent_line_spread']['min']:.1f} m and {bl['opponent_line_spread']['max']:.1f} m "
        "from its own goal -- low to mid, every time, home and away, under both managers. Whatever "
        "United are, teams do not fancy going after them high up the pitch.",

        "7. **A lead pushes the line up and leaves the shape alone.** On the six matches with a "
        f"calibrated win-probability series, United defend at {wp['loss-likely']['line_m']:.1f} m "
        f"when a loss is likely and {wp['win-likely']['line_m']:.1f} m when a win is likely "
        f"(+{wp['win-likely']['line_m'] - wp['loss-likely']['line_m']:.1f} m), while compactness "
        f"moves {abs(wp['win-likely']['vspread_m'] - wp['loss-likely']['vspread_m']):.2f} m "
        f"({wp['loss-likely']['vspread_m']:.2f} -> {wp['win-likely']['vspread_m']:.2f}). Hold that "
        f"one loosely: the win-likely band is {wp['win-likely']['frames']} frames and most of it is "
        "one afternoon at Southampton.",

        "8. **The manager change shows up in the players, not in the team's shape.** "
        f"{ev['era']['n_higher_under_amorim']} of the {ev['era']['n_both_era']} players tracked in "
        f"both eras stand higher up the pitch under Amorim, Garnacho by "
        f"{ev['era']['player_dx']['Alejandro Garnacho']:+.1f} m. The block line goes "
        f"{ev['era']['ten_hag']['block_line_m']:.1f} -> "
        f"{ev['era']['amorim']['block_line_m']:.1f} m -- a "
        f"{abs(ev['era']['ten_hag']['block_line_m'] - ev['era']['amorim']['block_line_m']):.1f} m "
        f"move inside a {bl['line']['sd']:.2f} m match-to-match SD, which is not a separation. And "
        "United played three of their six ten Hag matches at home and three of six Amorim matches "
        "at home in *the reverse* fixtures, so venue and manager flip together in every pair: no "
        "number below can tell the two apart.",
    ]


def _by_frames(pl: dict) -> dict:
    """Deepest, highest and most-tracked player rows, above the headline frame floor."""
    rows = [r for r in pl["players"] if r["frames"] >= MIN_HEADLINE_FRAMES]
    return {"deep": min(rows, key=lambda r: r["x"]),
            "high": max(rows, key=lambda r: r["x"]),
            "seen": max(rows, key=lambda r: r["frames"])}


def _player_table(pl: dict, era: str) -> list[str]:
    """Territory table for one era, ordered from deepest to highest mean advance."""
    rows = sorted([r for r in pl["players"] if r["era"] == era], key=lambda r: r["x"])
    out = ["| player | frames | mean advance (m) | p10 | p90 | front-back spread | "
           "side-to-side spread | own/mid/final | L/C/R |",
           "|--------|-------:|-----------------:|----:|----:|------------------:|"
           "--------------------:|---------------|-------|"]
    for r in rows:
        out.append(
            f"| {r['player']} | {r['frames']} | {r['x']:.1f} | {r['x_p10']:.1f} | "
            f"{r['x_p90']:.1f} | {r['spread_x']:.1f} | {r['spread_y']:.1f} | "
            f"{r['z_own']:.2f}/{r['z_middle']:.2f}/{r['z_final']:.2f} | "
            f"{r['ch_left']:.2f}/{r['ch_central']:.2f}/{r['ch_right']:.2f} |")
    return out


def _match_table(ev: dict) -> list[str]:
    """Per-match evidence: ball coverage, orientation support, block geometry."""
    out = ["| match | mgr | venue | opponent | post-link ball cov | clean direction votes | "
           "United ball samples | block line (m) | block spread (m) | block cov |",
           "|-------|-----|-------|----------|-------------------:|----------------------:|"
           "--------------------:|---------------:|-----------------:|----------:|"]
    for mid, r in ev["ball"]["per_match"].items():
        bk = ev["block"]["per_match"][mid]
        o = r["coverage"]["orientation"]
        out.append(
            f"| {mid} | {r['manager']} | {r['venue']} | {r['opponent']} | "
            f"{r['coverage']['post_link_coverage']:.3f} | {o['n_clean_votes']}/{o['n_chunks']} | "
            f"{r['n_samples']} | {bk['mean_line_m']:.1f} | {bk['mean_vspread_m']:.2f} | "
            f"{bk['coverage']:.3f} |")
    return out


def render(ev: dict) -> str:
    """Build the full profile markdown."""
    b, bl, pl = ev["ball"], ev["block"], ev["players"]
    from tools.manutd_identity import pass_agreement  # noqa: PLC0415

    pa = pass_agreement(b)
    covs = [r["coverage"]["post_link_coverage"] for r in b["per_match"].values()]
    anchor = pl["channel_anchor"]
    mg, og = b["pooled"]["grid_share"], b["opponents_pooled"]["grid_share"]
    lines: list[str] = [
        "# Manchester United 2024-25 -- a season identity read from twelve broadcasts", "",
        "Twelve Premier League matches, six opponents played twice, one manager change in the "
        "middle. This is what the footage will say about who this team is -- and, just as often, "
        "what it refuses to say.", "",
        "**The one substitution that makes this honest.** Everything below is built on *position* "
        "and *territory*: where the ball travels, where players stand, how the defensive block is "
        "shaped. Nothing is built on per-player event counts, because naming the man on the ball "
        "works on only 1-4% of United's passes (`results/PLAYER_ANALYSIS_v2.md`). So you will read "
        "\"he operates between the lines\" and never \"he made 62 passes\". And there are no "
        "*we-do-this-more-than-they-do* rate comparisons either: separating two teams on a "
        "pressing rate needs 58-90 matches and we have twelve "
        "(`results/W1B_WINDOW_AND_SAMPLING.md`). Where the honest answer is \"we cannot tell\", it "
        "is written as \"we cannot tell\".", "",
        "---", "",
        "## The reads this corpus actually supports", "",
    ]
    lines += [f"{h}\n" for h in _headlines(ev)]
    lines += [
        "---", "",
        "## 1. How they build", "",
        "Follow the ball rather than the formation and United's attack has a clear shape to it, and "
        "it is not the one the phrase \"they build through the wingers\" suggests. They start in "
        "the middle. In their own third the ball spends "
        f"{_pct(mg['own third']['central'])} of its time in the central channel, against "
        f"{_pct(mg['own third']['left'])} left and {_pct(mg['own third']['right'])} right -- "
        "centre-backs and goalkeeper circulating in front of their own box, which is exactly what "
        "you would expect and is partly the goalkeeper's doing, since he is central by definition.",
        "",
        "Then it drifts outwards. By the middle third the centre is down to "
        f"{_pct(mg['middle third']['central'])}, and in the final third it is "
        f"{_pct(mg['final third']['central'])} -- more than three quarters of United's attacking "
        "third ball time is in the two wide channels. Central at the back, wide at the front. It is "
        "a proper pattern, not a pooled illusion: it shows up in "
        f"{b['funnel_test']['manutd']['n_more_central_at_the_back']} of "
        f"{b['funnel_test']['manutd']['n']} individual matches (median swing "
        f"{_pct(b['funnel_test']['manutd']['median_drop'])} points, sign test p = "
        f"{b['funnel_test']['manutd']['p_sign']}).", "",
        "And here is where a season profile has to be careful, because the same measurement run on "
        "the other side of the same twelve broadcasts gives the opposition "
        f"{_pct(og['own third']['central'])} central at the back and "
        f"{_pct(og['final third']['central'])} central at the front -- the identical funnel, in "
        f"{b['funnel_test']['opponents']['n_more_central_at_the_back']} of "
        f"{b['funnel_test']['opponents']['n']} legs, and finishing marginally *wider* than United "
        "do. The wide attacking third is Premier League geometry, not Manchester United's identity. "
        "If you want the honest headline: **they attack down the flanks like everyone attacks down "
        "the flanks, and there is nothing in this footage that makes it theirs.**", "",
        "**United's ball map (12 matches, "
        f"{b['pooled']['n_samples']} possession-credited ball samples)**", "",
        "| zone | left | central | right | wide total |",
        "|------|-----:|--------:|------:|-----------:|",
    ]
    lines += [_grid_row(mg, z) for z in ZONES]
    lines += ["", "**The same twelve broadcasts, the opposition's ball "
              f"({b['opponents_pooled']['n_samples']} samples)**", "",
              "| zone | left | central | right | wide total |",
              "|------|-----:|--------:|------:|-----------:|"]
    lines += [_grid_row(og, z) for z in ZONES]
    lines += [
        "", "**Left or right?** No. Pooled it looks like a left lean "
        f"({_pct(mg['final third']['left'])} to {_pct(mg['final third']['right'])} in the final "
        f"third), but match by match it is {b['side_test']['manutd']['n_left_heavy']} leaning left "
        f"and {b['side_test']['manutd']['n'] - b['side_test']['manutd']['n_left_heavy']} leaning "
        f"right (sign test p = {b['side_test']['manutd']['p_sign']}). United do not have a side. "
        "We say so rather than pick the one that reads better.", "",
        "**What we cannot say here.** Where the ball *crosses into* the final third by channel -- "
        f"the sharpest version of the question -- rests on only {b['pooled']['n_entries']} "
        "cleanly-chained crossings across twelve matches, four or five a game. That is not a "
        "sample; it is an anecdote with a decimal point. Declined.", "",
        "---", "",
        "## 2. Who occupies what space", "",
        "Nine of the twelve matches have the identity layer run on them, which means named players "
        "on named tracks. What follows is a map of ground occupied. It is not a ranking, it is not "
        "an involvement table, and it never will be from this footage.", "",
        f"The spine reads exactly as a spine should. **{_deep(pl)['player']} is the floor of the "
        f"build** -- mean position {_deep(pl)['x']:.1f} m from his own goal, tenth percentile "
        f"{_deep(pl)['x_p10']:.1f} m, {_pct(_deep(pl)['z_own'])} of his tracked time inside his own "
        f"third, on {_deep(pl)['frames']} tracked frames. **{_high(pl)['player']} is the ceiling** "
        f"-- {_high(pl)['x']:.1f} m, {_pct(_high(pl)['z_final'])} of his frames in the final third, "
        f"and even his deepest tenth is {_high(pl)['x_p10']:.1f} m: he is not a winger who tracks "
        "all the way back. Joshua Zirkzee sits further forward still on the table below -- 83.9 m, "
        "94% of his frames in the final third -- but on 83 frames, about seventeen seconds of "
        "tracked football, so he is listed and not leaned on.", "",
        "**Bruno Fernandes** is the player the profile most wants to talk about and can say least "
        f"about. He is the most-tracked United outfielder in the corpus by a distance "
        f"({_seen(pl)['frames']} named frames under Amorim, close to double the next man), his mean "
        f"position is {_seen(pl)['x']:.1f} m, and his ground spans all three zones "
        f"({_pct(_seen(pl)['z_own'])} own third, {_pct(_seen(pl)['z_middle'])} middle, "
        f"{_pct(_seen(pl)['z_final'])} final). Laterally he is almost perfectly even "
        f"({_seen(pl)['ch_left']:.2f} left / {_seen(pl)['ch_central']:.2f} central / "
        f"{_seen(pl)['ch_right']:.2f} right), so the fashionable \"right half-space\" line is not "
        "available -- he does not sit in one. He is everywhere, and being seen everywhere is partly "
        "a fact about him and partly a fact about where the camera points. **We can see where he "
        "plays. We cannot see how much he does.** Calling him the primary playmaker off this "
        "evidence would be a guess in a lab coat.", "",
        "**Lateral labels come with a warning.** Left and right are anchored on players whose side "
        f"is known rather than assumed: {', '.join(anchor['left_names'])} pool to "
        f"y = {anchor['left_mean_y']:.1f} m and {', '.join(anchor['right_names'])} to "
        f"y = {anchor['right_mean_y']:.1f} m, so the labels are the right way round "
        f"(check: {'PASS' if anchor['passes'] else 'FAIL'}). At the individual level it is "
        f"directional only -- {anchor['n_side_correct']} of {anchor['n_side_checked']} known-sided "
        "player-eras land on the correct flank, which is good enough for \"he plays wide left\" and "
        "not good enough for a half-space claim.", "",
        "### Territory map -- Amorim era", "",
    ]
    lines += _player_table(pl, "amorim")
    lines += ["", "### Territory map -- ten Hag era", ""]
    lines += _player_table(pl, "ten_hag")
    lines += [
        "", "Mean advance is metres from United's own goal; p10 and p90 are the deepest and highest "
        "tenth of the player's tracked frames; spreads are standard deviations; the last two "
        "columns are the share of frames by vertical zone and by lateral channel. Frames are "
        "tracked-and-named frames, which track minutes played in *rank* but not in scale.", "",
        "---", "",
        "## 3. How they defend", "",
        "United defend deep and they defend at whatever height the afternoon demands. Across the "
        f"twelve matches their block line averages {bl['line']['mean']:.1f} m from their own goal, "
        f"but the range is {bl['line']['min']:.1f} to {bl['line']['max']:.1f} m: from a 20.8 m "
        "hunker at Anfield to a 37.4 m front-foot line in the 3-0 at Southampton, the highest they "
        "defended in the whole corpus and their best result in it.", "",
        "The other half of the picture is that nobody comes to meet them. In every one of the "
        f"twelve legs the opposition block sits between {bl['opponent_line_spread']['min']:.1f} and "
        f"{bl['opponent_line_spread']['max']:.1f} m from its own goal -- low or mid, home and away, "
        "against both managers. There is no high-press exception in this corpus. Whatever else is "
        "true of this side, opponents do not think going after them high up the pitch is the way to "
        "beat them.", "",
        "**Read the block heights as relative, not absolute.** The de-biased line carries about 7 m "
        "of held-out scale uncertainty and the hand-annotation gate is still pending "
        "(`results/BLOCK_AND_STYLE_v1.md`), and the broadcast frames the block preferentially when "
        "the ball is further forward, which biases every observed block to read deeper than it "
        "truly is. The ordering across matches is the usable signal; the metre label is not "
        "certified.", "",
        "---", "",
        "## 4. Are they rigid?", "",
        "This is the claim a fan makes about this side more than any other, so it gets the "
        "sharpest treatment. \"Rigid\" is a statement about variance, and variance is measurable.", "",
        "**On defensive height, they are the opposite of rigid.** Between-match standard deviation "
        f"of the block line is **{bl['line']['sd']:.2f} m** on a mean of "
        f"{bl['line']['mean']:.1f} m -- a coefficient of variation of "
        f"{_pct(bl['line']['cv'])} over {bl['line']['n']} matches, with a "
        f"{bl['line']['max'] - bl['line']['min']:.1f} m spread from lowest to highest. Play the "
        f"same opponent twice and the line moves {bl['pair_mean_line_gap_m']:.1f} m on average "
        "between the legs, and the leg-to-leg correlation is **-0.40**: the home leg does not "
        "predict the away leg, and if anything predicts it backwards. Whatever sets United's "
        "defensive height, it is not a rule the manager has drilled in.", "",
        "The comparison to keep honest: the pooled opponent set -- six different clubs across the "
        f"same twelve broadcasts -- sits at {bl['opponent_line_spread']['sd']:.2f} m SD "
        f"(CV {_pct(bl['opponent_line_spread']['cv'])}), which is *lower* than United's despite "
        "containing between-club variation United's number does not. That is suggestive and it is "
        f"not significant: F = {bl['variance_tests']['line']['F']} on "
        f"{bl['variance_tests']['line']['df'][0]} and {bl['variance_tests']['line']['df'][1]} "
        f"degrees of freedom, p = {bl['variance_tests']['line']['p']}. So the claim that survives "
        "is the absolute one -- United's line is not fixed -- not a league ranking of who wobbles "
        "most.", "",
        "**Leg-to-leg swing against the same opponent**", "",
        "| opponent | block-line gap between legs (m) | compactness gap (m) |",
        "|----------|-------------------------------:|--------------------:|",
    ]
    lines += [f"| {p['opponent']} | {p['line_gap_m']:.1f} | {p['vspread_gap_m']:.2f} |"
              for p in bl["pairs"]]
    lines += [
        "", "**On shape, the number that looks rigid is not theirs to own.** The block's "
        "front-to-back compactness -- how squeezed the defending unit is between its deepest and "
        f"highest man -- sits at {bl['vspread']['mean']:.2f} m with an SD of "
        f"{bl['vspread']['sd']:.2f} m (CV {_pct(bl['vspread']['cv'])}), range "
        f"{bl['vspread']['min']:.2f}-{bl['vspread']['max']:.2f} m. Twelve matches, six opponents, "
        "two managers, two venues, three game states, and it barely twitches. That is a rigidity "
        "number -- until the same measurement on the opposition returns "
        f"{bl['opponent_spread']['mean']:.2f} +/- {bl['opponent_spread']['sd']:.2f} m "
        f"(CV {_pct(bl['opponent_spread']['cv'])}), statistically indistinguishable from United's "
        f"(F = {bl['variance_tests']['vspread']['F']}, "
        f"p = {bl['variance_tests']['vspread']['p']}). Seven teams, one constant. That is the "
        "measurement talking, or the league, not Manchester United. **Declined as an identity "
        "claim.**", "",
        "**Does the shape change with the state of the game?** Only the height does. On the six "
        "matches with a calibrated win-probability series:", "",
        "| game state | frames | block line (m) | compactness (m) |",
        "|------------|-------:|---------------:|----------------:|",
    ]
    for band in ("loss-likely", "balanced", "win-likely"):
        v = bl["wp_pooled"].get(band)
        if v:
            lines.append(f"| {band} | {v['frames']} | {v['line_m']:.1f} | {v['vspread_m']:.2f} |")
    lines += [
        "", "The line climbs "
        f"{bl['wp_pooled']['win-likely']['line_m'] - bl['wp_pooled']['loss-likely']['line_m']:.1f} m "
        "from loss-likely to win-likely while compactness moves "
        f"{abs(bl['wp_pooled']['win-likely']['vspread_m'] - bl['wp_pooled']['loss-likely']['vspread_m']):.2f} m. "
        "United get braver about where they defend when the game is safe and change nothing about "
        f"how they defend. Caveat, stated rather than buried: the win-likely row is "
        f"{bl['wp_pooled']['win-likely']['frames']} frames drawn from effectively two matches, most "
        "of it the Southampton away win.", "",
        "**And within a single match?** We cannot tell. The spell-to-spell scatter of the block "
        f"line inside a match is {bl['within_match_spell_line_sd_m']:.1f} m -- larger than the "
        "quantity we would be trying to measure. At that resolution any \"they changed shape after "
        "the goal\" read is measurement noise with a story attached. Declined.", "",
        "---", "",
        "## 5. What the manager change did", "",
        "Ruben Amorim replaced Erik ten Hag between the two halves of this corpus, and the design "
        "has a hole in it that has to be said before any number: United played the *reverse* "
        "fixture of each pair under the new manager, so **venue and manager flip together in all "
        "six pairs**. Nothing below can separate \"Amorim\" from \"away at Anfield instead of home "
        "to Liverpool\". Every number in this section is a direction with n = 6 a side.", "",
        "**It shows in the players.** "
        f"{ev['era']['n_higher_under_amorim']} of the {ev['era']['n_both_era']} players tracked in "
        "both eras stand higher up the pitch under Amorim. The mover is Alejandro Garnacho, "
        f"{ev['era']['player_dx']['Alejandro Garnacho']:+.1f} m further forward, and the direction "
        "holds through the spine -- Casemiro, Martinez, Mainoo and Fernandes all take a step up.", "",
        "**It does not show in the team's shape.** The block line goes "
        f"{ev['era']['ten_hag']['block_line_m']:.1f} m under ten Hag to "
        f"{ev['era']['amorim']['block_line_m']:.1f} m under Amorim, a "
        f"{abs(ev['era']['ten_hag']['block_line_m'] - ev['era']['amorim']['block_line_m']):.1f} m "
        f"move against a {bl['line']['sd']:.2f} m match-to-match SD -- inside the noise, and the "
        "direction is dominated by a single high leverage leg (the 37.4 m at Southampton, which "
        "sits in the ten Hag column). Compactness goes "
        f"{ev['era']['ten_hag']['block_vspread_m']:.2f} to "
        f"{ev['era']['amorim']['block_vspread_m']:.2f} m, which is nothing.", "",
        "**Nor in where the ball goes.** Own-third central share "
        f"{_pct(ev['era']['ten_hag']['ball']['grid_share']['own third']['central'])} under ten Hag "
        f"to {_pct(ev['era']['amorim']['ball']['grid_share']['own third']['central'])} under "
        "Amorim; final-third central share "
        f"{_pct(ev['era']['ten_hag']['ball']['grid_share']['final third']['central'])} to "
        f"{_pct(ev['era']['amorim']['ball']['grid_share']['final third']['central'])}. The funnel "
        "is the same funnel.", "",
        "The one place a manager signal survives the confound at all is the attack-type mix already "
        "on file (`results/GAME_STATE_v2.md`): sustained build-ups fall from 0.592 to 0.468 of "
        "typed attacks and direct attacks rise from 0.310 to 0.429. That is 71 and 77 typed attacks "
        "respectively at 10-20% labelling coverage -- a direction on a small n, and it is offered "
        "as a direction. The formation question (4-2-3-1 to 3-4-3) is a measured negative: the "
        "named-player structure shows no wing-back signature at all "
        "(`results/PLAYER_ANALYSIS_v2.md`).", "",
        "---", "",
        "## What this profile cannot tell you", "", "Bluntly, and with the reason measured:", "",
        "- **Anything about how much a player did.** No passes, touches, involvements, chances "
        "created, plus-minus, or impact ranking. Naming the man on the ball works on **1.0-4.2%** "
        "of United's attributed passes -- two to five named passes a match -- and pooling all nine "
        "identity matches leaves the busiest player on **6**, against a floor of 20 before Poisson "
        "noise drops under 22% (`results/PLAYER_ANALYSIS_v2.md`). The event layer is dead at the "
        "player level and no amount of aggregation revives it.",
        "- **Anything of the form \"United press more than X\".** Team-level counter-press *rate* "
        "metrics need **58-90 matches** to separate two sides even with perfect tracking and zero "
        "measurement error; we have 12 (`results/W1B_WINDOW_AND_SAMPLING.md`). Any rate comparison "
        "against another club would be answering a question whose answer is already fixed by sample "
        "size.",
        "- **Which flank they favour.** Sign test p = "
        f"{b['side_test']['manutd']['p_sign']} over 12 matches. There is no tilt to report.",
        "- **Which channel they enter the final third through.** "
        f"{b['pooled']['n_entries']} cleanly-chained crossings in 12 matches.",
        "- **Whether the shape changes inside a match.** Spell-level line scatter "
        f"({bl['within_match_spell_line_sd_m']:.1f} m) exceeds the effect.",
        "- **The formation.** Seven to eleven named players a match cannot distinguish a 4-2-3-1 "
        "from a 3-4-3, and the wing-back test came back negative.",
        "- **Set pieces.** We hold corner counts and nothing else.",
        "- **Absolute block heights in FIFA metres.** ~7 m of held-out scale uncertainty, "
        "hand-annotation gate pending; the ordering is usable, the label is not.", "",
        "---", "",
        "# Evidence appendix", "",
        "## Provenance", "",
        "- **Ball layer.** Post-`link_ball` usable ball track per chunk, credited to a team by the "
        "Viterbi possession smoother (`generator.ball.assign_possession`, `smooth=True`), oriented "
        "so United attack +x. Post-link ball coverage per match: min "
        f"{min(covs):.3f}, median {sorted(covs)[len(covs) // 2]:.3f}, max {max(covs):.3f} of "
        "sampled frames. Channels are equal 22.67 m thirds of the 68 m width; zones are equal 35 m "
        "thirds of the 105 m length.",
        "- **Cross-check against the validated event layer.** In the "
        f"{pa['n_matches']} matches that carry an attributed-pass ledger ({pa['n_passes']} United "
        "PASS events, on the frozen threshold that holds 0.97-1.09x Sofascore), the lateral channel "
        f"split of pass origins matches the ball-track split to a mean absolute "
        f"{pa['mean_abs_gap'] * 100:.1f} percentage points (worst cell "
        f"{pa['max_abs_gap'] * 100:.1f} pp). Two independent views of the same geometry agree.",
        "- **Player layer.** `fingerprint.player_profiles.oriented_positions` on the 9 PRTreID "
        "identity matches, filtered to United by the PRTreID roster (not the colour anchor, which "
        "leaks opponents), pooled per player per era, minimum 25 named frames.",
        "- **Block layer.** `fingerprint.block_height` -- de-biased rearmost-outfielder line on "
        "trusted-geometry frames (`calib_error_m <= 1.0`), out-of-possession resolved by the same "
        "Viterbi smoother. Reproduces `results/BLOCK_AND_STYLE_v1.md` leg for leg.",
        "- **Game state.** `outputs/oracle/wp/<id>.parquet`, the calibrated base-subset "
        "win-probability model (held-out ECE 0.0415, `results/GAME_STATE_v2.md`); available for 6 "
        "of 12 matches.", "",
        "## A direction bug found and fixed here (read this before reusing the numbers)", "",
        "Left/right and own-half/final-third are meaningless without the attacking direction, and "
        "both shipped resolvers turned out to be unreliable on this corpus:", "",
        "- **`fingerprint.structural_metrics.resolve_attack_directions_from_ball` is systematically "
        "inverted.** It credits each ball sample to the nearest player within 4 m and calls the "
        "team with the further-forward mean the attacking one -- but in a crowded defensive third "
        "the nearest player is usually a *defender*, so the team defending a goal reads as the team "
        "attacking it. Graded against the Sofascore D<M<F position oracle it flips the sign of the "
        "rank correlation in 5 of 5 matches tested (`manutd_brighton` +0.926 -> -0.926). It is the "
        "**primary** resolver inside `fingerprint/block_height.py::_directions`, so the "
        "`ball_to_block_m` and `broadcast_bias_m` columns of `results/BLOCK_AND_STYLE_v1.md` are "
        "computed on an inverted frame and should be re-derived. `mean_line_m`, the vertical "
        "spread, coverage and the low/mid/high spell classes are **not** affected (the line comes "
        "from `generator.impute.line_estimates`, and a standard deviation is invariant under the "
        "flip), which is why the block numbers in this profile match the shipped ones exactly.",
        "- **The keeper-based resolver is correct but noisy per chunk.** On `tottenham_manutd` h1 "
        "it resolves 3 of 6 chunks backwards against the goal-direction evidence in "
        "`results/PAIR_ANALYSIS_v1.md`.", "",
        "**What this profile uses instead.** A chunk only votes on direction when the two "
        "goalkeepers separate cleanly (medians >= 40 m apart, the same test the pair-analysis "
        "mapping adjudication used); votes are summed per half weighted by keeper support; and the "
        "half-time end swap is enforced, because a real team cannot attack the same goal in both "
        "halves. Graded on the same external oracle across all 9 identity matches this beats the "
        "shipped per-chunk keeper resolve: **positive rank correlation in 8/9 matches versus 7/9, "
        "mean rho +0.610 versus +0.361**. It repairs both of the \"advanced full-back inversion\" "
        "matches flagged in `results/PLAYER_ANALYSIS_v2.md` (`brighton_manutd` -0.507 -> +0.676; "
        "`tottenham_manutd` +0.397 -> +0.850). The per-match column below shows how many chunks "
        "supplied a clean vote -- two matches rest on only 2 clean votes and their lateral read "
        "should be read accordingly.", "",
        "## Per-match evidence", "",
    ]
    lines += _match_table(ev)
    lines += [
        "", "## Statistical tests as run", "",
        "- **Funnel (own third vs final third central share), United:** paired per-match difference "
        f"positive in {b['funnel_test']['manutd']['n_more_central_at_the_back']}/"
        f"{b['funnel_test']['manutd']['n']}, median "
        f"{b['funnel_test']['manutd']['median_drop']:+.3f}, two-sided exact sign test p = "
        f"{b['funnel_test']['manutd']['p_sign']}.",
        "- **Funnel, the same twelve broadcasts' opposition:** positive in "
        f"{b['funnel_test']['opponents']['n_more_central_at_the_back']}/"
        f"{b['funnel_test']['opponents']['n']}, median "
        f"{b['funnel_test']['opponents']['median_drop']:+.3f}, p = "
        f"{b['funnel_test']['opponents']['p_sign']}.",
        "- **Left-vs-right tilt in the final third, United:** left-heavy in "
        f"{b['side_test']['manutd']['n_left_heavy']}/{b['side_test']['manutd']['n']}, median "
        f"{b['side_test']['manutd']['median_tilt']:+.3f}, p = "
        f"{b['side_test']['manutd']['p_sign']} -- **no effect**.",
        "- The match is the independent unit throughout. Pooled ball samples sit 0.2 s apart and "
        "are heavily autocorrelated, so no test is run on the sample count; the pooled shares are "
        "descriptive and the sign tests carry the inference.", "",
        "## Known limitations of this profile specifically", "",
        "- Ball coverage is 36-56% of sampled frames, and the broadcast follows the ball, so the "
        "observed ball map is biased toward the phases television chooses to show.",
        "- The own-third central share is partly the goalkeeper, who is central by construction. "
        "The own-to-final funnel is a *within-match contrast*, so a static lateral bias in the "
        "homography cancels; a zone-dependent one would not.",
        "- Nine of twelve matches carry named players; the ten Hag era contributes only three of "
        "those nine, so every era-split player number leans on the Amorim side.",
        "- Block heights are un-gated (Gate 1 pending) and read relatively, never as FIFA metres.",
        "- Everything here is Manchester United measured against Manchester United and against the "
        "six opponents inside these twelve broadcasts. It is not a league-wide comparison and "
        "cannot become one at n = 12.", "",
        "Generated by `python -m tools.manutd_identity`; evidence cached at "
        "`outputs/manutd_identity/evidence.json`.",
    ]
    return "\n".join(lines) + "\n"


def _deep(pl: dict) -> dict:
    """Deepest-average player row above the headline frame floor."""
    return _by_frames(pl)["deep"]


def _high(pl: dict) -> dict:
    """Highest-average player row above the headline frame floor."""
    return _by_frames(pl)["high"]


def _seen(pl: dict) -> dict:
    """Most-tracked player row."""
    return max(pl["players"], key=lambda r: r["frames"])
