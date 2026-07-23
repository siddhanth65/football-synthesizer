# Style-Research Source Map

Companion to `STYLE_RESEARCH_XGFC.md` (the xG FC online catalogue) and `STYLE_RESEARCH_METHODS.md`
(the ranked synthesis). This file is the **wider source map**: the substacks, blogs, coaches,
clubs, federations, institutes, university labs and recurring authors that sit at the intersection
of computer vision, football tactics/formation analysis and player-data analytics — the genre
`thexgfootballclub.substack.com` writes in, plus the papers behind it.

**Reading lens (from the project brief):** we run a *censored broadcast* pipeline — single moving
camera, ~11.8 of 22 players visible, 26.4% of frames with usable pitch geometry, 32-52% usable ball
track, GS-HOTA 22.85. Every source below is tagged for what it gives us *at that data regime*.
Sources that assume full-pitch commercial tracking (TRACAB / Second Spectrum / Metrica) are flagged
`FLAG: full-tracking` — usable for vocabulary and method calibration, not for direct computation.

One section per source type. Obvious repeats (Spielverlagerung, Friends of Tracking, Training
Ground Guru, KU Leuven DTAI, MLSA, MIT Sloan, Van Haaren, Memmert) are merged. A shortlist of 8
sources worth monitoring continuously closes the file.

---

## 1. Substacks & practitioner newsletters

The papers-to-practice register — the exact voice our grounded pundit reports imitate.

### The xG Football Club — Alex Marin Felices *(anchor exemplar)*
- **URL:** https://thexgfootballclub.substack.com
- **Focus:** Reviews academic football-analytics papers and turns them into actionable metrics —
  xT buildup value, passing fingerprints, chance-creation structure. Author is a Nottingham Forest
  data scientist.
- **Why (broadcast/ManU):** The anchor. Its paper reviews flag which methods assume full
  event/tracking data vs event-only inputs — a ready-made filter for what our censored pipeline can
  actually implement.
- **Key works:** *The 12 Essential Reads from Year 1*; *Quantifying Player Performance in Buildup
  Play with Expected Threat*; *This is How Football Teams Create Scoring Opportunities*.

### The Transfer Flow — Ted Knutson (StatsBomb founder)
- **URL:** https://www.thetransferflow.com
- **Focus:** Team analysis, transfer-market reasoning, how professionals read squads from data.
- **Why:** Demonstrates the analyst's voice — style claims tied to specific metrics with stated
  confidence — the register our LLM narration layer should imitate for ManU opposition analysis.
- **Key works:** *The Transfer Flow* archive; *The Transfer Flow Podcast*.

### Talking Tactics — Daryl Dao
- **URL:** https://www.talking-tactics.com
- **Focus:** EPL tactics explainers blending video stills with data points; maintains a curated
  list of analytics substacks.
- **Why:** Built from broadcast stills — direct evidence of what stylistic claims are defensible
  from broadcast view alone; its recommendations page is a further index of the ecosystem.
- **Key works:** *Talking Tactics recommendations page*; *Premier League tactics explainer series*.

### Dead Ball Analytics
- **URL:** https://deadballanalytics.substack.com
- **Focus:** xG-driven match/league analysis (EPL, La Liga, Ligue 1); conversion over/under-
  performance narratives.
- **Why:** Match narratives generated almost entirely from shot events plus xG — the minimum-data
  tier our pipeline always has even when geometry and ball track drop out. Fallback report template.
- **Key works:** *Beyond the Stats: How Expected Goals (xG) Transforms Football Strategy and Player
  Analysis*.

---

## 2. Tactical blogs & match-report sites

Video-native tactical vocabulary and the closest existing products to our target output.

### Spielverlagerung.com  *(co-founded by Rene Maric, now Bayern Head of Coach Development)*
- **URL:** https://spielverlagerung.com
- **Focus:** Deep tactical theory grounded in watching video, not proprietary data — half-spaces,
  zone 14, diagonality, pressing typologies (man-oriented vs zonal, pressing traps), breaking low
  blocks, juego de posicion.
- **Why:** Its vocabulary is video-native: half-space occupation and pressing triggers are
  observable on broadcast frames with usable geometry, so they give our ManU reports language that
  does not presuppose full-pitch tracking. The canonical qualitative pressing/positional taxonomy
  written by working coaches.
- **Key works:** *Tactical Theory series* (spielverlagerung.com/tactical-theory); *The Half-spaces*;
  Rene Maric on breaking down low blocks.

### Between the Posts
- **URL:** https://betweentheposts.net
- **Focus:** Data-informed tactical match reports for top leagues — pass networks, xG plots, prose
  narrative combined.
- **Why:** The closest existing product to our target output. Their pass networks come from event
  data only, proving the report format works without commercial tracking.
- **Key works:** *Tactical match report archive*; *Weekend preview series*.

### Get Goalside — Mark Thompson
- **URL:** https://www.getgoalsideanalytics.com
- **Focus:** Football-data industry essays, defensive metrics, a *Research in Focus* series
  translating academic work, plus explainers of industry method lineage (pitch control, EPV, style
  models) with data-requirement caveats.
- **Why:** Repeatedly examines what can and cannot be inferred from limited event data — directly
  useful for honest claims from spotted events without commercial tracking. Its defensive-metrics
  writing matters because defense is where censored tracking hurts most (off-ball players out of
  frame), and its method-history posts are a cheap map of which metrics genuinely need full tracking.
- **Key works:** *Research in Focus: Karun Singh's Expected Threat*; *Everything you need to know
  about pitch control*; *Get Goalside 100*.

### Total Football Analysis
- **URL:** https://totalfootballanalysis.com
- **Focus:** Coach/analyst-written tactical theory and how-to-coach guides — rest defence, rest
  attack, per-manager style profiles; 12,000+ articles since 2018.
- **Why:** Its "how to coach X" guides operationalize concepts like rest-defence structure (numbers
  behind the ball, +1 rule) into countable quantities — several computable from partial broadcast
  geometry when the in-possession back line is on-screen; the rest flag as needing full tracking.
- **Key works:** *How to Coach Rest Defence*; *Coaching Rest Attack Tactics*; *Head Coach Analysis
  series*.

### Training Ground Guru
- **URL:** https://trainingground.guru
- **Focus:** Trade press for club staff — profiles of heads of data and recruitment analysts, how
  PL analysis departments actually operate, conferences and case studies.
- **Why:** Ground truth on what a real ManU/EPL opposition scouting pack contains and who consumes
  it; the Brentford runs-in-behind case study shows which questions staffs actually put to data —
  keeps our reports aligned with practitioner formats and honest about where answers used tracking
  we lack.
- **Key works:** *Profiles section* (club data-staff interviews); *How clubs like Brentford turn
  data into an advantage*; podcast *Rene Maric — From Blogging to Bayern Munich*.

---

## 3. Coaches' writing & coach education

Plain-language, video-based operational definitions — no tracking dependency — that we translate
partial-frame observations into.

| Source | URL | Focus | Broadcast-computable hook | Key works |
|---|---|---|---|---|
| The Coaches' Voice / CV Academy | https://learning.coachesvoice.com | UEFA-licensed-coach match analyses & masterclasses; extensive ManU build-up breakdowns | **Direct ground truth for OUR team** — their ManU 4-2-3-1 double-pivot press / man-oriented high press / 4-4-1-1 mid-block claims are a natural validation target our pipeline should reproduce or refute | *ManU 4 Newcastle 1 (press structure)*; *Newcastle 1 ManU 0 (man-oriented high press)*; *ManU 1 Arsenal 1 (4-4-1-1 mid-block)* |
| Coach Notes | https://coachnotes.co.uk | Coach explainers of tactical concepts (rest defence, manager style profiles) | "Count ManU players behind the ball on trusted in-possession frames" — purely concept-based | *Understanding Rest Defence*; *Daniel Farke profile* |
| Running The Show | https://runningtheshowblog.wordpress.com | Rest-defence & counter-marking essays with numerical-advantage rules (3v2 behind ball) | Concrete countable rules estimable whenever the deep structure is in frame, with a visibility caveat | *Tactical Theory: Rest-Defence and Counter-Marking* |
| Jed C. Davies | https://www.goodreads.com/author/show/7550694.Jed_C_Davies | Coach-author codifying positional play / possession styles (tiki-taka, rondos, positional games) | Event-level pass data summarized against his zones/half-spaces/third-man vocabulary, no tracking | *Coaching the Tiki Taka Style of Play*; *The Philosophy of Football*; *Rondos & Positional Games* |
| Kieran Smith | http://coachkieransmith.blogspot.com | UEFA-licensed coach educator on juego de posicion | Turns partial-frame shape observations into coach-phrased structure claims; video-based | *Coach Kieran Smith blog*; positional-play courses/book with Jed Davies (2016) |

---

## 4. Team-style / public-data authors

Best-in-class narration of team style from public/event data — our report register.

### John Muller — *space space space* / futi
- **URL:** https://spacespacespaceletter.com (also johnspacemuller.substack.com)
- **Focus:** Team-style essays explaining what makes elite teams tick, built from event data and
  viz; now building **futi** to bring club-quality analytics to the public.
- **Why:** Best-in-class model for narrating team style from public/event data — exactly the
  register our grounded reports need. futi's model explainers document club-quality metrics rebuilt
  from accessible data, mirroring our constraint.
- **Key works:** *space space space archive*; *Get ready to meet futi*; *Why I left The Athletic*.

### Ian Graham — *How to Win the Premier League*
- **URL:** https://www.penguin.co.uk/books/462193/how-to-win-the-premier-league-by-graham-ian/9781804950302
- **Focus:** Insider account by Liverpool's Director of Research (2012-2023) — how possession-value
  models, opposition analysis and honest uncertainty handling actually drove decisions at a top EPL
  club.
- **Why:** Ground truth on what an EPL club data team considers a decision-grade stylistic claim vs
  noise — calibrates the ambition and honesty bar for our ManU pack (which claims survive coach
  scrutiny, which get laughed out of the room).
- **Key works:** *How to Win the Premier League: The Inside Story of Football's Data Revolution*
  (2024).

---

## 5. Club & federation data teams

How practitioner teams turn data into named tactical concepts a scout uses.

| Source | URL | Focus | Why (broadcast/ManU) | Key works |
|---|---|---|---|---|
| DFB-Akademie (German FA) | https://www.dfb-akademie.de | Coach-education academy; joint DFL/Sportec research on automated detection of tactical patterns (counterpressing, corner-defence roles) | Defines coach-recognizable patterns in machine-detectable terms — the vocabulary a ManU report should speak. **FLAG: full-tracking** — patterns estimable only on 26.4% usable-geometry frames or as event proxies | *Detection of tactical patterns using semi-supervised GNNs* (Sloan 2022); *Individual role classification for players defending corners* (JQAS 2022) |
| DFB data team — Bauer & Anzer | https://journals.sagepub.com/doi/10.3233/JSA-220620 | Federation data-science: counterpressing detection, expected passes, shot xG, formation-in-context | Best public example of turning tracking into scout vocabulary. **FLAG: full-tracking** — reuse labels/feature definitions, re-derive estimators from events + partial geometry | *Data-driven detection of counterpressing* (DMKD 2021); *Expected passes* (DMKD 2022); *Putting team formations into context* (2023, w/ Shaw) |
| Google DeepMind x Liverpool FC | https://deepmind.google/blog/tacticai-ai-assistant-for-football-tactics/ | Geometric deep learning on corner kicks (TacticAI, Nature Comms 2024); agenda paper *Game Plan* | Flagship club-lab open publication; TacticAI works from broadcast-style visual data, *Game Plan* frames analytics under partial observability. Set-piece modelling is high-signal, low-tracking-need for a ManU pack | *TacticAI* (Nature Communications 2024); *Game Plan* (JAIR 2021, arXiv:2011.09192) |
| Analytics FC | https://analyticsfc.co.uk/blog/ | Consultancy blog around TransferLab — predictive player metrics, recruitment case studies | Shows how consultancies package event-data-derived style/player metrics into client-facing deliverables — a commercial benchmark for our pack; runs on event data (100+ leagues) | *TransferLab methodology pages*; *TransferLab Emerge* |

---

## 6. Academic labs & research groups

The papers behind the genre.

### KU Leuven — DTAI Sports Analytics Lab (Jesse Davis group)  *(merged: 3 listings)*
- **URL:** https://dtai.cs.kuleuven.be/sports/
- **Focus:** ML on soccer event streams — action valuation (VAEP, Atomic-VAEP), playing-style
  decomposition (SoccerMix), player vectors, pass-decision models (un-xPass); open-source
  `socceraction` toolkit.
- **Why:** Their entire framework runs on **event data only** — VAEP/xT-style valuation and
  SoccerMix style clustering are directly buildable from our spotted passes/shots/goals with
  lineup-prior identity; `socceraction` is drop-in code. Their explainability work models how to
  caveat claims, which our validated-or-nothing rule requires.
- **Key works:** *Actions Speak Louder than Goals* (VAEP, KDD 2019); *SoccerMix* (ECML PKDD 2020);
  *Exploring VAEP* (interactive blog); `socceraction` / `un-xPass`.

### DTAI Sports Analytics Lab (Jesse Davis) — see above (merged).

### German Sport University Cologne — Institute of Exercise Training & Sport Informatics (Daniel Memmert group)  *(merged: 2 listings)*
- **URL:** https://www.dshs-koeln.de/en/institute-of-exercise-training-and-sport-informatics/
- **Focus:** Big-data pattern identification, formation classification, tactical creativity, space
  control; the SOCCER analysis tool; DFL-funded *Position data in elite soccer*; runs the Match
  Analysis MSc.
- **Why:** The academic home of formation/pressing/space-control pattern recognition German coach
  education draws on; their formation-clustering is what our lineup-prior identity assignment feeds
  into. **FLAG: full-tracking** — adapt or flag when applied to 26.4% usable geometry.
- **Key works:** *Does 4-4-2 exist?* (arXiv:1910.00412); *Data Analytics in Football* (Routledge,
  Memmert & Raabe); SOCCER tool.

### DTAI, Univ. of Tübingen — Data-Driven Tactical Performance Analysis
- **URL:** https://uni-tuebingen.de/en/... /data-driven-tactical-performance-analysis-in-football/
- **Focus:** ML quantification of tactical behaviour from synchronized positional + event data
  (DFB/DFL collaboration).
- **Why:** Bridges practitioner expertise and data science — defensible definitions of offensive-
  performance quantification and pattern detection to cite. **FLAG:** positional methods presume
  full tracking; event components transfer directly.
- **Key works:** research programme page; survey *Data analytics in the football industry*
  (Sci & Med in Football 2024).

### Forcher et al. — KIT Karlsruhe defensive-play tracking research
- **URL:** https://journals.sagepub.com/doi/abs/10.1177/17479541221075734
- **Focus:** Scoping reviews on analyzing defensive play with player tracking data.
- **Why:** The definitive map of which defensive-style constructs (compactness, pressing intensity,
  line behaviour) have been quantified — a menu for our ManU out-of-possession profile. **FLAG:
  full-tracking** — each construct needs a censored-broadcast adaptation or "not computable" label.
- **Key works:** *The use of player tracking data to analyze defensive play — a scoping review*
  (IJSSC 2022); *Integrating physical and tactical factors using positional data* (PMC9671036).

### Keisuke Fujii — Sports Behavior Group, Nagoya University
- **URL:** https://takedalab.g.sp.m.is.nagoya-u.ac.jp/groups/sports-behavior-group
- **Focus:** Trajectory prediction/imputation, off-ball evaluation (C-OBSO), RL tactical analysis,
  open-source OpenSTARLab — explicitly works on broadcast-estimated data.
- **Why:** One of few labs that builds valuation metrics designed to run on incomplete,
  video-estimated tracking rather than vendor feeds — trajectory-prediction-as-baseline is a
  template for scoring ManU off-ball style from partial views.
- **Key works:** *Evaluation of Creating Scoring Opportunities via Trajectory Prediction* (MLSA
  2022); *OpenSTARLab*.

### Weidi Xie group, Shanghai Jiao Tong University  *(note: the flagship Chinese soccer-video lab is SJTU, not Tsinghua)*
- **URL:** https://arxiv.org/abs/2412.01820
- **Focus:** Soccer broadcast video understanding — MatchVision encoder, SoccerReplay-1988 dataset,
  automatic commentary (MatchTime).
- **Why:** SOTA in extracting events and semantics straight from broadcast video — an
  alternative/upgrade path for our spotting models; their commentary generation parallels our
  grounded pundit-report generation.
- **Key works:** *Towards Universal Soccer Video Understanding* (CVPR 2025); *MatchTime* (EMNLP
  2024); *SoccerReplay-1988*.

### TU Munich — Chair of Performance Analysis & Sports Informatics (Daniel Link, Martin Lames)
- **URL:** https://www.hs.mh.tum.de/en/trainingswissenschaft/home/
- **Focus:** Spatiotemporal performance analysis, the Dangerousity metric, optical-tracking
  validity studies.
- **Why:** Their TRACAB validity study is *the* published accuracy benchmark — the right citation
  when we report honest error bars on broadcast-derived positions; Dangerousity is an
  event+position metric adaptable to partial data.
- **Key works:** *Real-time quantification of dangerousity* (PLOS ONE 2016); *Football-specific
  validity of TRACAB's optical tracking* (PLOS ONE 2020).

### Ulf Brefeld — Machine Learning group, Leuphana University
- **URL:** https://ml3.leuphana.de/
- **Focus:** Probabilistic movement models, semi-supervised tactical pattern detection (recurring
  co-author of Anzer/Bauer/Fassmeyer).
- **Why:** The ML engine behind the DFB pattern papers; his weak/semi-supervised approaches matter
  because we can only hand-label a few ManU pattern examples. **FLAG:** published models use full
  tracking.
- **Key works:** *Detection of tactical patterns using semi-supervised GNNs* (Sloan 2022);
  *Probabilistic movement models and zones of control* (MLJ 2019).

### SoccerNet consortium (Univ. Liège — Van Droogenbroeck/Cioppa + KAUST — Ghanem/Giancola)  *(incl. sn-gamestate / TrackLab)*
- **URL:** https://www.soccer-net.org/
- **Focus:** Broadcast-video benchmarks — action spotting, camera calibration, tracking/re-ID,
  jersey numbers, game state reconstruction (GS-HOTA). Open data, baselines, annual challenge
  reports; the `sn-gamestate` / `tracklab` reference code.
- **Why:** **The single most relevant academic source** — it defines and benchmarks our exact
  regime (censored broadcast, spotting events, GSR scored by GS-HOTA, the metric this repo
  optimizes). The 200x30s dataset (9.37M pitch-line points, 2.36M positions) is our legal
  calibration/validation ground truth; TrackLab is the architecture to diff ours against.
- **Key works:** *SoccerNet Game State Reconstruction* (CVPRW 2024, arXiv:2404.11335);
  *SoccerNet-v2* (CVPRW 2021); *SoccerNet-Tracking* (CVPRW 2022); github.com/SoccerNet/sn-gamestate.

### Friends of Tracking  *(merged: 3 listings — practitioner education collective)*
- **URL:** https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w
- **Focus:** Tutorials by club analysts (Barcelona, Benfica, DFB, Hammarby, English FA) on tracking
  and event data — pitch control, physical metrics, valuing actions — with open code.
- **Why:** The canonical reference implementations for pitch control and tracking metrics, and the
  clearest statement of their data requirements: pitch control needs all 22 players, so with ~11.8
  visible anything derived from these tutorials must be flagged partial-pitch or skipped. Uses the
  Metrica open sample — useful for validating our geometry code.
- **Key works:** *LaurieOnTracking* tutorial series; Friends-of-Tracking YouTube course (2020);
  *SoccermaticsForPython*; Metrica sample-data.

### Soccermatics — David Sumpter
- **URL:** https://soccermatics.readthedocs.io/
- **Focus:** Open course from Python basics through xG, player evaluation, tracking data, possession
  value; companion Medium essays.
- **Why:** Runnable reference implementations for every metric tier we might ship; each chapter
  states its input data, letting us map methods onto our event-only vs geometry-available vs
  ball-track-available regimes.
- **Key works:** *Soccermatics course* (readthedocs); *Explaining Expected Threat* (Medium).

### American Soccer Analysis
- **URL:** https://www.americansocceranalysis.com
- **Focus:** Community site with fully published xG and goals-added (g+) methodologies for MLS.
- **Why:** The most transparent open methodology writing anywhere — a template for our own model
  cards with error bars. **Flag:** g+ assumes complete event feeds; on censored events it must be
  re-derived or marked unavailable.
- **Key works:** *Expected Goals 3.0 Methodology*; *What are Expected Goals (xG)?*.

---

## 7. Key recurring authors (methods)

The names that recur across the labs and conferences above; individual profiles worth tracking.

| Author | URL | What they own | Broadcast verdict |
|---|---|---|---|
| Jan Van Haaren | https://www.janvanhaaren.be/ | Annual *Soccer Analytics Review* surveys; VAEP co-author; Club Brugge data science | **Fastest way to stay exhaustive** — reviews index broadcast-tracking / event-only / full-tracking methods separately, so we tag candidates by regime before adopting |
| Karun Singh | https://karun.in/blog/expected-threat.html | Origin of Expected Threat (xT) — reproducible Markov-chain possession value from event data | Implementable on our pass events, but only on trusted-geometry frames (xT needs pass start/end coords) — flag the dependency |
| Tom Decroos | https://tomdecroos.github.io/ | SPADL event representation, VAEP, SoccerMix | Canonical blueprint for turning sparse event streams into team/player style fingerprints — closest template for a ManU profile from censored events |
| Pieter Robberechts & Maaike Van Roy (KU Leuven) | https://dtai.cs.kuleuven.be/sports/publications | Pass success/selection, set-piece value, in-match win probability, explainable EVs | Event-data set-piece & pass-tendency models; *Why Would I Trust Your Numbers?* matches our validated-or-nothing discipline |
| Gabriel Anzer | https://www.semanticscholar.org/paper/A-Goal-Scoring-Probability-Model...  | xG/xPass from synchronized position+event data, event-tracking sync, goal-origin clustering | **FLAG: full-tracking (TRACAB)** — use for method calibration and synchronization ideas, not on 26.4% geometry |
| Pascal Bauer (DFB) | https://journals.sagepub.com/doi/10.3233/JSA-220620 | Automatic tactical-pattern detection — counterpressing, overlapping runs, contextual formation | Defines the tactical vocabulary a ManU report needs. **FLAG:** Bundesliga tracking — borrow taxonomy & weak-supervision, not raw models |
| Laurie Shaw | https://www.sloansportsconference.com/people/laurie-shaw | Formation/phase detection, corner-kick playbooks, open pitch-control code | Dynamic formation & set-piece routine clustering are exactly the artifacts a pack needs. **FLAG: full 22-player tracking** — needs imputation literature first |
| Javier Fernández | https://arxiv.org/abs/2011.09426 | Space creation/occupation, deep-learning EPV, SoccerMap (ex-FC Barcelona) | Defines modern space/possession-value language. **FLAG: full-tracking** — SoccerMap's coarse-grid surface degrades more gracefully to partial views than point models |
| Luke Bornn | http://www.lukebornn.com/ | Spatiotemporal sports stats; co-founder Zelus Analytics | Recurring senior author behind Wide Open Spaces/EPV; papers page is a free index of the tracking canon — the ceiling to report honestly against |
| William Spearman (Liverpool FC) | https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals | Physics-based pitch control, off-ball scoring opportunity (OBSO) | Standard "where does this team generate danger" machinery. **FLAG: all 22 + ball** — computable only on the 26.4% trusted-frame subset with caveats |
| Patrick Lucey (Stats Perform) | https://www.patricklucey.com/ | 100+ papers on multi-agent trajectory modelling, play retrieval, trajectory imputation ("ghosting"), AutoStats | **The industry line on inferring full-pitch context from partial broadcast** — trajectory imputation is the upgrade path for our 26.4%-geometry problem; AutoStats proves broadcast-only skeleton tracking is production-viable |
| Hyunsung Kim (KAIST / Fitogether) | https://sites.google.com/view/hyunsungkim/research | Ball-trajectory inference from player context, multi-agent imputation, event-tracking sync without annotated locations | **Attacks our two worst gaps** — inferring the ball when the track is unusable (we're at 32-52%) and imputing off-screen players | 
| Jonas Theiner & Ralph Ewerth (LUH / TIB) | https://mm4spa.github.io/tvcalib/ | Camera calibration & field registration from single broadcast frames | TVCalib directly addresses why only 26.4% of our broadcast has usable geometry and how to raise it / quantify per-frame confidence |

Key works: Kim — *Ball Trajectory Inference ... Set Transformer & Hierarchical Bi-LSTM* (KDD 2023),
*MIDAS* (arXiv:2408.10878), *ELASTIC*. Theiner — *TVCalib* (WACV 2023). Lucey — AutoStats;
trajectory-imputation / play-retrieval papers. Shaw — *Routine Inspection: A Playbook for Corner
Kicks* (SSAC 2021, 1st place), *Dynamic Analysis of Team Strategy* (arXiv 2019). Fernández — *Wide
Open Spaces* (SSAC 2018), *SoccerMap* (ECML/PKDD 2020). Spearman — *Beyond Expected Goals* (SSAC
2018). Anzer — *Goal Scoring Probability Model* (Frontiers 2021), *Expected Passes* (DMKD 2022).

---

## 8. Conferences & workshops

Where the papers land — the recurring sweep venues.

| Venue | URL | Focus | Broadcast relevance | Anchor papers |
|---|---|---|---|---|
| **SoccerNet Challenges** (annual results) | https://arxiv.org/abs/2508.19182 | GSR, team ball-action spotting, tracking, calibration, jersey numbers — winning methods summarized | **Densest index of what beats our current components, task by task, on broadcast-only input** | *SoccerNet 2025 Results* (arXiv:2508.19182); *2024 Results* (arXiv:2409.10587) |
| CVSports (CVPR workshop) | https://vap.aau.dk/cvsports/ | Main CVPR sports-CV venue; hosts SoccerNet tracks — spotting, GSR, calibration, jersey, 3D from replays | Every method for our moving-single-camera / partial-visibility constraint debuts here | *SoccerNet-v3D* (CVPRW 2025, arXiv:2504.10106); *Uncertainty-Aware Jersey Number Recognition* (CVPRW 2025) |
| MMSports (ACM MM) | http://mmsports.multimedia-computing.de/mmsports2025/ | Broadcast augmentation, player re-ID, jersey recognition from low-res broadcast | Where **PRTreID** (already in our pipeline) was published; low-res-broadcast tricks matching our 26.4% reality | *MMSports 2025 proceedings*; *Jersey Number Recognition from Low-Resolution Broadcast* (2023) |
| MIT Sloan Sports Analytics Conference | https://www.sloansportsconference.com/ | Flagship applied venue; pitch control, EPV, corner playbooks, tactical GNNs debuted here | **FLAG:** nearly all soccer tracking papers assume full commercial tracking — usable for concepts and for defining what we *cannot* honestly compute | *Beyond Expected Goals* (2018); *Wide Open Spaces* (2018); *Routine Inspection* (2021) |
| MLSA (ECML/PKDD) | https://dtai.cs.kuleuven.be/events/MLSA26/ | Annual ML-for-sports workshop (12+ editions, Springer); soccer-heavy, KU Leuven-run | The venue where **partial-data and data-acquisition problems like ours are publishable, not embarrassments** — mineable index of event-only + imputation methods each year | *MLSA proceedings 2016-2025*; pass-receiver prediction challenge papers |
| Hudl StatsBomb Conference (research papers) | https://blogarchive.statsbomb.com/news/statsbomb-conference-2024-research-papers/ | Team play-style modelling, tactical pattern extraction, pressing, build-up classification on event + 360 data | Several papers derive style from events + partial (visible-player) 360 data rather than full tracking — **directly transferable**; *Modelling Team Play Style Using Tracking Data* (2024) needs full tracking, flag as inspiration only | *An Events and 360 Data-Driven Approach ...* (2023); *RisingBALLER: A Player is a Token* (2024) |

---

## 9. Commercial vendors (methodology transfer)

The industry proof that censored broadcast geometry supports serious analysis. Copy the taxonomy
and reporting discipline; the estimation engines are proprietary and mostly full-tracking.

### SkillCorner — methodology & product
- **URL:** https://skillcorner.com/products/football/xy-tracking-data
- **Focus:** Broadcast-video tracking at scale — homography/line detection, re-ID, deep-learning
  extrapolation of off-camera players to a continuous 22-player feed.
- **Why:** **The industry benchmark for exactly our input regime** (single broadcast camera,
  players off-screen). Their explicit separation of *observed* vs *extrapolated* positions is the
  template for how we should report trusted-frame vs imputed geometry.
- **Key works:** *XY Tracking Data* methodology page (off-camera extrapolation); *Data On Demand*;
  *v3 pipeline notes*.

### StatsBomb 360 — freeze-frame methodology  *(merged: 2 listings)*
- **URL:** https://blogarchive.statsbomb.com/news/statsbomb-data-case-studies-freeze-frames-and-defender-locations/
- **Focus:** Event data enriched with locations of only the players visible in the broadcast frame
  at each event — "camera-censored" snapshots — plus analysis patterns built on them (defender
  locations at shots, line-breaking passes).
- **Why:** **Commercial proof that censored broadcast geometry (visible players only) supports
  serious opposition analysis** — the exact regime our pipeline produces (~11.8/22 visible). Their
  case studies show which style metrics survive partial visibility. **Flag:** 360 itself is a
  commercial product; only the methodology transfers.
- **Key works:** *Introducing StatsBomb 360*; *Freeze Frames and Defender Locations*; xG model
  explainers.

### Hudl StatsBomb — company research blog
- **URL:** https://statsbomb.com/articles/
- **Focus:** xG model detail, shot freeze frames, 360 framing, team/league analyses.
- **Why:** Freeze frames and 360 are camera-derived positional snapshots of visible players at
  events — the same censored geometry our pipeline produces; their articles show what that partial
  view supports.
- **Key works:** *StatsBomb xG model explainers*; *StatsBomb Originals* interview series.

### Stats Perform / Opta Vision & Playing Styles
- **URL:** https://www.statsperform.com/products/opta-vision/
- **Focus:** AI-enriched fusion of event and tracking data; a named **Playing Styles** taxonomy
  (2025-26 launch), off-ball runs, predictive metrics.
- **Why:** Their Playing Styles taxonomy is the **industry-standard vocabulary a ManU pack will be
  compared against** — worth mirroring its categories from our event side. **Flag:** Opta Vision
  fuses full commercial tracking; copy the taxonomy, not the estimation method.
- **Key works:** *Opta Vision product page*; *Playing Style Interactions*; *20 Different Ways to
  Talk About Football Using OptaAI in 2025*.

---

## 10. Institutes & governing-body education

Defines the professional deliverable and the coach-endorsed metric dictionary.

| Institute | URL | Focus | Why (broadcast/ManU) | Key works |
|---|---|---|---|---|
| FIFA Training Centre / Technical Study Group | https://www.fifatrainingcentre.com | Enhanced Football Intelligence (EFI) metrics with operational definitions; the FIFA Football Language phase/behaviour taxonomy; TSG reports | **Closest thing to an official coach-endorsed metric dictionary** (line height, defensive line breaks, pressure on ball, phases) — adopt EFI-style definitions and state per metric whether it is event-, partial-geometry- or full-tracking-computable. The France PMSR ground-truth PDFs in this repo come from this ecosystem | *The FIFA Football Language*; *EFI at World Cup 2022*; *Are EFI data valuable?* (PMC10765435) |
| PFSA (Professional Football Scouts Association) | https://thepfsa.co.uk | UK body running Performance / Opposition Analysis / Technical Scouting courses | **Defines the professional deliverable format of an opposition scouting pack** — structure, detail, "digestible messages for a manager" — a checklist for what our demo artifact must contain; video-first, no tracking dependency | *PFSA Level 2 Performance Analysis*; *Opposition Analysis Level 1/2* |
| The Coaches' Voice / CV Academy | https://learning.coachesvoice.com | *(also in §3)* ManU match tactical analyses | Direct qualitative ground truth / validation target for our team | see §3 |
| Barca Innovation Hub | https://barcainnovationhub.fcbarcelona.com | Advanced Tactical Analysis certificate; possession/positional-play codification ("the Barca way"); Metrica collaboration | Club-institute view of how a possession identity is codified and taught, plus a formal curriculum. **FLAG:** much is built on Metrica full-pitch tracking — only the conceptual frameworks and video coursework transfer | *Certificate in Advanced Football Tactical Analysis*; *How our new research helped unlock the Barca way* |
| UEFA Research Grant Programme | https://uefaacademy.com/courses/rgp/ | UEFA-funded academic research with national associations; up to EUR 15-20k / 9-month project | Curated archive of practitioner-endorsed research questions (each needs an association recommendation) — grounds which stylistic metrics coaches actually care about. Methods vary per project; check each for full-tracking dependence | *RGP annual selected-projects announcements*; *2024/25 research grant call* |

---

## 11. Datasets, open code & bridge papers

Free ground truth and the specific method upgrades for our worst gaps.

| Resource | URL | What it gives us |
|---|---|---|
| **SkillCorner Open Data** | https://github.com/SkillCorner/opendata | 10 full matches of real broadcast tracking with **visibility/extrapolation flags** — free ground truth for what commercial-grade censored tracking looks like; benchmark our 26.4% / ~11.8-visible rates against theirs and test whether our style metrics compute on flagged data. Loadable via kloppy |
| **Continuous football player tracking from discrete broadcast data** (R. Soc. Open Sci. 2025) | https://royalsocietypublishing.org/rsos/article/12/10/251175/236076/ | Peer-reviewed method for **precisely our upgrade path** — reconstructing continuous 22-player trajectories from discrete, event-anchored visible-player freeze frames. The bridge between our 26.4% geometry and full-tracking metrics |
| **From Broadcast to Minimap** (Golovkin & Kovalenko, Constructor Tech) | https://arxiv.org/abs/2504.06357 | SOTA end-to-end SoccerNet GSR — fine-tuned detection, SegFormer camera-param estimation, re-id + orientation + jersey voting; 1st place GSR 2024. **Component-by-component upgrade menu** for pushing GS-HOTA past 22.85 |
| **SoccerNet GSR / sn-gamestate / TrackLab** (Somers et al.) | https://arxiv.org/abs/2404.11335 | Formalizes our exact task (broadcast to 2D minimap with identity, scored by GS-HOTA) — **this IS our problem statement and metric**; TrackLab is the reference architecture to diff ours against |
| **PnLCalib** (Gutiérrez-Pérez & Agudo, IRI Barcelona) | https://arxiv.org/abs/2404.08401 | Camera calibration via keypoints + field-line refinement; SOTA on SoccerNet-Calibration. **Directly attacks the 26.4%-usable-geometry bottleneck** — line refinement over sparse keypoints raises the fraction of frames with trusted homography, multiplying downstream coverage |
| **PRTreID** (Mansourian, Somers et al.) | https://arxiv.org/abs/2401.09942 | Part-based multi-task re-ID + team affiliation + role classification — **already in our pipeline** (broke same-kit wall, GS-HOTA 19.83 -> 22.85). Anchor citation; BPBreID lineage is where further re-id gains live |
| **T-DEED** (Xarles et al.) | https://openaccess.thecvf.com/content/CVPR2024W/CVsports/ | Encoder-decoder for precise frame-level event spotting; won Ball Action Spotting 2024, backbone of 2025 winners. **Reference architecture** for the pass/action spotting our stylistic analysis depends on |
| **int8.io** (Kamil Czarnogórski) | https://int8.io/team-ball-action-spotting-challenge-2025 | Full engineering writeup of the challenge-winning Team Ball Action Spotting system (60.03 Team-mAP@1, modified T-DEED) — **rare reusable recipe** for improving our team-attributed pass/shot spotting |
| **WASB-SBDT** (NTT Communications) | https://github.com/nttcom/WASB-SBDT | Strong baseline for sports ball detection/tracking with temporal-consistency inference — **the most promising published lever to raise our 32-52% usable ball coverage** |
| **GTATrack** (SoccerTrack 2025 winner) | https://arxiv.org/abs/2602.00484 | Deep-EIoU + Global Tracklet Association — **the mechanism behind our recent GS-HOTA gain**, best-practice relinking recipe; Deep-EIoU handles erratic broadcast-camera motion that breaks Kalman trackers |
| **McByte / No Train Yet Gain** (Stanczyk & Bremond, INRIA) | https://arxiv.org/abs/2506.01373 | Training-free MOT using temporally propagated segmentation masks as an association cue; strong on SoccerNet-tracking — zero-training association upgrade candidate |

---

## Shortlist — 8 sources to monitor continuously

Ranked by ongoing signal-per-effort for a censored-broadcast ManU style pipeline.

| # | Source | Cadence | Why it stays on the list |
|---|---|---|---|
| 1 | **SoccerNet Challenges results** (arXiv, annual) | Yearly (+ challenge season) | Our exact regime and metric (GS-HOTA); densest task-by-task index of what beats our components on broadcast-only input |
| 2 | **Jan Van Haaren — Soccer Analytics Review** | Yearly | The single fastest exhaustive sweep; tags every method by data regime so we filter for event-only vs full-tracking without re-running this search |
| 3 | **KU Leuven DTAI Sports Analytics Lab** | Continuous (publications feed) | Their event-only frameworks (VAEP, SoccerMix, un-xPass, socceraction) are the ones we can actually build on spotted events |
| 4 | **The xG Football Club** (Alex Marin Felices) | Per-post | The anchor genre — pre-digests academic papers and flags their data assumptions; our narration register |
| 5 | **CVSports + MMSports proceedings** | Yearly (CVPR / ACM MM) | Where every partial-visibility, low-res-broadcast CV method (calibration, re-ID, spotting) debuts; PRTreID and T-DEED came from here |
| 6 | **SkillCorner** (open data + methodology blog) | Continuous | The commercial benchmark for our input regime and the observed-vs-extrapolated reporting template; free flagged ground truth |
| 7 | **The Coaches' Voice — ManU analyses** | Per-match | Direct qualitative ground truth for OUR team; validation targets our pipeline should reproduce or refute |
| 8 | **Hyunsung Kim + Keisuke Fujii** (imputation / broadcast-estimated valuation) | Continuous (Scholar) | The two research lines aimed squarely at our worst gaps — ball inference (32-52%) and off-screen player imputation (~11.8/22 visible) |
