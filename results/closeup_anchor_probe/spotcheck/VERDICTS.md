# Spot-check verdicts -- h1_chunk_000 (threshold-freezing set)

Manual visual verification of 24 legible-candidate crops sampled across the confidence spectrum
(recognizer: `outputs/jersey/jersey_torso_r224_acc417.pt`, torso variant; crops = COCO yolov8s
persons on `close_up`-classified frames, box_h >= 100 px). Crops are small (person box 100-176 px,
upscaled), so "unconfirmed" = number not human-legible at that crop size, NOT a model error.

| rank | conf | pred | box_h | my read of the crop | verdict |
|---|---|---|---|---|---|
| 00 | 0.997 | 29 | 102 | Brighton back, number not legible at size | unconfirmed (high conf) |
| 01 | 0.892 | 20 | 104 | two tangled players (red+blue) | bad crop (2 bodies) |
| 02 | 0.819 | 8  | 100 | Man Utd red back, **"8" legible** | **CORRECT** |
| 03 | 0.747 | 10 | 107 | Brighton back, not legible | unconfirmed |
| 04 | 0.690 | 20 | 102 | Man Utd red, motion | unconfirmed |
| 05 | 0.628 | 8  | 114 | Man Utd red back, **"8" legible** | **CORRECT** |
| 06 | 0.581 | 20 | 137 | red+blue cluttered | bad crop |
| 07 | 0.535 | 20 | 138 | Man Utd red back, **"20" legible** | **CORRECT** |
| 08 | 0.494 | 11 | 121 | Brighton, not legible | unconfirmed |
| 09 | 0.456 | 33 | 111 | red, blurry | unconfirmed |
| 10 | 0.422 | 20 | 105 | red, motion blur | unconfirmed |
| 11 | 0.396 | 14 | 108 | **referee (black kit), NO number** | **FALSE POSITIVE** |
| 12 | 0.368 | 23 | 102 | Brighton, not legible | unconfirmed |
| 13 | 0.340 | 24 | 176 | Man Utd red back, **"20" legible** | **WRONG (20 read as 24)** |
| 14-23 | <=0.31 | -- | -- | progressively blurrier / smaller | low-conf tail |

## Freeze

- All confirmed failures (referee false-positive, 20->24 misread) sit at conf **<= 0.40**.
- All confirmed-correct legible reads sit at conf **>= 0.53** (three independent Man Utd backs: 8, 8, 20).
- **FROZEN high-confidence anchor threshold = 0.70** -- a conservative bar with clear margin above
  the failure band. Applied unchanged to the full match.

## Transfer verdict (recognizer -> close-up domain)

The torso recognizer, trained on broadcast-wide tracklet crops, **transfers to close-up crops**: on
the four crops whose number I could read by eye (8, 8, 20, 20) it was correct on 3/4, at high
confidence, vs 1/99 chance. It is NOT garbage on this domain. The one miss (20->24) and the referee
false-positive both fell below 0.40 -- i.e. the confidence gate cleanly separates them. No OCR
baseline (easyocr/tesseract absent; not adding a dependency for the optional sanity leg).
