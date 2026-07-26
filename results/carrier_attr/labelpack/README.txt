Carrier labelling pack. For each id: <id>_crop.jpg is the carrier crop at the kick moment; <id>_ctx-2..+2.jpg are full frames around it with the carrier boxed in yellow.
Fill labels.csv: player_name (exact Sofascore spelling, or UNKNOWN / WRONG_PLAYER_BOXED), confidence_1to3, notes.
Model predictions are deliberately NOT included.

EASIEST WAY: double-click label.html (offline, no server) and label with the roster buttons; it exports labels_filled.csv, which tools/score_carrier_labels.py reads directly.
