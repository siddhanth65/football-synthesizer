# BAS pass/drive validation (B-3 stage 1)

Operating point (frozen on brighton_manutd h1): PASS confidence >= 0.4, same-class peaks merged within 1 s.
Replay filter (pre-committed hypothesis): keep peaks in live_wide segments (gap-merge 2 s, pad 1 s).

`raw` = every BAS peak. `live` = replay-filter arm (+dedup). `op` = confidence+dedup arm. `ratio_*` vs Sofascore attempted; a half's ratio is shown only when BAS covers >= 90% of the half's frame span (else PARTIAL).

## brighton_manutd

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 729 | 428 | 538 | 543 | 1.343 | 0.788 | 0.991 |
| h2 | 654 | 384 | 443 | 445 | 1.470 | 0.863 | 0.996 |
| **match** | 1383 | 812 | 981 | 988 | 1.400 | 0.822 | 0.993 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 667 | 420 | 424 |
| h2 | 600 | 410 | 343 |
| **match** | 1267 | 830 | 767 |

## manutd_liverpool

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 698 | 343 | 510 | 524 | 1.332 | 0.655 | 0.973 |
| h2 | 668 | 343 | 471 | 447 | 1.494 | 0.767 | 1.054 |
| **match** | 1366 | 686 | 981 | 971 | 1.407 | 0.706 | 1.010 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 621 | 326 | 397 |
| h2 | 571 | 328 | 361 |
| **match** | 1192 | 654 | 758 |

## manutd_fulham

Coverage -- h1: 5/5 chunks (100% of frames), h2: 5/5 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 694 | 363 | 500 | 485 | 1.431 | 0.748 | 1.031 |
| h2 | 661 | 304 | 445 | 381 | 1.735 | 0.798 | 1.168 |
| **match** | 1355 | 667 | 945 | 866 | 1.565 | 0.770 | 1.091 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 636 | 390 | 392 |
| h2 | 541 | 286 | 286 |
| **match** | 1177 | 676 | 678 |

## palace_manutd

Coverage -- h1: 6/6 chunks (100% of frames), h2: 5/6 chunks (99% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 639 | 278 | 447 | 467 | 1.368 | 0.595 | 0.957 |
| h2 | 721 | 301 | 518 | 485 | 1.487 | 0.621 | 1.068 |
| **match** | 1360 | 579 | 965 | 952 | 1.429 | 0.608 | 1.014 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 532 | 273 | 345 |
| h2 | 625 | 284 | 374 |
| **match** | 1157 | 557 | 719 |

## manutd_tottenham

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 743 | 343 | 541 | 526 | 1.413 | 0.652 | 1.029 |
| h2 | 746 | 334 | 538 | 505 | 1.477 | 0.661 | 1.065 |
| **match** | 1489 | 677 | 1079 | 1031 | 1.444 | 0.657 | 1.047 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 657 | 328 | 416 |
| h2 | 622 | 326 | 398 |
| **match** | 1279 | 654 | 814 |

## southampton_manutd

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 706 | 436 | 520 | 492 | 1.435 | 0.886 | 1.057 |
| h2 | 756 | 473 | 567 | 599 | 1.262 | 0.790 | 0.947 |
| **match** | 1462 | 909 | 1087 | 1091 | 1.340 | 0.833 | 0.996 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 630 | 426 | 411 |
| h2 | 676 | 465 | 451 |
| **match** | 1306 | 891 | 862 |
