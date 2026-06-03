# Zero-rate analysis — real vs `run3-CoT-no0`

- real rows: 6,453
- synth rows: 28,120
- steps30pre bins: `['zero', 'very_low', 'low', 'med', 'high', 'very_high']` (edges=[0, 1, 50, 200, 500, 1500, inf])
- min_n per cell: 20

## T1 — marginal zero rate (sanity check)

| avail | n_real | nz_real | zr_real | n_synth | nz_synth | zr_synth | delta_zr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 809 | 284 | 0.351 | 3917 | 129 | 0.033 | -0.318 |
| True | 5644 | 3199 | 0.567 | 24203 | 1291 | 0.053 | -0.514 |

## T2 — zero rate × slot × avail

| slot | avail | n_real | nz_real | zr_real | n_synth | nz_synth | zr_synth | delta_zr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | False | 73 | 28 | 0.384 | 618 | 85 | 0.138 | -0.246 |
| 1 | True | 1224 | 881 | 0.72 | 5006 | 770 | 0.154 | -0.566 |
| 2 | False | 178 | 51 | 0.287 | 775 | 1 | 0.001 | -0.286 |
| 2 | True | 1115 | 574 | 0.515 | 4849 | 122 | 0.025 | -0.49 |
| 3 | False | 157 | 51 | 0.325 | 751 | 20 | 0.027 | -0.298 |
| 3 | True | 1160 | 616 | 0.531 | 4873 | 136 | 0.028 | -0.503 |
| 4 | False | 247 | 101 | 0.409 | 1008 | 16 | 0.016 | -0.393 |
| 4 | True | 1032 | 534 | 0.517 | 4616 | 150 | 0.032 | -0.485 |
| 5 | False | 154 | 53 | 0.344 | 765 | 7 | 0.009 | -0.335 |
| 5 | True | 1113 | 594 | 0.534 | 4859 | 113 | 0.023 | -0.511 |

## T3 — zero rate × steps30pre bin × avail

Hurdle test: if bin=`zero` has near-100% zero in real but synth doesn't, the LLM is ignoring the 'no recent activity' signal.

| s30_bin | avail | n_real | nz_real | zr_real | n_synth | nz_synth | zr_synth | delta_zr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | False | 172 | 45 | 0.262 | 349 | 7 | 0.02 | -0.242 |
| high | True | 544 | 185 | 0.34 | 2220 | 48 | 0.022 | -0.318 |
| low | False | 157 | 56 | 0.357 | 1027 | 16 | 0.016 | -0.341 |
| low | True | 1323 | 657 | 0.497 | 6173 | 134 | 0.022 | -0.475 |
| med | False | 242 | 78 | 0.322 | 662 | 6 | 0.009 | -0.313 |
| med | True | 990 | 394 | 0.398 | 3655 | 63 | 0.017 | -0.381 |
| very_high | False | 59 | 6 | 0.102 | 112 | 0 | 0.0 | -0.102 |
| very_high | True | 123 | 29 | 0.236 | 692 | 20 | 0.029 | -0.207 |
| very_low | False | 61 | 24 | 0.393 | 558 | 12 | 0.022 | -0.371 |
| very_low | True | 643 | 393 | 0.611 | 3321 | 116 | 0.035 | -0.576 |
| zero | False | 118 | 75 | 0.636 | 1209 | 88 | 0.073 | -0.563 |
| zero | True | 2021 | 1541 | 0.762 | 8142 | 910 | 0.112 | -0.65 |

## T4 — full 5×6×2 grid: coverage 46 dense / 14 sparse / 0 empty

(See `T4_zero_gap.csv` — too wide for inline display.)

Top 10 cells with biggest synth-zero shortfall (delta_zr most negative):

| slot | s30_bin | avail | n_real | nz_real | zr_real | n_synth | nz_synth | zr_synth | delta_zr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | zero | False | 35 | 25 | 0.714 | 230 | 6 | 0.026 | -0.688 |
| 4 | zero | True | 260 | 190 | 0.731 | 1150 | 71 | 0.062 | -0.669 |
| 1 | zero | True | 753 | 664 | 0.882 | 3006 | 666 | 0.222 | -0.66 |
| 3 | zero | True | 362 | 248 | 0.685 | 1398 | 58 | 0.041 | -0.644 |
| 3 | very_low | True | 126 | 84 | 0.667 | 669 | 18 | 0.027 | -0.64 |
| 5 | zero | True | 333 | 229 | 0.688 | 1332 | 66 | 0.05 | -0.638 |
| 2 | zero | True | 313 | 210 | 0.671 | 1256 | 49 | 0.039 | -0.632 |
| 5 | zero | False | 21 | 13 | 0.619 | 212 | 3 | 0.014 | -0.605 |
| 5 | very_low | True | 149 | 92 | 0.617 | 779 | 12 | 0.015 | -0.602 |
| 2 | very_low | True | 141 | 87 | 0.617 | 655 | 26 | 0.04 | -0.577 |

## Persona stat cross-check (T2 raw vs aggregated `per_slot_zero_pct`)

| slot | avail | n | n_zero | zero_rate | ci_lo | ci_hi | persona_mean_zero_pct | persona_median_zero_pct | n_uids | delta_vs_persona |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | False | 73 | 28 | 0.384 | 0.281 | 0.498 | 0.449 | 0.367 | 28 | -0.065 |
| 1 | True | 1224 | 881 | 0.72 | 0.694 | 0.744 | 0.718 | 0.694 | 37 | 0.002 |
| 2 | False | 178 | 51 | 0.287 | 0.225 | 0.357 | 0.316 | 0.333 | 36 | -0.029 |
| 2 | True | 1115 | 574 | 0.515 | 0.485 | 0.544 | 0.511 | 0.514 | 37 | 0.004 |
| 3 | False | 157 | 51 | 0.325 | 0.257 | 0.402 | 0.324 | 0.25 | 36 | 0.001 |
| 3 | True | 1160 | 616 | 0.531 | 0.502 | 0.56 | 0.536 | 0.511 | 37 | -0.005 |
| 4 | False | 247 | 101 | 0.409 | 0.349 | 0.471 | 0.355 | 0.333 | 37 | 0.054 |
| 4 | True | 1032 | 534 | 0.517 | 0.487 | 0.548 | 0.508 | 0.514 | 37 | 0.009 |
| 5 | False | 154 | 53 | 0.344 | 0.274 | 0.422 | 0.333 | 0.333 | 35 | 0.011 |
| 5 | True | 1113 | 594 | 0.534 | 0.504 | 0.563 | 0.54 | 0.529 | 37 | -0.006 |

## Hypothesis verdict

**H2 (concentrated loss)** — CV(|Δzero_rate|)=0.37 is high. Worst 5 cells:
```
[
  {
    "slot": 4,
    "s30_bin": "very_high",
    "avail": true,
    "zr_real": 0.185,
    "zr_synth": 0.058,
    "delta_zr": -0.127
  },
  {
    "slot": 5,
    "s30_bin": "very_high",
    "avail": true,
    "zr_real": 0.156,
    "zr_synth": 0.012,
    "delta_zr": -0.144
  },
  {
    "slot": 2,
    "s30_bin": "high",
    "avail": false,
    "zr_real": 0.17,
    "zr_synth": 0.0,
    "delta_zr": -0.17
  },
  {
    "slot": 3,
    "s30_bin": "low",
    "avail": false,
    "zr_real": 0.208,
    "zr_synth": 0.03,
    "delta_zr": -0.178
  },
  {
    "slot": 2,
    "s30_bin": "med",
    "avail": false,
    "zr_real": 0.25,
    "zr_synth": 0.0,
    "delta_zr": -0.25
  }
]
```
→ Option C (two-stage hurdle: Bernoulli + LLM for positives) is the right fix.
