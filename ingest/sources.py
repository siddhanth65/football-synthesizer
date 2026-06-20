"""Data-source registry: major NATIONAL-TEAM international tournaments, 2018-2026.

Per the locked scope: focus on major international (national-team) tournaments from 2018-2026 and
leverage *all* available data per source. Club football and friendlies are out of scope.

Three data modalities per tournament, each with an honest availability flag (verify in week 1):
- ``footage``     -- broadcast video (YouTube etc.), the generator's input.
- ``statsbomb_360`` -- public StatsBomb 360 freeze-frames (training / validation backbone).
- ``fifa_efi``    -- FIFA Enhanced Football Intelligence post-match reports (validation oracle, C6).
                     FIFA-run tournaments only (EFI debuted at the 2022 World Cup).

Run ``python -m ingest.sources --list``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum


class Avail(str, Enum):
    """Availability of a data modality for a tournament."""

    YES = "yes"
    PARTIAL = "partial"
    NO = "no"
    TBD = "tbd"  # tournament ongoing / not yet verified


@dataclass(frozen=True)
class Tournament:
    """One national-team international tournament and its data availability."""

    key: str
    name: str
    year: int
    confederation: str  # FIFA / UEFA / CONMEBOL / CAF
    gender: str  # men / women
    footage: Avail
    statsbomb_360: Avail
    fifa_efi: Avail
    note: str = ""


# Availability flags are best-effort and MUST be verified in week-1 EDA (mirrors the parent
# project's approach). FIFA EFI exists only for FIFA tournaments from 2022 onward.
TOURNAMENTS: tuple[Tournament, ...] = (
    # --- Men, FIFA World Cup ---
    Tournament("wc2018", "FIFA World Cup", 2018, "FIFA", "men", Avail.YES, Avail.NO, Avail.NO,
               "Pre-EFI; 360 not released."),
    Tournament("wc2022", "FIFA World Cup", 2022, "FIFA", "men", Avail.YES, Avail.PARTIAL, Avail.YES,
               "EFI debut; the France-Senegal report is from this style. Primary oracle source."),
    Tournament("wc2026", "FIFA World Cup", 2026, "FIFA", "men", Avail.TBD, Avail.TBD, Avail.YES,
               "Underway; live footage sparse/rights-locked; EFI being produced."),
    # --- Men, UEFA Euro ---
    Tournament("euro2020", "UEFA Euro", 2020, "UEFA", "men", Avail.YES, Avail.YES, Avail.NO,
               "StatsBomb free 360 release."),
    Tournament("euro2024", "UEFA Euro", 2024, "UEFA", "men", Avail.YES, Avail.PARTIAL, Avail.NO),
    # --- Men, CONMEBOL Copa America ---
    Tournament("copa2019", "Copa America", 2019, "CONMEBOL", "men", Avail.YES, Avail.NO, Avail.NO),
    Tournament("copa2021", "Copa America", 2021, "CONMEBOL", "men", Avail.YES, Avail.NO, Avail.NO),
    Tournament("copa2024", "Copa America", 2024, "CONMEBOL", "men", Avail.YES, Avail.PARTIAL, Avail.NO),
    # --- Women, FIFA World Cup ---
    Tournament("wwc2019", "FIFA Women's World Cup", 2019, "FIFA", "women", Avail.YES, Avail.NO, Avail.NO),
    Tournament("wwc2023", "FIFA Women's World Cup", 2023, "FIFA", "women", Avail.YES, Avail.YES, Avail.YES,
               "360 released; FIFA EFI produced."),
    # --- Women, UEFA Euro ---
    Tournament("weuro2022", "UEFA Women's Euro", 2022, "UEFA", "women", Avail.YES, Avail.YES, Avail.NO),
    Tournament("weuro2025", "UEFA Women's Euro", 2025, "UEFA", "women", Avail.YES, Avail.TBD, Avail.NO),
    # --- Men, CAF AFCON (breadth / transfer; footage only) ---
    Tournament("afcon2021", "Africa Cup of Nations", 2021, "CAF", "men", Avail.YES, Avail.NO, Avail.NO,
               "Breadth + cross-confederation transfer test."),
    Tournament("afcon2023", "Africa Cup of Nations", 2023, "CAF", "men", Avail.YES, Avail.NO, Avail.NO),
)


def with_efi() -> list[Tournament]:
    """Tournaments that have FIFA EFI reports (the validation oracle, C6)."""
    return [t for t in TOURNAMENTS if t.fifa_efi in (Avail.YES, Avail.TBD)]


def with_360() -> list[Tournament]:
    """Tournaments with public StatsBomb 360 (the modelling/validation backbone)."""
    return [t for t in TOURNAMENTS if t.statsbomb_360 in (Avail.YES, Avail.PARTIAL)]


def main() -> None:
    """Print the data-source registry."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list all tournaments")
    ap.parse_args()
    hdr = f"{'key':10} {'tournament':28} {'yr':4} {'conf':9} {'sex':5} {'video':8} {'360':8} {'EFI':8}"
    print(hdr)
    print("-" * len(hdr))
    for t in TOURNAMENTS:
        print(
            f"{t.key:10} {t.name:28} {t.year:<4} {t.confederation:9} {t.gender:5} "
            f"{t.footage.value:8} {t.statsbomb_360.value:8} {t.fifa_efi.value:8}"
        )
    print(f"\n{len(TOURNAMENTS)} tournaments | with 360: {len(with_360())} | with FIFA EFI: {len(with_efi())}")


if __name__ == "__main__":
    main()
