"""Runtime bridge to the trained GAT in the sibling ``football-state-of-play`` repo.

The relational model (the GAT, its ``build_data`` graph contract and the trained checkpoint) lives in
``football-state-of-play`` and is treated as **read-only**. Rather than vendoring or pip-installing it
(its top-level ``eval`` package would collide with this repo's ``eval``), we inject its root onto
``sys.path`` at call time and import only the two things we need -- ``data.graphs.build_data`` and
``models.gnn.GAT`` -- then run our own thin predict loop (mirroring ``eval/ood_demo`` and
``eval/cv_bridge`` without depending on either).

This closes the loop end-to-end: a positions parquet -> contract :class:`FreezeFrame`s
(:mod:`generator.to_frames`) -> the trained GAT -> a clip-level relational readout.

Module import is light (contract + stdlib only); torch and the sibling packages are imported lazily
inside :func:`load_model`, so importing this module never requires torch-geometric to be installed.

Run ``python -m generator.sop_bridge --positions <path/to/positions.parquet>`` under an interpreter
that has the modelling stack (e.g. ``football-state-of-play/.venv``).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from generator.contract import PITCH_WIDTH, FreezeFrame, Substrate, to_statsbomb_dataframe
from generator.to_frames import MIN_PLAYERS, frames_from_positions

logger = logging.getLogger(__name__)

#: Environment variable overriding where ``football-state-of-play`` lives.
SOP_PATH_ENV = "FOOTBALL_SOP_PATH"
#: Environment variable overriding which interpreter has the modelling stack (torch-geometric).
SOP_PYTHON_ENV = "FOOTBALL_SOP_PYTHON"
#: Default sibling location (``../football-state-of-play`` relative to this repo root).
_DEFAULT_SOP_ROOT = Path(__file__).resolve().parents[1].parent / "football-state-of-play"
_CKPT_RELATIVE = Path("results/checkpoints/gnn.pt")
#: Guard against an infinite re-exec loop if the target interpreter also lacks the stack.
_REEXEC_FLAG = "_FS_SOP_BRIDGE_REEXEC"


def sop_root() -> Path:
    """Resolve the ``football-state-of-play`` repo root (``$FOOTBALL_SOP_PATH`` or the sibling dir)."""
    root = Path(os.environ.get(SOP_PATH_ENV, _DEFAULT_SOP_ROOT)).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"football-state-of-play not found at {root}. Set ${SOP_PATH_ENV} to its root."
        )
    return root


def has_modelling_runtime() -> bool:
    """True if torch-geometric (and hence the GAT) can be imported in this interpreter."""
    return importlib.util.find_spec("torch_geometric") is not None


def sop_python() -> Path | None:
    """Locate an interpreter with the modelling stack: ``$FOOTBALL_SOP_PYTHON`` or the SoP ``.venv``.

    Returns:
        Path to a Python executable, or ``None`` if none is found.
    """
    override = os.environ.get(SOP_PYTHON_ENV)
    if override:
        p = Path(override)
        return p if p.exists() else None
    try:
        root = sop_root()
    except FileNotFoundError:
        return None
    for cand in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if cand.exists():
            return cand
    return None


def reexec_under_sop_if_needed(argv: list[str] | None = None) -> None:
    """If this interpreter lacks torch-geometric, re-run this module under one that has it.

    A no-op when the modelling stack is already importable (or we have already re-exec'd once).
    Lets ``python -m generator.sop_bridge ...`` work from *any* interpreter -- it hops to the SoP
    ``.venv`` automatically and exits with that run's return code.

    Args:
        argv: CLI args to forward (defaults to ``sys.argv[1:]``).
    """
    if has_modelling_runtime() or os.environ.get(_REEXEC_FLAG):
        return
    py = sop_python()
    if py is None:
        raise RuntimeError(
            "torch-geometric is not available in this interpreter and no modelling interpreter "
            f"was found. Set ${SOP_PYTHON_ENV} to a Python that has it (e.g. the SoP .venv)."
        )
    args = sys.argv[1:] if argv is None else argv
    logger.warning("torch-geometric absent here; re-running under %s", py)
    env = {**os.environ, _REEXEC_FLAG: "1"}
    proc = subprocess.run([str(py), "-m", "generator.sop_bridge", *args], env=env, check=False)
    sys.exit(proc.returncode)


def checkpoint_path() -> Path:
    """Path to the trained GAT checkpoint inside the sibling repo."""
    return sop_root() / _CKPT_RELATIVE


def _inject_sop_path() -> Path:
    """Put the sibling repo root on ``sys.path`` (front) so ``data``/``models`` import. Idempotent."""
    root = sop_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_model(ckpt: Path | None = None):
    """Inject the sibling path, build the GAT and load the trained weights (eval mode).

    Args:
        ckpt: Override checkpoint path; defaults to the sibling repo's ``results/checkpoints/gnn.pt``.

    Returns:
        The trained ``models.gnn.GAT`` in eval mode.
    """
    _inject_sop_path()
    import torch  # noqa: PLC0415 - lazy: only needed when actually running the model

    from models.gnn import GAT  # noqa: PLC0415 - sibling import, available after path injection

    ckpt = ckpt or checkpoint_path()
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained GAT checkpoint at {ckpt}.")
    model = GAT()
    # Map to CPU: the checkpoint was saved on CUDA but this may be a CPU-only interpreter.
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    model.eval()
    return model


def _row_for(frame: FreezeFrame) -> SimpleNamespace:
    """Build the possession ``row`` ``build_data`` expects (ball location + graph-level context)."""
    if frame.ball is not None:
        bx, by = frame.ball
    else:  # no ball located: fall back to the on-ball actor, else the squad centroid.
        actors = [p for p in frame.players if p.is_actor]
        if actors:
            bx, by = actors[0].x, actors[0].y
        else:
            bx = sum(p.x for p in frame.players) / frame.n_players
            by = PITCH_WIDTH / 2
    return SimpleNamespace(
        trigger_x=float(bx),
        trigger_y=float(by),
        play_pattern=frame.play_pattern,
        from_counter=frame.from_counter,
        trigger_time_s=frame.trigger_time_s,
    )


def predict_frame(model, frame: FreezeFrame) -> dict[str, float]:
    """Run the trained GAT on one contract :class:`FreezeFrame`.

    Returns:
        ``{success, dynamic_xt, p_defstop, top_receiver}`` (NaN for any head the model omits).
    """
    _inject_sop_path()
    import torch  # noqa: PLC0415

    from data.graphs import build_data  # noqa: PLC0415

    data = build_data(to_statsbomb_dataframe(frame), _row_for(frame))  # no labels -> inference graph
    with torch.no_grad():
        out = model(data)
        recv = out.get("receiver_probs")
        return {
            "success": float(torch.sigmoid(out["success"])),
            "dynamic_xt": float(out["dxt"]),
            "p_defstop": (
                float(torch.sigmoid(out["defsuccess"])) if "defsuccess" in out else float("nan")
            ),
            "top_receiver": float(recv.max()) if recv is not None else float("nan"),
        }


def run_clip(
    positions_path: str | Path,
    *,
    substrate: Substrate = Substrate.BROADCAST_CV,
    ckpt: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the trained GAT over every quality-gated frame of a positions parquet.

    Args:
        positions_path: A ``cv-football`` / tracking positions parquet.
        substrate: Source-substrate tag for the emitted frames.
        ckpt: Override checkpoint path.
        limit: Stop after this many qualifying frames (smoke testing).

    Returns:
        One row per qualifying frame: ``frame, n_players, success, dynamic_xt, p_defstop,
        top_receiver``.
    """
    import torch  # noqa: PLC0415

    positions = pd.read_parquet(positions_path)
    model = load_model(ckpt)
    rows: list[dict] = []
    with torch.no_grad():
        for fr, frame in frames_from_positions(positions, substrate=substrate):
            try:
                pred = predict_frame(model, frame)
            except Exception as exc:  # noqa: BLE001 - log the frame and carry on
                logger.warning("frame %s: %s", fr, exc)
                continue
            rows.append({"frame": fr, "n_players": frame.n_players, **pred})
            if limit is not None and len(rows) >= limit:
                break
    return pd.DataFrame(rows)


def clip_readout(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate a :func:`run_clip` frame table to clip-level means."""
    return {
        "n_frames": int(len(df)),
        "mean_success": float(df["success"].mean()),
        "mean_dynamic_xt": float(df["dynamic_xt"].mean()),
        "mean_p_defstop": float(df["p_defstop"].mean()),
    }


def main() -> None:
    """Run the bridge on a positions parquet and print a clip-level relational readout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reexec_under_sop_if_needed()  # hop to the SoP .venv if torch-geometric is missing here
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="positions parquet (cv-football / tracking)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N qualifying frames")
    args = ap.parse_args()

    df = run_clip(args.positions, limit=args.limit)
    if df.empty:
        print(f"No frames passed the >= {MIN_PLAYERS}-player gate (broadcast too zoomed).")
        return
    r = clip_readout(df)
    print(f"\nRan the trained GAT on {r['n_frames']} qualifying frames.")
    print("Clip-level relational readout (means):")
    print(f"  P(success)     : {r['mean_success']:.3f}")
    print(f"  Dynamic-xT     : {r['mean_dynamic_xt']:.4f}")
    print(f"  P(def recovers): {r['mean_p_defstop']:.3f}")
    print("\nVideo -> contract frames -> relational metrics. Aggregate per team for a CV identity.")


if __name__ == "__main__":
    main()
