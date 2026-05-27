"""Evaluate molecular SMILES for physical plausibility using AIMNet2.

Pipeline:  SMILES  ->  RDKit 3D conformer  ->  AIMNet2 energy / forces / charges.

The RDKit prep (parse, 3D embed, MMFF) is CPU-bound and parallelised with
threads.  AIMNet2 uses torch.autograd.grad internally (for forces), so it
must run sequentially on the main thread to avoid graph corruption.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

_AIMNET_CALC = None

_SUPPORTED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}


def _get_calculator():
    """Return a shared AIMNet2Calculator instance (loaded on first call)."""
    global _AIMNET_CALC
    if _AIMNET_CALC is None:
        from aimnet.calculators import AIMNet2Calculator
        _AIMNET_CALC = AIMNet2Calculator("aimnet2")
        logger.info("[AIMNet] Loaded AIMNet2 calculator (aimnet2 model).")
    return _AIMNET_CALC


@dataclass
class AIMNetResult:
    """Physical-plausibility scores for a single molecule."""
    smiles: str
    energy: float
    max_force_norm: float
    mean_force_norm: float
    n_atoms: int


@dataclass
class AIMNetFailure:
    """Records why evaluation failed for a molecule."""
    smiles: str
    stage: str              # "parse" | "embed" | "aimnet"
    reason: str


PreparedMol = Tuple[str, np.ndarray, np.ndarray]  # (smiles, coords, numbers)

_stderr_lock = threading.Lock()


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect fd 2 during RDKit prep (UFFTYPER, kekulize, etc.)."""
    with _stderr_lock:
        with open(os.devnull, "w") as devnull:
            old_stderr = os.dup(2)
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)


def _prepare_smiles(smiles: str) -> Union[PreparedMol, AIMNetFailure]:
    """CPU-only: SMILES -> 3D coords + atomic numbers (thread-safe)."""
    with _suppress_stderr():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return AIMNetFailure(smiles, "parse", "RDKit could not parse SMILES")

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ret = AllChem.EmbedMolecule(mol, params)
        if ret != 0:
            return AIMNetFailure(smiles, "embed", f"EmbedMolecule returned {ret}")

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        conf = mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float64)
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)

        unsupported = set(numbers.tolist()) - _SUPPORTED_ELEMENTS
        if unsupported:
            names = [f"Z={z}" for z in sorted(unsupported)]
            return AIMNetFailure(smiles, "aimnet", f"Unsupported elements: {names}")

        return (smiles, coords, numbers)


def _run_aimnet(prepared: PreparedMol) -> AIMNetResult | AIMNetFailure:
    """GPU: run AIMNet2 on already-prepared coords/numbers (NOT thread-safe)."""
    smiles, coords, numbers = prepared
    calc = _get_calculator()
    try:
        result = calc(
            {"coord": coords, "numbers": numbers, "charge": 0.0},
            forces=True,
        )
    except Exception as exc:
        logger.warning("AIMNet failure: %s  SMILES='%s'", exc, smiles)
        return AIMNetFailure(smiles, "aimnet", str(exc))

    energy = float(result["energy"].cpu()) if hasattr(result["energy"], "cpu") else float(result["energy"])
    forces = result["forces"]
    if hasattr(forces, "cpu"):
        forces = forces.cpu().numpy()
    elif hasattr(forces, "numpy"):
        forces = forces.numpy()
    force_norms = np.linalg.norm(forces, axis=-1)

    return AIMNetResult(
        smiles=smiles,
        energy=energy,
        max_force_norm=float(force_norms.max()),
        mean_force_norm=float(force_norms.mean()),
        n_atoms=len(numbers),
    )


def evaluate_smiles(smiles: str) -> AIMNetResult | AIMNetFailure:
    """Score a single SMILES string (sequential convenience wrapper)."""
    prep = _prepare_smiles(smiles)
    if isinstance(prep, AIMNetFailure):
        return prep
    return _run_aimnet(prep)


def evaluate_smiles_list(
    smiles_list: list[str],
    n_workers: int = 0,
) -> list[AIMNetResult | AIMNetFailure]:
    """Evaluate a list of SMILES with parallel RDKit prep, sequential AIMNet.

    Args:
        smiles_list: SMILES strings to evaluate.
        n_workers:   Threads for RDKit prep.  0 = fully sequential.
    """
    if n_workers <= 0:
        return [evaluate_smiles(s) for s in smiles_list]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        prepared = list(pool.map(_prepare_smiles, smiles_list))

    results: list[AIMNetResult | AIMNetFailure] = []
    for prep in prepared:
        if isinstance(prep, AIMNetFailure):
            results.append(prep)
        else:
            results.append(_run_aimnet(prep))

    return results
