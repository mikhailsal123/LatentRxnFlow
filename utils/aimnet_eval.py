"""Evaluate molecular SMILES for physical plausibility using AIMNet2.

Pipeline:  SMILES  ->  RDKit 3D conformer  ->  AIMNet2 energy / forces.

RDKit prep (parse, 3D embed, MMFF) is CPU-bound and parallelised with threads.
AIMNet2 inference runs on the main thread in GPU batches.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

_AIMNET_CALC = None
_AIMNET_EVAL_CACHE: dict[str, "AIMNetResult | AIMNetFailure"] = {}

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
    # Take largest fragment only. Small standalone fragments (for example F/Cl/O)
    # are trivially valid but uninformative for this AIMNet diagnostic.
    if "." in smiles:
        smiles = max(smiles.split("."), key=len)

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


def _normalize_smiles_for_aimnet(smiles: str) -> str:
    """Normalize SMILES key for AIMNet evaluation cache."""
    if "." in smiles:
        return max(smiles.split("."), key=len)
    return smiles


def _prepare_smiles_with_timing(
    smiles: str,
) -> tuple[PreparedMol | AIMNetFailure, dict[str, float]]:
    """Timed variant of _prepare_smiles for profiling RDKit stages."""
    smiles = _normalize_smiles_for_aimnet(smiles)

    t_parse = 0.0
    t_addhs = 0.0
    t_embed = 0.0
    t_mmff = 0.0
    t_extract = 0.0

    with _suppress_stderr():
        t0 = time.time()
        mol = Chem.MolFromSmiles(smiles)
        t_parse += time.time() - t0
        if mol is None:
            return AIMNetFailure(smiles, "parse", "RDKit could not parse SMILES"), {
                "parse_s": t_parse,
                "addhs_s": t_addhs,
                "embed_s": t_embed,
                "mmff_s": t_mmff,
                "extract_s": t_extract,
            }

        t0 = time.time()
        mol = Chem.AddHs(mol)
        t_addhs += time.time() - t0

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        t0 = time.time()
        ret = AllChem.EmbedMolecule(mol, params)
        t_embed += time.time() - t0
        if ret != 0:
            return AIMNetFailure(smiles, "embed", f"EmbedMolecule returned {ret}"), {
                "parse_s": t_parse,
                "addhs_s": t_addhs,
                "embed_s": t_embed,
                "mmff_s": t_mmff,
                "extract_s": t_extract,
            }

        t0 = time.time()
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        t_mmff += time.time() - t0

        t0 = time.time()
        conf = mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float64)
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        t_extract += time.time() - t0

        unsupported = set(numbers.tolist()) - _SUPPORTED_ELEMENTS
        if unsupported:
            names = [f"Z={z}" for z in sorted(unsupported)]
            return AIMNetFailure(smiles, "aimnet", f"Unsupported elements: {names}"), {
                "parse_s": t_parse,
                "addhs_s": t_addhs,
                "embed_s": t_embed,
                "mmff_s": t_mmff,
                "extract_s": t_extract,
            }

    return (smiles, coords, numbers), {
        "parse_s": t_parse,
        "addhs_s": t_addhs,
        "embed_s": t_embed,
        "mmff_s": t_mmff,
        "extract_s": t_extract,
    }


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


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _run_aimnet_batch_mol_idx(
    prepared_list: list[PreparedMol | AIMNetFailure],
    batch_size: int,
) -> list[AIMNetResult | AIMNetFailure]:
    """Run AIMNet in variable-size GPU batches using mol_idx."""
    calc = _get_calculator()
    results: list[AIMNetResult | AIMNetFailure | None] = [None] * len(prepared_list)

    valid_items: list[tuple[int, PreparedMol]] = []
    for i, item in enumerate(prepared_list):
        if isinstance(item, AIMNetFailure):
            results[i] = item
        else:
            valid_items.append((i, item))

    for start in range(0, len(valid_items), batch_size):
        chunk = valid_items[start : start + batch_size]
        global_indices = [idx for idx, _ in chunk]
        chunk_mols = [mol for _, mol in chunk]
        B = len(chunk_mols)

        atom_counts = [len(m[2]) for m in chunk_mols]
        mol_offsets = np.zeros(B + 1, dtype=np.int64)
        mol_offsets[1:] = np.cumsum(atom_counts, dtype=np.int64)
        n_total = int(mol_offsets[-1])

        coords = np.empty((n_total, 3), dtype=np.float64)
        numbers = np.empty((n_total,), dtype=np.int64)
        mol_idx = np.empty((n_total,), dtype=np.int64)

        for local_idx, (_, c, z) in enumerate(chunk_mols):
            s = int(mol_offsets[local_idx])
            e = int(mol_offsets[local_idx + 1])
            coords[s:e] = c
            numbers[s:e] = z
            mol_idx[s:e] = local_idx

        try:
            out = calc(
                {
                    "coord": coords,
                    "numbers": numbers,
                    "mol_idx": mol_idx,
                    "charge": np.zeros(B, dtype=np.float64),
                },
                forces=True,
            )
        except Exception as exc:
            logger.warning("AIMNet mol_idx batch failure for %d mols: %s", B, exc)
            for gidx, mol in zip(global_indices, chunk_mols):
                results[gidx] = AIMNetFailure(mol[0], "aimnet", str(exc))
            continue

        energies = _to_numpy(out["energy"])
        forces = _to_numpy(out["forces"])
        if np.ndim(energies) == 0:
            energies = np.full((B,), float(energies), dtype=np.float64)

        for local_idx, (gidx, mol) in enumerate(zip(global_indices, chunk_mols)):
            s = int(mol_offsets[local_idx])
            e = int(mol_offsets[local_idx + 1])
            force_norms = np.linalg.norm(forces[s:e], axis=-1)
            results[gidx] = AIMNetResult(
                smiles=mol[0],
                energy=float(energies[local_idx]),
                max_force_norm=float(force_norms.max()),
                mean_force_norm=float(force_norms.mean()),
                n_atoms=len(mol[2]),
            )

    return [r for r in results if r is not None]


def evaluate_smiles(smiles: str) -> AIMNetResult | AIMNetFailure:
    """Score a single SMILES string (sequential convenience wrapper)."""
    prep = _prepare_smiles(smiles)
    if isinstance(prep, AIMNetFailure):
        return prep
    return _run_aimnet(prep)


def evaluate_smiles_list(
    smiles_list: list[str],
    n_workers: int = 0,
    batch_size: int = 64,
    return_timing: bool = False,
) -> list[AIMNetResult | AIMNetFailure] | tuple[list[AIMNetResult | AIMNetFailure], dict[str, float]]:
    """Evaluate a list of SMILES with parallel RDKit prep and batched AIMNet.

    Args:
        smiles_list: SMILES strings to evaluate.
        n_workers:   Threads for RDKit prep.  0 = fully sequential.
        batch_size:  Molecules per AIMNet GPU batch.
        return_timing: If True, also return stage timings in seconds.
    """
    global _AIMNET_EVAL_CACHE
    t0 = time.time()
    rdkit_detail = {
        "parse_s": 0.0,
        "addhs_s": 0.0,
        "embed_s": 0.0,
        "mmff_s": 0.0,
        "extract_s": 0.0,
    }
    normalized_smiles = [_normalize_smiles_for_aimnet(s) for s in smiles_list]
    unique_miss_keys = list(dict.fromkeys(
        s for s in normalized_smiles if s not in _AIMNET_EVAL_CACHE
    ))

    if return_timing:
        if n_workers > 0:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                timed = list(pool.map(_prepare_smiles_with_timing, unique_miss_keys))
        else:
            timed = [_prepare_smiles_with_timing(s) for s in unique_miss_keys]
        prepared = [item[0] for item in timed]
        for _, d in timed:
            for k in rdkit_detail:
                rdkit_detail[k] += d[k]
    else:
        if n_workers > 0:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                prepared = list(pool.map(_prepare_smiles, unique_miss_keys))
        else:
            prepared = [_prepare_smiles(s) for s in unique_miss_keys]
    t1 = time.time()

    computed = _run_aimnet_batch_mol_idx(prepared, batch_size=batch_size)
    if len(computed) != len(unique_miss_keys):
        raise RuntimeError(
            f"AIMNet cache fill mismatch: got {len(computed)} results for "
            f"{len(unique_miss_keys)} cache misses."
        )
    for key, result in zip(unique_miss_keys, computed):
        _AIMNET_EVAL_CACHE[key] = result

    results = [_AIMNET_EVAL_CACHE[s] for s in normalized_smiles]
    t2 = time.time()

    if return_timing:
        return results, {
            "rdkit_prep_s": t1 - t0,
            "aimnet_compute_s": t2 - t1,
            "aimnet_total_s": t2 - t0,
            "cache_hits": float(len(smiles_list) - len(unique_miss_keys)),
            "cache_misses": float(len(unique_miss_keys)),
            "rdkit_parse_s": rdkit_detail["parse_s"],
            "rdkit_addhs_s": rdkit_detail["addhs_s"],
            "rdkit_embed_s": rdkit_detail["embed_s"],
            "rdkit_mmff_s": rdkit_detail["mmff_s"],
            "rdkit_extract_s": rdkit_detail["extract_s"],
        }
    return results
