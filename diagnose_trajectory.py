"""Trajectory Physical-Plausibility Diagnostic via AIMNet2.

For each reaction in a small test subset, this script:
  1. Runs the pretrained model's ODE integration, decoding a molecule at each step.
  2. Builds a LERP control (linear interpolation src_enc -> tgt_enc in latent space).
  3. Evaluates every decoded intermediate with AIMNet2 (energy, forces).
  4. Compares the two trajectories on validity, energy smoothness, and force norms.

Results are logged to wandb (tables, per-reaction scalars, and final summary).

Usage:
    python diagnose_trajectory.py --config configs/diagnose_trajectory.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from tqdm import tqdm

warnings.filterwarnings("ignore")

FLOW_ROOT = Path(__file__).resolve().parent

import sys

if str(FLOW_ROOT) not in sys.path:
    sys.path.append(str(FLOW_ROOT))

from data.uspto_main_product import USPTOReact2MainProduct, collate_fn
from models.flow_nerf_model import FlowNERFModel, SimpleArgs, DecoderConfig
from utils.aimnet_eval import (
    AIMNetFailure,
    AIMNetResult,
    _suppress_stderr,
    evaluate_smiles,
    evaluate_smiles_list,
)
from utils.data_utils import result2mol
from utils.encoder_utils import load_checkpoint
from utils.experiment import load_config
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> FlowNERFModel:
    """Instantiate FlowNERFModel from config and load checkpoint weights."""
    model_cfg = cfg["model"]

    args_nn = SimpleArgs()

    class ModelConfig:
        pass

    args_nn.model = ModelConfig()
    args_nn.model.flow_cond_head = model_cfg.get("flow_cond_head", "controlnet")
    args_nn.model.film_hidden_dim = model_cfg.get(
        "film_hidden_dim", model_cfg.get("latent_dim", 256) * 2
    )
    args_nn.model.film_init_zero = model_cfg.get("film_init_zero", True)
    args_nn.model.film_s_gamma = model_cfg.get("film_s_gamma", 1.0)
    args_nn.model.film_s_beta = model_cfg.get("film_s_beta", 0.2)
    args_nn.model.cond_pool = model_cfg.get("cond_pool", "gated")
    args_nn.model.cond_drop_prob = model_cfg.get("cond_drop_prob", 0.2)
    args_nn.model.force_zero_cond = model_cfg.get("force_zero_cond", False)
    args_nn.model.condition_source = model_cfg.get("condition_source", "fp")

    dec = model_cfg.get("decoder", {}) or {}
    decoder_cfg = DecoderConfig(
        delta_source=dec.get("delta_source", "tf"),
        input_mode=dec.get("input_mode", "fuse"),
        ode_method=dec.get("ode_method", "heun"),
    )

    model = FlowNERFModel(
        latent_dim=model_cfg["latent_dim"],
        cond_dim=model_cfg["cond_dim"],
        time_embed_dim=model_cfg["time_embed_dim"],
        ntoken=model_cfg.get("ntoken", 128),
        args=args_nn,
        flow_weight=model_cfg.get("flow_weight", 1e-2),
        detach_encoder_for_flow=model_cfg.get("detach_encoder_for_flow", True),
        flow_sampling_cfg=model_cfg.get("flow_sampling_cfg", None),
        fm_sigma=model_cfg.get("fm_sigma", 0.0),
        decoder_cfg=decoder_cfg,
        use_conditional_flow=model_cfg.get("use_conditional_flow", False),
        nfe=dec.get("nfe", 20),
    ).to(device)

    load_checkpoint(model, {"checkpoint_path": cfg["eval"]["checkpoint_path"]})
    model.eval()
    return model


def structures_to_smiles(
    element: torch.Tensor,
    src_mask: torch.Tensor,
    structures: List[Dict[str, torch.Tensor]],
) -> List[List[str]]:
    """Decode a list of per-step structure dicts into per-step SMILES lists.

    Args:
        element:    [B, L] atomic numbers (constant across steps).
        src_mask:   [B, L] padding mask.
        structures: list of n_steps+1 dicts each with {bond, aroma, charge} [B, ...].

    Returns:
        smiles_per_step: list of length n_steps+1, each a list of B SMILES strings.
    """
    B = element.shape[0]
    smiles_per_step: List[List[str]] = []

    for step_struct in structures:
        bond = step_struct["bond"]
        aroma = step_struct["aroma"]
        charge = step_struct["charge"]

        step_smiles: List[str] = []
        for j in range(B):
            try:
                with _suppress_stderr():
                    _, smi, _ = result2mol(
                        (element[j], src_mask[j], bond[j], aroma[j], charge[j], None)
                    )
            except Exception:
                smi = ""
            step_smiles.append(smi)
        smiles_per_step.append(step_smiles)

    return smiles_per_step


def lerp_decode(
    model: FlowNERFModel,
    src_enc: torch.Tensor,
    tgt_enc: torch.Tensor,
    tensors_gpu: Dict[str, Any],
    n_steps: int,
    temperature: float,
) -> List[Dict[str, torch.Tensor]]:
    """Build LERP control: linearly interpolate src_enc -> tgt_enc and decode.

    Returns a list of n_steps+1 structure dicts {bond, aroma, charge} on CPU.
    """
    L, B, D = src_enc.shape
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=src_enc.device)

    structures: List[Dict[str, torch.Tensor]] = []
    for alpha in alphas:
        z_interp = (1.0 - alpha) * src_enc + alpha * tgt_enc  # [L, B, D]
        delta = z_interp - src_enc
        z0_flat = src_enc.reshape(L * B, D)
        d_flat = delta.reshape(L * B, D)
        fused = model.delta_fuser(
            torch.cat([z0_flat, d_flat], dim=-1)
        ).view(L, B, D)

        result = model.backbone.M_decoder.sample(
            src_embedding=fused,
            src_bond=tensors_gpu["src_bond"],
            padding_mask=tensors_gpu["src_mask"],
            temperature=temperature,
        )
        structures.append({k: v.cpu() for k, v in result.items()})

    return structures


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    aimnet_results: List[AIMNetResult | AIMNetFailure],
) -> Dict[str, Any]:
    """Compute summary metrics from a sequence of AIMNet evaluations."""
    import math

    n_total = len(aimnet_results)
    n_success = sum(isinstance(r, AIMNetResult) for r in aimnet_results)

    # Count failures by stage
    failures = [r for r in aimnet_results if isinstance(r, AIMNetFailure)]
    failure_counts = {}
    for f in failures:
        failure_counts[f.stage] = failure_counts.get(f.stage, 0) + 1

    energies = [r.energy if isinstance(r, AIMNetResult) else None for r in aimnet_results]

    # Filter out NaN energies (unsupported elements that slipped through)
    finite_energies = [e for e in energies if e is not None and not math.isnan(e)]
    if len(finite_energies) >= 2:
        diffs = np.diff(finite_energies)
        energy_smoothness = float(np.std(diffs))
    else:
        energy_smoothness = None

    force_norms_mean = [
        r.mean_force_norm for r in aimnet_results
        if isinstance(r, AIMNetResult) and not math.isnan(r.mean_force_norm)
    ]
    force_norms_max = [
        r.max_force_norm for r in aimnet_results
        if isinstance(r, AIMNetResult) and not math.isnan(r.max_force_norm)
    ]

    return {
        "validity_rate": n_success / n_total if n_total > 0 else 0.0,
        "n_success": n_success,
        "n_failed": len(failures),
        "n_total": n_total,
        "failure_counts": failure_counts,
        "energies": energies,
        "energy_smoothness": energy_smoothness,
        "mean_force_norm_avg": float(np.mean(force_norms_mean)) if force_norms_mean else None,
        "max_force_norm_worst": float(np.max(force_norms_max)) if force_norms_max else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory plausibility diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("train", {}).get("seed", 42))

    diag_cfg = cfg.get("diagnose", {})
    n_reactions = diag_cfg.get("n_reactions", 100)
    n_steps = diag_cfg.get("n_trajectory_steps", 10)
    aimnet_workers = diag_cfg.get("aimnet_workers", 8)
    save_dir = Path(diag_cfg.get("save_path", "experiments/trajectory_diagnostic"))
    save_dir.mkdir(parents=True, exist_ok=True)
    temperature = cfg.get("eval", {}).get("temperature", 0.7)

    # -- wandb --
    use_wandb = cfg.get("eval", {}).get("use_wandb", True)
    wandb.init(
        project=cfg.get("experiment", {}).get("project", "flow-nerf-diagnostic"),
        name=cfg.get("experiment", {}).get("name", "trajectory-aimnet-diagnostic"),
        config=cfg,
        mode="online" if use_wandb else "disabled",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.log({"device": str(device)})

    # -- Load model --
    model = build_model(cfg, device)

    # -- Load data --
    pickle_path = cfg["eval"]["pickle_path"]
    with open(pickle_path, "rb") as f:
        data_list = pickle.load(f)

    subset = data_list[:n_reactions]
    dataset = USPTOReact2MainProduct(data_list=subset, if_shuffle=False)

    from torch.utils.data import DataLoader

    batch_size = cfg.get("eval", {}).get("batch_size", 128)
    num_workers = cfg.get("data", {}).get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # -- Per-reaction wandb table --
    reaction_table = wandb.Table(columns=[
        "rxn_idx",
        "model_validity", "lerp_validity",
        "model_energy_smoothness", "lerp_energy_smoothness",
        "model_mean_force", "lerp_mean_force",
        "model_smiles_final", "lerp_smiles_final",
    ])

    # -- Run diagnostic --
    all_results: List[Dict[str, Any]] = []
    global_rxn_idx = 0

    for batch in tqdm(loader, desc="Diagnosing trajectories"):
        tensors_gpu: Dict[str, Any] = {}
        for k, v in batch.reactant.data.items():
            tensors_gpu[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        if hasattr(batch.condition, "data") and isinstance(batch.condition.data, dict):
            for k, v in batch.condition.data.items():
                tensors_gpu[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        element = tensors_gpu["element"].cpu()    # [B, L]
        src_mask = tensors_gpu["src_mask"].cpu()   # [B, L]
        B = element.shape[0]

        # --- Model trajectory ---
        with torch.no_grad():
            traj_out = model.sample_trajectory_structures(
                tensors_gpu, temperature=temperature, n_steps=n_steps,
            )

        model_smiles = structures_to_smiles(element, src_mask, traj_out["structures"])

        # --- LERP control trajectory ---
        with torch.no_grad():
            lerp_structs = lerp_decode(
                model,
                src_enc=traj_out["src_enc"],
                tgt_enc=traj_out["tgt_enc"],
                tensors_gpu=tensors_gpu,
                n_steps=n_steps,
                temperature=temperature,
            )

        lerp_smiles = structures_to_smiles(element, src_mask, lerp_structs)

        # --- Evaluate all samples x steps with AIMNet2 in parallel ---
        n_steps_total = len(model_smiles)  # n_steps + 1
        all_smiles = []
        for j in range(B):
            for step in model_smiles:
                all_smiles.append(step[j])
            for step in lerp_smiles:
                all_smiles.append(step[j])

        all_aimnet = evaluate_smiles_list(all_smiles, n_workers=aimnet_workers)

        for j in range(B):
            rxn_idx = global_rxn_idx + j
            offset = j * 2 * n_steps_total
            sample_model_aimnet = all_aimnet[offset : offset + n_steps_total]
            sample_lerp_aimnet = all_aimnet[offset + n_steps_total : offset + 2 * n_steps_total]

            model_metrics = compute_trajectory_metrics(sample_model_aimnet)
            lerp_metrics = compute_trajectory_metrics(sample_lerp_aimnet)

            wandb.log({
                "per_rxn/model_validity": model_metrics["validity_rate"],
                "per_rxn/lerp_validity": lerp_metrics["validity_rate"],
                "per_rxn/model_energy_smoothness": model_metrics["energy_smoothness"],
                "per_rxn/lerp_energy_smoothness": lerp_metrics["energy_smoothness"],
                "per_rxn/model_mean_force": model_metrics["mean_force_norm_avg"],
                "per_rxn/lerp_mean_force": lerp_metrics["mean_force_norm_avg"],
            }, step=rxn_idx)

            reaction_table.add_data(
                rxn_idx,
                model_metrics["validity_rate"],
                lerp_metrics["validity_rate"],
                model_metrics["energy_smoothness"],
                lerp_metrics["energy_smoothness"],
                model_metrics["mean_force_norm_avg"],
                lerp_metrics["mean_force_norm_avg"],
                model_smiles[-1][j] if model_smiles else "",
                lerp_smiles[-1][j] if lerp_smiles else "",
            )

            result = {
                "rxn_idx": rxn_idx,
                "model_smiles": [step[j] for step in model_smiles],
                "lerp_smiles": [step[j] for step in lerp_smiles],
                "model_metrics": model_metrics,
                "lerp_metrics": lerp_metrics,
                "model_aimnet_details": [
                    asdict(r) if isinstance(r, (AIMNetResult, AIMNetFailure)) else None
                    for r in sample_model_aimnet
                ],
                "lerp_aimnet_details": [
                    asdict(r) if isinstance(r, (AIMNetResult, AIMNetFailure)) else None
                    for r in sample_lerp_aimnet
                ],
            }
            all_results.append(result)

        global_rxn_idx += B

    # -- Aggregate summary --
    import math

    def _finite(vals):
        return [v for v in vals if v is not None and not math.isnan(v)]

    # Total step-level counts across all reactions
    total_steps = sum(r["model_metrics"]["n_total"] for r in all_results)
    model_successes = sum(r["model_metrics"]["n_success"] for r in all_results)
    lerp_successes = sum(r["lerp_metrics"]["n_success"] for r in all_results)
    model_failures = sum(r["model_metrics"]["n_failed"] for r in all_results)
    lerp_failures = sum(r["lerp_metrics"]["n_failed"] for r in all_results)

    # Aggregate failure reasons across all reactions
    model_failure_reasons: Dict[str, int] = {}
    lerp_failure_reasons: Dict[str, int] = {}
    for r in all_results:
        for stage, count in r["model_metrics"].get("failure_counts", {}).items():
            model_failure_reasons[stage] = model_failure_reasons.get(stage, 0) + count
        for stage, count in r["lerp_metrics"].get("failure_counts", {}).items():
            lerp_failure_reasons[stage] = lerp_failure_reasons.get(stage, 0) + count

    # Reactions with usable energy data (no NaN, at least 2 valid steps)
    model_smoothness = _finite([r["model_metrics"]["energy_smoothness"] for r in all_results])
    lerp_smoothness = _finite([r["lerp_metrics"]["energy_smoothness"] for r in all_results])
    model_forces = _finite([r["model_metrics"]["mean_force_norm_avg"] for r in all_results])
    lerp_forces = _finite([r["lerp_metrics"]["mean_force_norm_avg"] for r in all_results])

    summary = {
        "n_reactions": len(all_results),
        "n_steps_per_trajectory": n_steps,
        "total_steps_evaluated": total_steps,
        "model": {
            "steps_succeeded": model_successes,
            "steps_failed": model_failures,
            "avg_validity_rate": model_successes / total_steps if total_steps > 0 else 0.0,
            "failure_reasons": model_failure_reasons,
            "n_reactions_with_energy_data": len(model_smoothness),
            "avg_energy_smoothness": float(np.mean(model_smoothness)) if model_smoothness else None,
            "median_energy_smoothness": float(np.median(model_smoothness)) if model_smoothness else None,
            "avg_mean_force_norm": float(np.mean(model_forces)) if model_forces else None,
        },
        "lerp_control": {
            "steps_succeeded": lerp_successes,
            "steps_failed": lerp_failures,
            "avg_validity_rate": lerp_successes / total_steps if total_steps > 0 else 0.0,
            "failure_reasons": lerp_failure_reasons,
            "n_reactions_with_energy_data": len(lerp_smoothness),
            "avg_energy_smoothness": float(np.mean(lerp_smoothness)) if lerp_smoothness else None,
            "median_energy_smoothness": float(np.median(lerp_smoothness)) if lerp_smoothness else None,
            "avg_mean_force_norm": float(np.mean(lerp_forces)) if lerp_forces else None,
        },
    }

    # -- Log summary to wandb --
    wandb.log({
        "summary/model_avg_validity": summary["model"]["avg_validity_rate"],
        "summary/lerp_avg_validity": summary["lerp_control"]["avg_validity_rate"],
        "summary/model_steps_succeeded": model_successes,
        "summary/model_steps_failed": model_failures,
        "summary/lerp_steps_succeeded": lerp_successes,
        "summary/lerp_steps_failed": lerp_failures,
        "summary/model_median_energy_smoothness": summary["model"]["median_energy_smoothness"],
        "summary/lerp_median_energy_smoothness": summary["lerp_control"]["median_energy_smoothness"],
        "summary/model_avg_mean_force": summary["model"]["avg_mean_force_norm"],
        "summary/lerp_avg_mean_force": summary["lerp_control"]["avg_mean_force_norm"],
    })

    # Log failure breakdown as a wandb table
    failure_table = wandb.Table(columns=["source", "failure_stage", "count"])
    for stage, count in sorted(model_failure_reasons.items()):
        failure_table.add_data("model", stage, count)
    for stage, count in sorted(lerp_failure_reasons.items()):
        failure_table.add_data("lerp", stage, count)
    wandb.log({"failure_breakdown": failure_table})

    wandb.log({"reactions": reaction_table})

    # -- Save JSON --
    results_path = save_dir / "diagnostic_results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_reaction": all_results}, f, indent=2)

    summary_path = save_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
