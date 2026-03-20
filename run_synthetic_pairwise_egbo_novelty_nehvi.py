#!/usr/bin/env python3
"""
Pairwise matched runner for:
- EGBO (acquisition top-k selection)
- EGBO_Novelty_v1 (novelty-aware selection)
- Traditional_NEHVI (qNEHVI only)

Goal: isolate improvement from novelty selector under matched settings.

Usage example:
python run_synthetic_pairwise_egbo_novelty_nehvi.py \
  --problems MW5,ZDT1 \
  --trials 3 --batches 6 --batch-size 8 \
  --qnehvi-candidates 8 --evo-candidates 72 \
  --reuse-initial-from benchmark_results_supplementary_ea_only_phase2_full \
  --reuse-algorithm EA_UNSGA3 --strict-reuse-initial
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination

from run_novelty_initial10_standalone import (
    FIRST_10_PROBLEMS,
    PROBLEMS,
    fit_gpytorch_mll_fast,
    generate_initial_samples,
    get_reference_directions_cached,
    load_initial_samples_from_existing_runs,
    optimize_novelty_constrained,
    optimize_novelty_unconstrained,
    select_candidate_subset_novelty,
    tkwargs,
)


def _mk_models(train_x_gp, train_y):
    models = [
        SingleTaskGP(train_x_gp, train_y[..., i:i + 1], outcome_transform=Standardize(m=1))
        for i in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def _constraint_fns(n_obj, n_constr):
    return [lambda Z, i=i: Z[..., i] for i in range(n_obj, n_obj + n_constr)]


def optimize_egbo_matched_unconstrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                        random_state=0, noise=0.0, verbose=False,
                                        qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
    del merit_weight
    torch.manual_seed(random_state)
    t0 = time.time()

    hv_calc = Hypervolume(ref_point=-ref_point)
    train_x = initial_x
    with torch.no_grad():
        train_obj = problem.evaluate(train_x)
    train_obj_noisy = train_obj + noise * torch.randn_like(train_obj)

    hvs, all_sel_idx, all_source_labels, all_acq_values = [], [], [], []

    standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
    standard_bounds[1] = 1
    train_x_gp = normalize(train_x, problem.bounds)
    model, mll = _mk_models(train_x_gp, train_obj_noisy)

    for batch in range(1, N_BATCH + 1):
        tb = time.time()
        fit_gpytorch_mll_fast(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.n_obj))),
            constraints=None,
            prune_baseline=True,
            cache_pending=True,
        )

        if problem.n_obj >= 5:
            qbo_x = draw_sobol_samples(bounds=standard_bounds, n=qnehvi_candidates, q=1).squeeze(-2)
        else:
            qbo_x, _ = optimize_acqf(
                acq,
                bounds=standard_bounds,
                q=qnehvi_candidates,
                num_restarts=1,
                raw_samples=8,
                options={"batch_limit": max(1, qnehvi_candidates), "maxiter": 20},
            )

        pareto_mask = is_non_dominated(train_obj)
        pareto_y = -train_obj[pareto_mask]
        pareto_x = train_x_gp[pareto_mask]

        algo = UNSGA3(
            pop_size=evo_candidates,
            ref_dirs=get_reference_directions_cached("energy", problem.n_obj, evo_candidates, seed=random_state),
            sampling=pareto_x.cpu().numpy(),
        )
        pm = PymooProblem(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=0,
            xl=np.zeros(problem.n_var),
            xu=np.ones(problem.n_var),
        )
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop.set("F", pareto_y.cpu().numpy())
        algo.tell(infills=pop)
        ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)

        candidates = torch.cat([qbo_x, ea_x])
        source_labels = ["qnehvi"] * qbo_x.shape[0] + ["nsga3"] * ea_x.shape[0]
        acq_vals = [float(acq(candidates[i].unsqueeze(0)).item()) for i in range(candidates.shape[0])]

        # Matched EGBO rule: top acquisition value only.
        sel_idx = np.argsort(np.asarray(acq_vals, dtype=float))[-BATCH_SIZE:]
        new_x_gp = torch.tensor(candidates.cpu().numpy()[sel_idx], **tkwargs)

        sel_sources = [source_labels[int(i)] for i in sel_idx]
        sel_bo = sum(1 for s in sel_sources if s == "qnehvi")
        sel_ea = len(sel_sources) - sel_bo

        new_x = unnormalize(new_x_gp.detach(), bounds=problem.bounds)
        with torch.no_grad():
            new_obj = problem.evaluate(new_x)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj + noise * torch.randn_like(new_obj)])

        hv = hv_calc.compute(train_obj[is_non_dominated(train_obj)])
        hvs.append(hv)

        all_sel_idx.append(sel_idx)
        all_source_labels.append(source_labels)
        all_acq_values.append(acq_vals)

        train_x_gp = normalize(train_x, problem.bounds)
        model, mll = _mk_models(train_x_gp, train_obj_noisy)

        print(
            f"  Batch {batch:>2}/{N_BATCH}: HV={hv:>6.3f} | "
            f"pool qNEHVI={qbo_x.shape[0]} NSGA3={ea_x.shape[0]} -> "
            f"selected qNEHVI={sel_bo} NSGA3={sel_ea} | "
            f"Time={time.time()-tb:>5.2f}s",
            flush=True,
        )

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), None, all_source_labels, all_acq_values


def optimize_egbo_matched_constrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                      random_state=0, noise=0.0, verbose=False,
                                      qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
    del merit_weight
    torch.manual_seed(random_state)
    t0 = time.time()

    hv_calc = Hypervolume(ref_point=-ref_point)
    train_x = initial_x
    with torch.no_grad():
        train_obj, train_con = problem.evaluate(train_x)
    train_obj_noisy = train_obj + noise * torch.randn_like(train_obj)
    train_con_noisy = train_con + noise * torch.randn_like(train_con)

    hvs, all_sel_idx, all_source_labels, all_acq_values = [], [], [], []

    standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
    standard_bounds[1] = 1
    train_x_gp = normalize(train_x, problem.bounds)
    model, mll = _mk_models(train_x_gp, torch.cat([train_obj_noisy, train_con_noisy], dim=-1))

    for batch in range(1, N_BATCH + 1):
        tb = time.time()
        fit_gpytorch_mll_fast(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.n_obj))),
            constraints=_constraint_fns(problem.n_obj, problem.n_constr),
            prune_baseline=True,
            cache_pending=True,
        )

        qbo_x, _ = optimize_acqf(
            acq,
            bounds=standard_bounds,
            q=qnehvi_candidates,
            num_restarts=1,
            raw_samples=8,
            options={"batch_limit": max(1, qnehvi_candidates), "maxiter": 20},
        )

        pareto_mask = is_non_dominated(train_obj)
        pareto_y = -train_obj[pareto_mask]
        pareto_x = train_x_gp[pareto_mask]
        pareto_con = train_con[pareto_mask]

        algo = UNSGA3(
            pop_size=evo_candidates,
            ref_dirs=get_reference_directions_cached("energy", problem.n_obj, evo_candidates, seed=random_state),
            sampling=pareto_x.cpu().numpy(),
        )
        pm = PymooProblem(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=problem.n_constr,
            xl=np.zeros(problem.n_var),
            xu=np.ones(problem.n_var),
        )
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop.set("F", pareto_y.cpu().numpy())
        pop.set("G", pareto_con.cpu().numpy())
        algo.tell(infills=pop)
        ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)

        candidates = torch.cat([qbo_x, ea_x])
        source_labels = ["qnehvi"] * qbo_x.shape[0] + ["nsga3"] * ea_x.shape[0]
        acq_vals = [float(acq(candidates[i].unsqueeze(0)).item()) for i in range(candidates.shape[0])]

        sel_idx = np.argsort(np.asarray(acq_vals, dtype=float))[-BATCH_SIZE:]
        new_x_gp = torch.tensor(candidates.cpu().numpy()[sel_idx], **tkwargs)

        sel_sources = [source_labels[int(i)] for i in sel_idx]
        sel_bo = sum(1 for s in sel_sources if s == "qnehvi")
        sel_ea = len(sel_sources) - sel_bo

        new_x = unnormalize(new_x_gp.detach(), bounds=problem.bounds)
        with torch.no_grad():
            new_obj, new_con = problem.evaluate(new_x)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_con = torch.cat([train_con, new_con])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj + noise * torch.randn_like(new_obj)])
        train_con_noisy = torch.cat([train_con_noisy, new_con + noise * torch.randn_like(new_con)])

        feasible = (train_con <= 0).all(dim=-1)
        if feasible.any():
            fobj = train_obj[feasible]
            hv = hv_calc.compute(fobj[is_non_dominated(fobj)])
        else:
            hv = 0.0
        hvs.append(hv)

        all_sel_idx.append(sel_idx)
        all_source_labels.append(source_labels)
        all_acq_values.append(acq_vals)

        train_x_gp = normalize(train_x, problem.bounds)
        model, mll = _mk_models(train_x_gp, torch.cat([train_obj_noisy, train_con_noisy], dim=-1))

        print(
            f"  Batch {batch:>2}/{N_BATCH}: HV={hv:>6.3f} | "
            f"pool qNEHVI={qbo_x.shape[0]} NSGA3={ea_x.shape[0]} -> "
            f"selected qNEHVI={sel_bo} NSGA3={sel_ea} | "
            f"Time={time.time()-tb:>5.2f}s",
            flush=True,
        )

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj, train_con]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), train_con.cpu().numpy(), all_source_labels, all_acq_values


def optimize_nehvi_matched_unconstrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                         random_state=0, noise=0.0, verbose=False,
                                         qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
    del qnehvi_candidates, evo_candidates, merit_weight
    torch.manual_seed(random_state)
    t0 = time.time()

    hv_calc = Hypervolume(ref_point=-ref_point)
    train_x = initial_x
    with torch.no_grad():
        train_obj = problem.evaluate(train_x)
    train_obj_noisy = train_obj + noise * torch.randn_like(train_obj)

    hvs, all_sel_idx, all_source_labels, all_acq_values = [], [], [], []
    standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
    standard_bounds[1] = 1

    train_x_gp = normalize(train_x, problem.bounds)
    model, mll = _mk_models(train_x_gp, train_obj_noisy)

    for batch in range(1, N_BATCH + 1):
        tb = time.time()
        fit_gpytorch_mll_fast(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.n_obj))),
            prune_baseline=True,
            cache_pending=True,
        )

        new_x_gp, acq_val = optimize_acqf(
            acq,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=1,
            raw_samples=8,
            options={"batch_limit": max(1, BATCH_SIZE), "maxiter": 20},
        )

        new_x = unnormalize(new_x_gp.detach(), bounds=problem.bounds)
        with torch.no_grad():
            new_obj = problem.evaluate(new_x)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj + noise * torch.randn_like(new_obj)])

        hv = hv_calc.compute(train_obj[is_non_dominated(train_obj)])
        hvs.append(hv)

        all_sel_idx.append(list(range(BATCH_SIZE)))
        all_source_labels.append(["qnehvi"] * BATCH_SIZE)
        all_acq_values.append([float(acq_val.item())] * BATCH_SIZE)

        train_x_gp = normalize(train_x, problem.bounds)
        model, mll = _mk_models(train_x_gp, train_obj_noisy)

        print(f"  Batch {batch:>2}/{N_BATCH}: HV={hv:>6.3f} | qNEHVI-only | Time={time.time()-tb:>5.2f}s", flush=True)

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), None, all_source_labels, all_acq_values


def optimize_nehvi_matched_constrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                       random_state=0, noise=0.0, verbose=False,
                                       qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
    del qnehvi_candidates, evo_candidates, merit_weight
    torch.manual_seed(random_state)
    t0 = time.time()

    hv_calc = Hypervolume(ref_point=-ref_point)
    train_x = initial_x
    with torch.no_grad():
        train_obj, train_con = problem.evaluate(train_x)
    train_obj_noisy = train_obj + noise * torch.randn_like(train_obj)
    train_con_noisy = train_con + noise * torch.randn_like(train_con)

    hvs, all_sel_idx, all_source_labels, all_acq_values = [], [], [], []
    standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
    standard_bounds[1] = 1

    train_x_gp = normalize(train_x, problem.bounds)
    model, mll = _mk_models(train_x_gp, torch.cat([train_obj_noisy, train_con_noisy], dim=-1))

    for batch in range(1, N_BATCH + 1):
        tb = time.time()
        fit_gpytorch_mll_fast(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.n_obj))),
            constraints=_constraint_fns(problem.n_obj, problem.n_constr),
            prune_baseline=True,
            cache_pending=True,
        )

        new_x_gp, acq_val = optimize_acqf(
            acq,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=1,
            raw_samples=8,
            options={"batch_limit": max(1, BATCH_SIZE), "maxiter": 20},
        )

        new_x = unnormalize(new_x_gp.detach(), bounds=problem.bounds)
        with torch.no_grad():
            new_obj, new_con = problem.evaluate(new_x)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_con = torch.cat([train_con, new_con])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj + noise * torch.randn_like(new_obj)])
        train_con_noisy = torch.cat([train_con_noisy, new_con + noise * torch.randn_like(new_con)])

        feasible = (train_con <= 0).all(dim=-1)
        if feasible.any():
            fobj = train_obj[feasible]
            hv = hv_calc.compute(fobj[is_non_dominated(fobj)])
        else:
            hv = 0.0
        hvs.append(hv)

        all_sel_idx.append(list(range(BATCH_SIZE)))
        all_source_labels.append(["qnehvi"] * BATCH_SIZE)
        all_acq_values.append([float(acq_val.item())] * BATCH_SIZE)

        train_x_gp = normalize(train_x, problem.bounds)
        model, mll = _mk_models(train_x_gp, torch.cat([train_obj_noisy, train_con_noisy], dim=-1))

        print(f"  Batch {batch:>2}/{N_BATCH}: HV={hv:>6.3f} | qNEHVI-only | Time={time.time()-tb:>5.2f}s", flush=True)

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj, train_con]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), train_con.cpu().numpy(), all_source_labels, all_acq_values


ALGORITHMS = {
    "EGBO": {
        "constrained": optimize_egbo_matched_constrained,
        "unconstrained": optimize_egbo_matched_unconstrained,
    },
    "EGBO_Novelty_v1": {
        "constrained": optimize_novelty_constrained,
        "unconstrained": optimize_novelty_unconstrained,
    },
    "Traditional_NEHVI": {
        "constrained": optimize_nehvi_matched_constrained,
        "unconstrained": optimize_nehvi_matched_unconstrained,
    },
}


def run_benchmark(problems, algorithms, n_trials, n_batch, batch_size, qnehvi_candidates, evo_candidates,
                  merit_weight, noise, output_dir, verbose,
                  reuse_initial_from=None, reuse_algorithm=None, strict_reuse_initial=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pname in problems:
        problem, is_constrained = PROBLEMS[pname]
        print(f"\n{'='*70}\nProblem: {pname} | constrained={is_constrained}\n{'='*70}", flush=True)

        if reuse_initial_from:
            initial_x_array, loaded = load_initial_samples_from_existing_runs(
                problem_name=pname,
                n_trials=n_trials,
                n_initial=18,
                n_var=problem.n_var,
                source_root=reuse_initial_from,
                source_algorithm=reuse_algorithm,
                strict=strict_reuse_initial,
            )
            print(f"  ↪ Reused initial coordinates from {reuse_initial_from} (loaded {loaded}/{n_trials} trials)", flush=True)
            if loaded < n_trials and not strict_reuse_initial:
                fallback = generate_initial_samples(problem, n_trials=n_trials, n_initial=18, seed=42)
                missing = np.isnan(initial_x_array).any(axis=(1, 2))
                initial_x_array[missing] = fallback[missing]
        else:
            initial_x_array = generate_initial_samples(problem, n_trials=n_trials, n_initial=18, seed=42)

        for algo_name in algorithms:
            algo_dir = output_dir / pname / algo_name
            algo_dir.mkdir(parents=True, exist_ok=True)

            optimizer = ALGORITHMS[algo_name]["constrained" if is_constrained else "unconstrained"]
            hv_list, times = [], []

            print(f"\n--- {pname} | {algo_name} ---", flush=True)

            for trial in range(n_trials):
                hv_path = algo_dir / f"trial_{trial:03d}_hv.csv"
                if hv_path.exists():
                    hv = np.loadtxt(hv_path, delimiter=",")
                    hv_list.append(np.atleast_1d(hv))
                    times.append(0.0)
                    print(f"  ⏩ Trial {trial+1}/{n_trials}: already complete", flush=True)
                    continue

                initial_x = torch.tensor(initial_x_array[trial], **tkwargs)
                hv, train, total_time, sel_idx, clean_obj, clean_con, source_labels, acq_values = optimizer(
                    problem=problem,
                    ref_point=problem.ref_point,
                    initial_x=initial_x,
                    N_BATCH=n_batch,
                    BATCH_SIZE=batch_size,
                    random_state=trial,
                    noise=noise,
                    verbose=verbose,
                    qnehvi_candidates=qnehvi_candidates,
                    evo_candidates=evo_candidates,
                    merit_weight=merit_weight,
                )

                hv_list.append(np.asarray(hv))
                times.append(float(total_time))

                np.savetxt(algo_dir / f"trial_{trial:03d}_hv.csv", np.asarray(hv), delimiter=",")
                np.savetxt(algo_dir / f"trial_{trial:03d}_train_data.csv", train, delimiter=",")
                np.savetxt(algo_dir / f"trial_{trial:03d}_selection_indices.csv", np.asarray(sel_idx), delimiter=",", fmt="%d")
                np.save(algo_dir / f"trial_{trial:03d}_clean_obj.npy", clean_obj)
                if clean_con is not None:
                    np.save(algo_dir / f"trial_{trial:03d}_clean_con.npy", clean_con)
                with open(algo_dir / f"trial_{trial:03d}_source_labels.json", "w") as f:
                    json.dump(source_labels, f, indent=2)
                with open(algo_dir / f"trial_{trial:03d}_acquisition_values.json", "w") as f:
                    json.dump(acq_values, f, indent=2)

                print(f"  ✓ Trial {trial+1}/{n_trials} done | final HV={float(hv[-1]):.4f} | {total_time:.2f}s", flush=True)

            np.savetxt(algo_dir / "all_trials_hv.csv", np.asarray(hv_list), delimiter=",")
            summary = {
                "problem": pname,
                "algorithm": algo_name,
                "n_trials": int(n_trials),
                "final_hv_mean": float(np.mean([float(h[-1]) for h in hv_list])) if hv_list else 0.0,
                "final_hv_std": float(np.std([float(h[-1]) for h in hv_list])) if hv_list else 0.0,
                "avg_time": float(np.mean(times)) if times else 0.0,
                "matched_settings": {
                    "batch_size": int(batch_size),
                    "qnehvi_candidates": int(qnehvi_candidates),
                    "evo_candidates": int(evo_candidates),
                    "merit_weight": float(merit_weight),
                },
            }
            with open(algo_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Matched pairwise EGBO / Novelty / NEHVI runner")
    p.add_argument("--problems", type=str, default="MW5,ZDT1", help='"first10", "all", or comma-separated list')
    p.add_argument("--algorithms", type=str, default="EGBO,EGBO_Novelty_v1,Traditional_NEHVI")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--qnehvi-candidates", type=int, default=8)
    p.add_argument("--evo-candidates", type=int, default=72)
    p.add_argument("--merit-weight", type=float, default=0.7)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-name", type=str, default="pairwise_matched_egbo_novelty_nehvi")
    p.add_argument("--reuse-initial-from", type=str, default=None)
    p.add_argument("--reuse-algorithm", type=str, default=None)
    p.add_argument("--strict-reuse-initial", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        n_trials, n_batch = 2, 6
    else:
        n_trials, n_batch = args.trials, args.batches

    if args.problems.lower() == "first10":
        problems = FIRST_10_PROBLEMS
    elif args.problems.lower() == "all":
        problems = list(PROBLEMS.keys())
    else:
        problems = [x.strip() for x in args.problems.split(",") if x.strip()]

    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    for p in problems:
        if p not in PROBLEMS:
            raise ValueError(f"Unknown problem: {p}")
    for a in algorithms:
        if a not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {a}")

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"benchmark_results_{args.output_name}_{ts}"
    else:
        output_dir = args.output_dir

    print("\nMatched comparison config:", flush=True)
    print(f"  Problems: {', '.join(problems)}", flush=True)
    print(f"  Algorithms: {', '.join(algorithms)}", flush=True)
    print(f"  Trials: {n_trials} | Batches: {n_batch} | Batch size: {args.batch_size}", flush=True)
    print(f"  qNEHVI candidates: {args.qnehvi_candidates} | NSGA-III candidates: {args.evo_candidates}", flush=True)
    if args.reuse_initial_from:
        print(f"  Reuse initials from: {args.reuse_initial_from}", flush=True)
        if args.reuse_algorithm:
            print(f"  Reuse algorithm folder: {args.reuse_algorithm}", flush=True)
        print(f"  Strict reuse: {args.strict_reuse_initial}", flush=True)

    run_benchmark(
        problems=problems,
        algorithms=algorithms,
        n_trials=n_trials,
        n_batch=n_batch,
        batch_size=args.batch_size,
        qnehvi_candidates=args.qnehvi_candidates,
        evo_candidates=args.evo_candidates,
        merit_weight=args.merit_weight,
        noise=args.noise,
        output_dir=output_dir,
        verbose=args.verbose,
        reuse_initial_from=args.reuse_initial_from,
        reuse_algorithm=args.reuse_algorithm,
        strict_reuse_initial=args.strict_reuse_initial,
    )

    print(f"\nCompleted. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
