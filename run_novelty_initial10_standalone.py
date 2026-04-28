#!/usr/bin/env python3
"""
Single-file standalone runner for EGBO_Novelty_v1 on the initial 10 benchmark problems.

No local module imports are required (only external Python packages).

Defaults (requested):
- qNEHVI candidate pool per batch: 8
- NSGA-III candidate pool per batch: 72
- Evaluated batch size: 8
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Threading defaults for CPU stability/performance
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# External deps
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ1, DTLZ2, DTLZ3, MW7, ZDT1, ZDT2, ZDT3
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination
from pymoo.util.ref_dirs import get_reference_directions

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tkwargs = {"dtype": torch.double, "device": DEVICE}


FIRST_10_PROBLEMS = [
    "DTLZ1", "DTLZ2_5obj", "DTLZ3",
    "ZDT1", "ZDT2", "ZDT3", "ZDT4",
    "MW3", "MW5", "MW7",
]


# -----------------------------------------------------------------------------
# Problem definitions (self-contained)
# -----------------------------------------------------------------------------
class MWBase:
    @staticmethod
    def g3(X, n_obj):
        contrib = 2.0 * torch.pow(
            X[..., n_obj - 1:] + (X[..., n_obj - 2:-1] - 0.5) * (X[..., n_obj - 2:-1] - 0.5) - 1.0,
            2.0,
        )
        return 1 + contrib.sum(axis=1)

    @staticmethod
    def LA1(A, B, C, D, theta):
        return A * torch.pow(torch.sin(B * np.pi * torch.pow(theta, C)), D)

    @staticmethod
    def LA2(A, B, C, D, theta):
        return A * torch.pow(torch.sin(B * torch.pow(theta, C)), D)


class Problem_MW3(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 2
    ref_point = torch.tensor([1.6, 1.6], **tkwargs)
    bounds = torch.vstack([torch.zeros(8, **tkwargs), torch.ones(8, **tkwargs)])

    def evaluate(self, X):
        g = MWBase.g3(X, n_obj=2)
        f1 = X[:, 0]
        f2 = g * (1.0 - f1 / g)
        c1 = f1 + f2 - 1.5 - MWBase.LA1(0.45, 0.75, 1.0, 6.0, np.sqrt(2.0) * f2 - np.sqrt(2.0) * f1)
        c2 = 0.5 - f1 - f2 + MWBase.LA1(0.3, 0.75, 1.0, 2.0, np.sqrt(2.0) * f2 - np.sqrt(2.0) * f1)
        return torch.stack([-f1, -f2], dim=-1), torch.stack([c1, c2], dim=-1)


class Problem_MW5(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 3
    ref_point = torch.tensor([1.6, 1.6], **tkwargs)
    bounds = torch.vstack([torch.zeros(8, **tkwargs), torch.ones(8, **tkwargs)])

    def __init__(self):
        super().__init__()
        self.constraint_tightness = 1.0
        self.constraint_shift = 0.0

    def evaluate(self, X):
        g1 = MWBase.g3(X, n_obj=2)
        f1 = g1 * X[:, 0]
        f2 = g1 * torch.sqrt(1.0 - torch.pow(f1 / g1, 2.0))
        atan = torch.arctan(f2 / (f1 + 1e-12))
        c1 = (f1 ** 2) + (f2 ** 2) - torch.pow(2.0 - MWBase.LA2(0.2, 2.0, 1.0, 1.0, atan), 2.0)
        t = 0.5 * np.pi - 2 * torch.abs(atan - 0.25 * np.pi)
        c2 = torch.pow(1.3 + MWBase.LA2(0.5, 6.0, 3.0, 1.0, t), 2.0) - (f1 ** 2) - (f2 ** 2)
        c3 = torch.pow(0.7 - MWBase.LA2(0.45, 6.0, 3.0, 1.0, t), 2.0) - (f1 ** 2) - (f2 ** 2)

        c1 = c1 * self.constraint_tightness + self.constraint_shift
        c2 = c2 * self.constraint_tightness + self.constraint_shift
        c3 = c3 * self.constraint_tightness + self.constraint_shift
        return torch.stack([-f1, -f2], dim=-1), torch.stack([c1, c2, c3], dim=-1)


class Problem_MW7(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 2
    ref_point = torch.tensor([1.2, 1.2], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = MW7(dim=8, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X), -self.base.evaluate_slack(X)


class Problem_ZDT1(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 0
    ref_point = torch.tensor([11.0, 11.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = ZDT1(dim=8, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X)


class Problem_ZDT2(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 0
    ref_point = torch.tensor([11.0, 11.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = ZDT2(dim=8, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X)


class Problem_ZDT3(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 0
    ref_point = torch.tensor([11.0, 11.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = ZDT3(dim=8, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X)


class Problem_ZDT4(torch.nn.Module):
    n_var = 8
    n_obj = 2
    n_constr = 0
    # ref_point must dominate all stuck-landscape solutions.
    # ZDT4's multimodal g function gives f2 ≈ 75-160 when trapped in local optima.
    # [11.0, 11.0] (old) caused HV=0 because f2 > 11 always.
    # [1.1, 200.0]: f1 anti-ideal (just outside [0,1]) and f2 anti-ideal (> worst observed ~160).
    ref_point = torch.tensor([1.1, 200.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.bounds = torch.vstack([torch.zeros(self.n_var, **tkwargs), torch.ones(self.n_var, **tkwargs)])

    def evaluate(self, X):
        x1 = X[..., 0]
        x_rest = 10.0 * X[..., 1:] - 5.0
        g = 1.0 + 10.0 * (self.n_var - 1) + torch.sum(x_rest ** 2 - 10.0 * torch.cos(4.0 * np.pi * x_rest), dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.sqrt(f1 / g))
        return torch.stack([-f1, -f2], dim=-1)


class Problem_DTLZ1(torch.nn.Module):
    n_var = 8
    n_obj = 3
    n_constr = 0
    ref_point = torch.tensor([400.0, 400.0, 400.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = DTLZ1(dim=8, num_objectives=3, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X)


class Problem_DTLZ3(torch.nn.Module):
    n_var = 8
    n_obj = 3
    n_constr = 0
    ref_point = torch.tensor([400.0, 400.0, 400.0], **tkwargs)

    def __init__(self):
        super().__init__()
        self.base = DTLZ3(dim=8, num_objectives=3, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds

    def evaluate(self, X):
        return self.base(X)


class Problem_DTLZ2_5obj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_var = 14
        self.n_obj = 5
        self.n_constr = 0
        self.base = DTLZ2(dim=14, num_objectives=5, negate=True).to(**tkwargs)
        self.bounds = self.base.bounds
        self.ref_point = torch.tensor([1.1] * 5, **tkwargs)

    def evaluate(self, X):
        return self.base(X)


PROBLEMS = {
    "DTLZ1": (Problem_DTLZ1(), False),
    "DTLZ2_5obj": (Problem_DTLZ2_5obj(), False),
    "DTLZ3": (Problem_DTLZ3(), False),
    "ZDT1": (Problem_ZDT1(), False),
    "ZDT2": (Problem_ZDT2(), False),
    "ZDT3": (Problem_ZDT3(), False),
    "ZDT4": (Problem_ZDT4(), False),
    "MW3": (Problem_MW3(), True),
    "MW5": (Problem_MW5(), True),
    "MW7": (Problem_MW7(), True),
}


# -----------------------------------------------------------------------------
# Optimizer helpers
# -----------------------------------------------------------------------------
def fit_gpytorch_mll_fast(mll):
    return fit_gpytorch_mll(mll, max_retries=1, options={"maxiter": 50})


def get_reference_directions_cached(method, n_obj, n_points, seed=None):
    try:
        return get_reference_directions(method, n_obj, n_points, seed=seed)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n_points, n_obj))


def select_candidate_subset_novelty(train_x_gp, candidates_gp, acq_vals, batch_size, merit_weight=0.7):
    merit_weight = float(np.clip(merit_weight, 0.0, 1.0))

    acq = np.asarray(acq_vals, dtype=float)
    finite = np.isfinite(acq)
    merit_norm = np.zeros_like(acq, dtype=float)
    if finite.any():
        lo = float(np.min(acq[finite]))
        hi = float(np.max(acq[finite]))
        if hi - lo > 1e-12:
            merit_norm[finite] = (acq[finite] - lo) / (hi - lo)
        else:
            merit_norm[finite] = 1.0

    Xc = candidates_gp.detach()
    Xt = train_x_gp.detach()
    selected = []
    remaining = set(range(Xc.shape[0]))

    while len(selected) < batch_size and remaining:
        rem_idx = np.array(sorted(remaining), dtype=int)
        base = Xt if len(selected) == 0 else torch.cat([Xt, Xc[np.asarray(selected, dtype=int)]], dim=0)
        nov = torch.cdist(Xc[rem_idx], base).min(dim=1).values.detach().cpu().numpy()
        n_lo, n_hi = float(np.min(nov)), float(np.max(nov))
        nov_norm = np.ones_like(nov) if n_hi - n_lo <= 1e-12 else (nov - n_lo) / (n_hi - n_lo)

        score = merit_weight * merit_norm[rem_idx] + (1.0 - merit_weight) * nov_norm
        pick = int(rem_idx[int(np.argmax(score))])
        selected.append(pick)
        remaining.remove(pick)

    return np.asarray(selected[:batch_size], dtype=int)


def _mk_models(train_x_gp, train_y):
    models = [SingleTaskGP(train_x_gp, train_y[..., i:i + 1], outcome_transform=Standardize(m=1)) for i in range(train_y.shape[-1])]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def optimize_novelty_unconstrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                   random_state=0, noise=0.0, verbose=False,
                                   qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
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
            qbo_x, _ = optimize_acqf(acq, bounds=standard_bounds, q=qnehvi_candidates,
                                     num_restarts=1, raw_samples=8,
                                     options={"batch_limit": max(1, qnehvi_candidates), "maxiter": 20})

        pareto_mask = is_non_dominated(train_obj)
        pareto_y = -train_obj[pareto_mask]
        pareto_x = train_x_gp[pareto_mask]

        algo = UNSGA3(
            pop_size=evo_candidates,
            ref_dirs=get_reference_directions_cached("energy", problem.n_obj, evo_candidates, seed=random_state),
            sampling=pareto_x.cpu().numpy(),
        )
        pm = PymooProblem(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=0,
                          xl=np.zeros(problem.n_var), xu=np.ones(problem.n_var))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop.set("F", pareto_y.cpu().numpy())
        algo.tell(infills=pop)
        ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)

        candidates = torch.cat([qbo_x, ea_x])
        source_labels = ["qnehvi"] * qbo_x.shape[0] + ["nsga3"] * ea_x.shape[0]

        acq_vals = [float(acq(candidates[i].unsqueeze(0)).item()) for i in range(candidates.shape[0])]
        sel_idx = select_candidate_subset_novelty(train_x_gp, candidates, acq_vals, BATCH_SIZE, merit_weight=merit_weight)
        new_x_gp = torch.tensor(candidates.cpu().numpy()[sel_idx], **tkwargs)

        sel_sources = [source_labels[int(i)] for i in sel_idx]
        sel_bo = sum(1 for s in sel_sources if s == "qnehvi")
        sel_ea = len(sel_sources) - sel_bo
        if new_x_gp.shape[0] > 1:
            min_pair = float(torch.pdist(new_x_gp).min().item())
        else:
            min_pair = float("nan")
        mean_min_train = float(torch.cdist(new_x_gp, train_x_gp).min(dim=1).values.mean().item())

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
            f"min_pair={min_pair:.3f} mean_min_train={mean_min_train:.3f} | "
            f"Time={time.time()-tb:>5.2f}s",
            flush=True,
        )

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), None, all_source_labels, all_acq_values


def optimize_novelty_constrained(problem, ref_point, initial_x, N_BATCH, BATCH_SIZE,
                                 random_state=0, noise=0.0, verbose=False,
                                 qnehvi_candidates=8, evo_candidates=72, merit_weight=0.7):
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

    def constraints():
        return [lambda Z, i=i: Z[..., i] for i in range(problem.n_obj, problem.n_obj + problem.n_constr)]

    for batch in range(1, N_BATCH + 1):
        tb = time.time()
        fit_gpytorch_mll_fast(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.n_obj))),
            constraints=constraints(),
            prune_baseline=True,
            cache_pending=True,
        )

        qbo_x, _ = optimize_acqf(acq, bounds=standard_bounds, q=qnehvi_candidates,
                                 num_restarts=1, raw_samples=8,
                                 options={"batch_limit": max(1, qnehvi_candidates), "maxiter": 20})

        pareto_mask = is_non_dominated(train_obj)
        pareto_y = -train_obj[pareto_mask]
        pareto_x = train_x_gp[pareto_mask]
        pareto_con = train_con[pareto_mask]

        algo = UNSGA3(
            pop_size=evo_candidates,
            ref_dirs=get_reference_directions_cached("energy", problem.n_obj, evo_candidates, seed=random_state),
            sampling=pareto_x.cpu().numpy(),
        )
        pm = PymooProblem(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=problem.n_constr,
                          xl=np.zeros(problem.n_var), xu=np.ones(problem.n_var))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop.set("F", pareto_y.cpu().numpy())
        pop.set("G", pareto_con.cpu().numpy())
        algo.tell(infills=pop)
        ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)

        candidates = torch.cat([qbo_x, ea_x])
        source_labels = ["qnehvi"] * qbo_x.shape[0] + ["nsga3"] * ea_x.shape[0]
        acq_vals = [float(acq(candidates[i].unsqueeze(0)).item()) for i in range(candidates.shape[0])]

        sel_idx = select_candidate_subset_novelty(train_x_gp, candidates, acq_vals, BATCH_SIZE, merit_weight=merit_weight)
        new_x_gp = torch.tensor(candidates.cpu().numpy()[sel_idx], **tkwargs)

        sel_sources = [source_labels[int(i)] for i in sel_idx]
        sel_bo = sum(1 for s in sel_sources if s == "qnehvi")
        sel_ea = len(sel_sources) - sel_bo
        if new_x_gp.shape[0] > 1:
            min_pair = float(torch.pdist(new_x_gp).min().item())
        else:
            min_pair = float("nan")
        mean_min_train = float(torch.cdist(new_x_gp, train_x_gp).min(dim=1).values.mean().item())

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
            f"min_pair={min_pair:.3f} mean_min_train={mean_min_train:.3f} | "
            f"Time={time.time()-tb:>5.2f}s",
            flush=True,
        )

    total_time = time.time() - t0
    return hvs, torch.hstack([train_x, train_obj, train_con]).cpu().numpy(), total_time, all_sel_idx, train_obj.cpu().numpy(), train_con.cpu().numpy(), all_source_labels, all_acq_values


def generate_initial_samples(problem, n_trials=10, n_initial=18, seed=42):
    torch.manual_seed(seed)
    return torch.rand(n_trials, n_initial, problem.n_var).cpu().numpy()


def load_initial_samples_from_existing_runs(
    problem_name,
    n_trials,
    n_initial,
    n_var,
    source_root,
    source_algorithm=None,
    strict=False,
):
    """Load initial coordinates from existing trial train_data CSVs.

    Expected layout:
      source_root/problem_name/<algorithm>/trial_000_train_data.csv

    We extract the first `n_initial` rows and first `n_var` columns.
    """
    source_root = Path(source_root)
    problem_root = source_root / problem_name
    if not problem_root.exists():
        raise FileNotFoundError(f"Problem folder not found in reuse path: {problem_root}")

    if source_algorithm:
        algo_dirs = [problem_root / source_algorithm]
    else:
        algo_dirs = sorted([d for d in problem_root.iterdir() if d.is_dir()])

    if not algo_dirs:
        raise FileNotFoundError(f"No algorithm folders found under {problem_root}")

    initial = np.full((n_trials, n_initial, n_var), np.nan, dtype=float)
    loaded = 0

    for trial in range(n_trials):
        trial_file = None
        for ad in algo_dirs:
            candidate = ad / f"trial_{trial:03d}_train_data.csv"
            if candidate.exists():
                trial_file = candidate
                break

        if trial_file is None:
            if strict:
                raise FileNotFoundError(
                    f"Missing trial_{trial:03d}_train_data.csv for problem {problem_name} "
                    f"in {problem_root}"
                )
            continue

        arr = np.loadtxt(trial_file, delimiter=",")
        arr = np.atleast_2d(arr)
        if arr.shape[0] < n_initial or arr.shape[1] < n_var:
            if strict:
                raise ValueError(
                    f"Insufficient shape in {trial_file}: got {arr.shape}, "
                    f"need at least ({n_initial}, {n_var})"
                )
            continue

        initial[trial] = arr[:n_initial, :n_var]
        loaded += 1

    if strict and loaded < n_trials:
        raise RuntimeError(
            f"Strict reuse requested but only loaded {loaded}/{n_trials} trials for {problem_name}"
        )

    return initial, loaded


def run_benchmark(problems, n_trials, n_batch, batch_size, qnehvi_candidates, evo_candidates,
                  merit_weight, noise, output_dir, verbose,
                  reuse_initial_from=None, reuse_algorithm=None, strict_reuse_initial=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_settings = {
        "batch_size": int(batch_size),
        "qnehvi_candidates": int(qnehvi_candidates),
        "evo_candidates": int(evo_candidates),
        "merit_weight": float(merit_weight),
        "noise": float(noise),
        "n_batch": int(n_batch),
    }

    results = {}

    for pname in problems:
        problem, is_constrained = PROBLEMS[pname]
        print(f"\n{'='*70}\nProblem: {pname} | constrained={is_constrained}\n{'='*70}", flush=True)

        problem_dir = output_dir / pname / "EGBO_Novelty_v1"
        problem_dir.mkdir(parents=True, exist_ok=True)

        if reuse_initial_from:
            try:
                initial_x_array, loaded = load_initial_samples_from_existing_runs(
                    problem_name=pname,
                    n_trials=n_trials,
                    n_initial=18,
                    n_var=problem.n_var,
                    source_root=reuse_initial_from,
                    source_algorithm=reuse_algorithm,
                    strict=strict_reuse_initial,
                )
                print(
                    f"  ↪ Reused initial coordinates from {reuse_initial_from} "
                    f"(loaded {loaded}/{n_trials} trials)",
                    flush=True,
                )
                if loaded < n_trials and not strict_reuse_initial:
                    fallback = generate_initial_samples(problem, n_trials=n_trials, n_initial=18, seed=42)
                    missing = np.isnan(initial_x_array).any(axis=(1, 2))
                    initial_x_array[missing] = fallback[missing]
                    print(
                        f"  ⚠ Filled {int(missing.sum())} missing trial(s) with generated initials",
                        flush=True,
                    )
            except Exception as exc:
                if strict_reuse_initial:
                    raise
                print(f"  ⚠ Reuse failed ({type(exc).__name__}: {exc}). Falling back to generated initials.", flush=True)
                initial_x_array = generate_initial_samples(problem, n_trials=n_trials, n_initial=18, seed=42)
        else:
            initial_x_array = generate_initial_samples(problem, n_trials=n_trials, n_initial=18, seed=42)

        hv_list = []
        times = []
        for trial in range(n_trials):
            ckpt = problem_dir / f"trial_{trial:03d}_hv.csv"
            meta_ckpt = problem_dir / f"trial_{trial:03d}_run_config.json"
            reuse_checkpoint = False
            if ckpt.exists() and meta_ckpt.exists():
                try:
                    with open(meta_ckpt, "r") as f:
                        prev_cfg = json.load(f)
                    reuse_checkpoint = all(prev_cfg.get(k) == v for k, v in run_settings.items())
                except Exception:
                    reuse_checkpoint = False
            elif ckpt.exists() and not meta_ckpt.exists():
                print(
                    f"  ⚠ Trial {trial+1}/{n_trials}: existing checkpoint has no config metadata; rerunning to avoid cross-weight contamination",
                    flush=True,
                )

            if reuse_checkpoint:
                hv = np.loadtxt(ckpt, delimiter=",")
                hv_list.append(hv)
                times.append(0.0)
                print(f"  ⏩ Trial {trial+1}/{n_trials}: already complete", flush=True)
                continue
            elif ckpt.exists():
                print(
                    f"  ↻ Trial {trial+1}/{n_trials}: checkpoint settings differ from current run; recomputing",
                    flush=True,
                )

            print(f"  ▶ Trial {trial+1}/{n_trials}", flush=True)
            initial_x = torch.tensor(initial_x_array[trial], **tkwargs)

            optimizer = optimize_novelty_constrained if is_constrained else optimize_novelty_unconstrained
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

            hv_list.append(hv)
            times.append(total_time)

            np.savetxt(problem_dir / f"trial_{trial:03d}_hv.csv", hv, delimiter=",")
            np.savetxt(problem_dir / f"trial_{trial:03d}_train_data.csv", train, delimiter=",")
            np.savetxt(problem_dir / f"trial_{trial:03d}_selection_indices.csv", np.asarray(sel_idx), delimiter=",", fmt="%d")
            with open(problem_dir / f"trial_{trial:03d}_run_config.json", "w") as f:
                json.dump(run_settings, f, indent=2)
            with open(problem_dir / f"trial_{trial:03d}_source_labels.json", "w") as f:
                json.dump(source_labels, f, indent=2)
            with open(problem_dir / f"trial_{trial:03d}_acquisition_values.json", "w") as f:
                json.dump(acq_values, f, indent=2)

        np.savetxt(problem_dir / "all_trials_hv.csv", np.asarray(hv_list), delimiter=",")

        summary = {
            "problem": pname,
            "algorithm": "EGBO_Novelty_v1",
            "n_trials": int(n_trials),
            "final_hv_mean": float(np.mean([h[-1] for h in hv_list])) if hv_list else 0.0,
            "final_hv_std": float(np.std([h[-1] for h in hv_list])) if hv_list else 0.0,
            "avg_time": float(np.mean(times)) if times else 0.0,
            "run_settings": run_settings,
        }
        with open(problem_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        results[pname] = summary
        print(f"  ✓ {pname} done | mean final HV={summary['final_hv_mean']:.4f}", flush=True)

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted. Results: {output_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Single-file Novelty EGBO runner for initial 10 problems")
    p.add_argument("--problems", type=str, default="first10", help='"first10", "all", or comma-separated list')
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--qnehvi-candidates", type=int, default=8)
    p.add_argument("--evo-candidates", type=int, default=72)
    p.add_argument("--merit-weight", type=float, default=0.7)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-name", type=str, default="novelty_initial10_singlefile")
    p.add_argument("--reuse-initial-from", type=str, default=None,
                   help="Path to existing benchmark results root to reuse initial coordinates")
    p.add_argument("--reuse-algorithm", type=str, default=None,
                   help="Algorithm folder name under each problem (e.g., EGBO, Traditional_NEHVI)")
    p.add_argument("--strict-reuse-initial", action="store_true",
                   help="Fail if any trial/problem initial coordinates cannot be reused")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {DEVICE}", flush=True)

    if args.quick:
        n_trials, n_batch = 2, 6
        print("Quick mode: 2 trials, 6 batches", flush=True)
    else:
        n_trials, n_batch = args.trials, args.batches

    if args.problems.lower() == "first10":
        problems = FIRST_10_PROBLEMS
    elif args.problems.lower() == "all":
        problems = list(PROBLEMS.keys())
    else:
        problems = [x.strip() for x in args.problems.split(",") if x.strip()]

    for p in problems:
        if p not in PROBLEMS:
            raise ValueError(f"Unknown problem: {p}. Available: {list(PROBLEMS.keys())}")

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"benchmark_results_{args.output_name}_{ts}"

    print("\nConfig:", flush=True)
    print(f"  Problems: {', '.join(problems)}", flush=True)
    print(f"  Trials: {n_trials}", flush=True)
    print(f"  Batches: {n_batch}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  qNEHVI candidates: {args.qnehvi_candidates}", flush=True)
    print(f"  NSGA-III candidates: {args.evo_candidates}", flush=True)
    print(f"  Merit weight: {args.merit_weight}", flush=True)
    if args.reuse_initial_from:
        print(f"  Reuse initials from: {args.reuse_initial_from}", flush=True)
        if args.reuse_algorithm:
            print(f"  Reuse algorithm folder: {args.reuse_algorithm}", flush=True)
        print(f"  Strict reuse: {args.strict_reuse_initial}", flush=True)

    run_benchmark(
        problems=problems,
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


if __name__ == "__main__":
    main()
