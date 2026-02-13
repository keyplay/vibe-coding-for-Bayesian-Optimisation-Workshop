"""
Multi-Fidelity Bayesian Optimisation using BOTorch (Hybrid Phased Strategy)
===========================================================================

This script optimises a simulated bioprocess using a phased multi-fidelity
Bayesian optimisation strategy powered by BOTorch.

Key components:
  - GP model: SingleTaskMultiFidelityGP (LinearTruncatedFidelityKernel)
  - Acquisition functions: qUpperConfidenceBound, qLogNoisyExpectedImprovement
  - Optimiser: optimize_acqf with FixedFeatureAcquisitionFunction

Strategy:
  Phase 0: Sobol initialisation at lowest fidelity (cost=10 each)
  Phase 1: Low-fidelity exploration with qUCB (cost=10 each)
  Phase 2: Mid-fidelity refinement with qLogNoisyEI (cost=575 each)
  Phase 3: High-fidelity confirmation experiments (cost=2100 each)

The class exposes self.X (N, 6) and self.Y (N,) after optimisation,
matching the hackathon submission interface.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from botorch import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples


# ======================== Configuration ========================

FIDELITY_COSTS = {0: 10, 1: 575, 2: 2100}
FIDELITY_DIM = 5  # Column index for fidelity (6th column)

# Discrete fidelity {0, 1, 2} <-> Normalised {0.0, 0.5, 1.0}
# BOTorch's LinearTruncatedFidelityKernel expects fidelity in [0, 1]
FID_MAP = {0: 0.0, 1: 0.5, 2: 1.0}
FID_MAP_INV = {v: k for k, v in FID_MAP.items()}

# Process variable bounds (5 design dimensions)
PROCESS_BOUNDS = np.array([
    [30, 40],   # Temperature [C]
    [6,  8],    # pH
    [0,  50],   # Feed 1 concentration [mM]
    [0,  50],   # Feed 2 concentration [mM]
    [0,  50],   # Feed 3 concentration [mM]
])

# Torch device and dtype configuration
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Bounds for the 5 design variables (used in optimize_acqf)
DESIGN_BOUNDS = torch.tensor(
    [[30, 6, 0, 0, 0],
     [40, 8, 50, 50, 50]], **tkwargs
)

# Full 6D bounds including normalised fidelity [0, 1]
FULL_BOUNDS = torch.tensor(
    [[30, 6, 0, 0, 0, 0.0],
     [40, 8, 50, 50, 50, 1.0]], **tkwargs
)


# ======================== GP Fitting ========================

def fit_mf_gp(train_X: torch.Tensor, train_Y: torch.Tensor):
    """
    Fit a SingleTaskMultiFidelityGP model.

    The model uses a LinearTruncatedFidelityKernel that automatically
    learns correlations between fidelity levels.

    Args:
        train_X: (N, 6) tensor, last column is normalised fidelity [0, 0.5, 1.0]
        train_Y: (N, 1) tensor of observations

    Returns:
        Fitted BOTorch GP model
    """
    model = SingleTaskMultiFidelityGP(
        train_X,
        train_Y,
        data_fidelities=[FIDELITY_DIM],
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(**tkwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ======================== BO Class ========================

class BO:
    """
    Multi-Fidelity Bayesian Optimisation with BOTorch (Hybrid Phased Strategy).

    Uses SingleTaskMultiFidelityGP as the surrogate model and BOTorch's
    optimize_acqf for acquisition function optimisation. Retains a phased
    budget allocation strategy across three fidelity levels.

    Attributes:
        X: np.ndarray of shape (N, 6) — all evaluated inputs
        Y: np.ndarray of shape (N,) — all observed titre values

    Args:
        obj_func:     Objective function (conduct_experiment), returns positive titre
        budget:       Total experimental budget
        n_initial:    Number of initial Sobol points (counted towards budget)
        phase1_cost:  Budget cap for Phase 1 (low-fidelity exploration)
        n_hf_reserve: Number of high-fidelity experiments to reserve for Phase 3
    """

    def __init__(
        self,
        obj_func,
        budget=15000,
        n_initial=7,
        phase1_cost=2000,
        n_hf_reserve=2,
    ):
        self.obj_func = obj_func
        self.budget = budget
        self.current_cost = 0
        self.n_initial = n_initial
        self.phase1_cost = phase1_cost
        self.n_hf_reserve = n_hf_reserve

        self.X = np.empty((0, 6))   # All evaluated inputs (N, 6)
        self.Y = np.empty(0)        # All observed outputs (N,)

    # -------------------- Helper Methods --------------------

    def _evaluate(self, X_new):
        """Evaluate a batch of points, store results, and deduct cost from budget."""
        X_new = np.atleast_2d(X_new)
        Y_new = np.array(self.obj_func(X_new))
        self.X = np.vstack([self.X, X_new])
        self.Y = np.append(self.Y, Y_new)
        for x in X_new:
            fid = int(np.round(np.clip(x[-1], 0, 2)))
            self.current_cost += FIDELITY_COSTS[fid]
        return Y_new

    def _to_torch(self, fidelity_filter=None):
        """
        Convert numpy data to torch tensors with normalised fidelity.

        Args:
            fidelity_filter: None for all data, or int to select a specific fidelity

        Returns:
            (train_X, train_Y) as torch tensors
        """
        if fidelity_filter is not None:
            mask = self.X[:, -1] == fidelity_filter
            X_np, Y_np = self.X[mask].copy(), self.Y[mask].copy()
        else:
            X_np, Y_np = self.X.copy(), self.Y.copy()

        # Normalise fidelity column: {0, 1, 2} -> {0.0, 0.5, 1.0}
        for orig, norm in FID_MAP.items():
            X_np[X_np[:, -1] == orig, -1] = norm

        train_X = torch.tensor(X_np, **tkwargs)
        train_Y = torch.tensor(Y_np, **tkwargs).unsqueeze(-1)
        return train_X, train_Y

    def _best_at_fidelity(self, fidelity):
        """Return the best observed parameters at a given fidelity (or None)."""
        mask = self.X[:, -1] == fidelity
        if not mask.any():
            return None
        return self.X[mask][np.argmax(self.Y[mask])]

    def _optimize_acqf_fixed_fid(self, model, acq_class, fid_norm, **acq_kwargs):
        """
        Optimise an acquisition function with fidelity fixed to a given value.

        Uses FixedFeatureAcquisitionFunction to pin the fidelity dimension,
        then optimises over the 5 design variables only.

        Args:
            model:     Fitted GP model
            acq_class: Acquisition function class (e.g. qUpperConfidenceBound)
            fid_norm:  Normalised fidelity value to fix (0.0, 0.5, or 1.0)
            **acq_kwargs: Additional keyword arguments for the acquisition function

        Returns:
            np.ndarray of shape (1, 6) with original discrete fidelity {0, 1, 2}
        """
        # Create the acquisition function
        acq_func = acq_class(model=model, **acq_kwargs)

        # Fix the fidelity dimension using FixedFeatureAcquisitionFunction
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq_function=acq_func,
            d=6,
            columns=[FIDELITY_DIM],
            values=[fid_norm],
        )

        candidates, _ = optimize_acqf(
            acq_function=fixed_acq,
            bounds=DESIGN_BOUNDS,
            q=1,
            num_restarts=10,
            raw_samples=2048,
            options={"maxiter": 200},
        )

        # Reconstruct the full 6D vector with original fidelity value
        fid_orig = FID_MAP_INV[fid_norm]
        x_np = candidates.detach().cpu().numpy().flatten()
        full_x = np.append(x_np, fid_orig)
        return full_x.reshape(1, -1)

    # -------------------- Main Optimisation Loop --------------------

    def run_optimization(self):
        torch.manual_seed(42)
        np.random.seed(42)

        # ========================================
        # Phase 0 — Initialisation (Sobol, fid=0, counted towards budget)
        # ========================================
        print("=" * 60)
        print("Phase 0: Initialisation (Sobol, fidelity=0) [BOTorch]")
        print("=" * 60)

        sobol_pts = draw_sobol_samples(
            bounds=DESIGN_BOUNDS, n=self.n_initial, q=1
        ).squeeze(1)  # (n_initial, 5)

        fid_col = torch.zeros(self.n_initial, 1, **tkwargs)
        X_init = torch.cat([sobol_pts, fid_col], dim=-1).cpu().numpy()
        self._evaluate(X_init)
        print(
            f"  {self.n_initial} initial points evaluated  |  "
            f"cost={self.current_cost}  |  best titre: {self.Y.max():.4f}"
        )

        # ========================================
        # Phase 1 — Low-fidelity exploration (fid=0, cost=10)
        # ========================================
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Low-fidelity exploration (budget cap: {self.phase1_cost}) [qUCB]")
        print("=" * 60)

        # 1a) Sobol batch exploration
        n_sobol = min(100, (self.phase1_cost // 2) // FIDELITY_COSTS[0])
        sobol_batch = draw_sobol_samples(
            bounds=DESIGN_BOUNDS, n=n_sobol, q=1
        ).squeeze(1)  # (n_sobol, 5)
        fid_col = torch.zeros(n_sobol, 1, **tkwargs)
        X_sobol = torch.cat([sobol_batch, fid_col], dim=-1).cpu().numpy()
        self._evaluate(X_sobol)
        print(
            f"  Sobol batch: {n_sobol} points  |  cost={self.current_cost}  |  "
            f"best={self.Y.max():.4f}"
        )

        # 1b) qUCB BO iterations (using fid=0 data only)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        bo_count = 0
        while self.current_cost + FIDELITY_COSTS[0] <= self.phase1_cost:
            try:
                train_X, train_Y = self._to_torch(fidelity_filter=0)
                model = fit_mf_gp(train_X, train_Y)
            except Exception as e:
                print(f"  GP fitting failed: {e}")
                break

            next_x = self._optimize_acqf_fixed_fid(
                model, qUpperConfidenceBound,
                fid_norm=FID_MAP[0],
                sampler=sampler,
                beta=4.0,
            )
            self._evaluate(next_x)
            bo_count += 1

        mask_0 = self.X[:, -1] == 0
        print(
            f"  qUCB-BO: {bo_count} iters  |  fid-0 total: {int(mask_0.sum())}  |  "
            f"cost={self.current_cost}  |  best fid-0: {self.Y[mask_0].max():.4f}"
        )

        # ========================================
        # Phase 2 — Mid-fidelity BO refinement (fid=1, cost=575)
        # ========================================
        hf_budget = self.n_hf_reserve * FIDELITY_COSTS[2]
        phase2_limit = self.budget - hf_budget
        print(f"\n{'=' * 60}")
        print(
            f"Phase 2: Mid-fidelity refinement [qLogNoisyEI] "
            f"(budget cap: {phase2_limit}, remaining: {phase2_limit - self.current_cost})"
        )
        print("=" * 60)

        # 2a) Seed mid-fidelity with top low-fidelity results
        #     Fid=0 feeds F2 at t=60, which sits between fid=1's F1(t=40) and F2(t=80).
        #     We use fid=0's best F2 as reference centre for F1(±5), F2(±5), F3(±10).
        mask_0 = self.X[:, -1] == 0
        n_top = min(3, int(mask_0.sum()))
        budget_for_seeds = phase2_limit - self.current_cost
        max_seeds = max(0, budget_for_seeds // FIDELITY_COSTS[1])

        if n_top > 0 and max_seeds > 0:
            top_idx = np.argsort(self.Y[mask_0])[-n_top:]
            top_points = self.X[mask_0][top_idx].copy()

            # Generate variants per top point:
            #   F1: centred on fid0's F2 ± 5 (t=60 ≈ t=40, transferable)
            #   F2: centred on fid0's F2 ± 5 (t=60 ≈ t=80, transferable)
            #   F3: centred on fid0's F2 ± 10 (t=120, farther but still informative)
            variants_per_top = max(1, min(3, max_seeds // n_top))
            seed_list = []
            for pt in top_points:
                f2_ref = pt[3]  # fid=0's best F2 value
                for _ in range(variants_per_top):
                    variant = pt.copy()
                    variant[2] = np.clip(f2_ref + np.random.uniform(-5, 5), 0, 50)  # F1: ref fid0-F2
                    variant[3] = np.clip(f2_ref + np.random.uniform(-5, 5), 0, 50)  # F2: ref fid0-F2
                    variant[4] = np.clip(f2_ref + np.random.uniform(-10, 10), 0, 50)  # F2: ref fid0-F2#np.random.uniform(0, 50)                               # F3: random
                    variant[-1] = 1  # Set to fid=1
                    seed_list.append(variant)
                    if len(seed_list) >= max_seeds:
                        break
                if len(seed_list) >= max_seeds:
                    break

            X_seed = np.array(seed_list)
            Y_seed = self._evaluate(X_seed)
            print(f"  Seed points ({len(X_seed)}, F1/F2/F3 all ref fid0-F2):")
            for i in range(len(X_seed)):
                print(
                    f"    #{i + 1} titre={Y_seed[i]:.4f}  |  "
                    f"F1={X_seed[i, 2]:.1f}, F2={X_seed[i, 3]:.1f}, F3={X_seed[i, 4]:.1f}"
                )

        # 2b) qLogNoisyEI BO iterations (using all data, MF-GP)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        bo_count = 0
        while self.current_cost + FIDELITY_COSTS[1] <= phase2_limit:
            try:
                train_X, train_Y = self._to_torch(fidelity_filter=None)
                model = fit_mf_gp(train_X, train_Y)
            except Exception as e:
                print(f"  GP fitting failed: {e}")
                break

            next_x = self._optimize_acqf_fixed_fid(
                model, qLogNoisyExpectedImprovement,
                fid_norm=FID_MAP[1],
                sampler=sampler,
                X_baseline=train_X,
            )
            Y_new = self._evaluate(next_x)
            bo_count += 1

            print(
                f"  Iter {bo_count}: titre={Y_new[0]:.4f}  |  "
                f"best={self.Y.max():.4f}  |  cost={self.current_cost}"
            )

        # ========================================
        # Phase 3 — High-fidelity confirmation (fid=2, cost=2100)
        # ========================================
        remaining = self.budget - self.current_cost
        n_hf = remaining // FIDELITY_COSTS[2]
        print(f"\n{'=' * 60}")
        print(f"Phase 3: High-fidelity confirmation ({n_hf} experiments) [BOTorch]")
        print("=" * 60)

        for i in range(n_hf):
            if self.current_cost + FIDELITY_COSTS[2] > self.budget:
                break

            if i == 0:
                # First HF experiment: directly evaluate the best fid-1 conditions
                ref = self._best_at_fidelity(1)
                if ref is None:
                    ref = self._best_at_fidelity(0)
                if ref is not None:
                    next_x = ref.copy().reshape(1, -1)
                    next_x[0, -1] = 2  # Set to fid=2
                else:
                    # Fallback: GP-guided selection
                    try:
                        train_X, train_Y = self._to_torch()
                        model = fit_mf_gp(train_X, train_Y)
                        next_x = self._optimize_acqf_fixed_fid(
                            model, qUpperConfidenceBound,
                            fid_norm=FID_MAP[2],
                            sampler=SobolQMCNormalSampler(
                                sample_shape=torch.Size([256])
                            ),
                            beta=0.1,
                        )
                    except Exception:
                        sobol_pt = draw_sobol_samples(
                            bounds=DESIGN_BOUNDS, n=1, q=1
                        ).squeeze().cpu().numpy()
                        next_x = np.append(sobol_pt, 2).reshape(1, -1)
            else:
                # Subsequent HF experiments: GP-guided with exploration (qUCB)
                try:
                    train_X, train_Y = self._to_torch()
                    model = fit_mf_gp(train_X, train_Y)
                    next_x = self._optimize_acqf_fixed_fid(
                        model, qUpperConfidenceBound,
                        fid_norm=FID_MAP[2],
                        sampler=SobolQMCNormalSampler(
                            sample_shape=torch.Size([256])
                        ),
                        beta=1.0,
                    )
                except Exception:
                    break

            Y_new = self._evaluate(next_x)
            nx = next_x.flatten()
            print(
                f"  HF #{i + 1}: titre={Y_new[0]:.4f}  |  "
                f"T={nx[0]:.2f}, pH={nx[1]:.2f}, "
                f"F1={nx[2]:.1f}, F2={nx[3]:.1f}, "
                f"F3={nx[4]:.1f}  |  cost={self.current_cost}"
            )

        # ========================================
        # Final report
        # ========================================
        self._print_results()

    def _print_results(self):
        """Print a summary of the optimisation results."""
        print(f"\n{'=' * 60}")
        print("Final Results [BOTorch]")
        print("=" * 60)
        print(f"Total experiments: {len(self.Y)}")
        print(f"Total cost: {self.current_cost} / {self.budget}")

        for fid in [0, 1, 2]:
            mask = self.X[:, -1] == fid
            if mask.any():
                n = int(mask.sum())
                best = self.Y[mask].max()
                bp = self.X[mask][np.argmax(self.Y[mask])]
                print(
                    f"\n  Fidelity {fid}: {n} experiments, "
                    f"best titre: {best:.6f}"
                )
                print(
                    f"    T={bp[0]:.2f}, pH={bp[1]:.2f}, "
                    f"F1={bp[2]:.1f}, F2={bp[3]:.1f}, F3={bp[4]:.1f}"
                )

        hf_mask = self.X[:, -1] == 2
        if hf_mask.any():
            score = self.Y[hf_mask].max()
            print(f"\n{'*' * 60}")
            print(f"  Final score (best high-fidelity titre): {score:.6f}")
            print(f"{'*' * 60}")
        else:
            print("\n  WARNING: No high-fidelity experiments were run!")


# ======================== Entry Point ========================

if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path so virtual_lab can be found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hackathon_dir = os.path.dirname(script_dir)
    sys.path.insert(0, hackathon_dir)

    from C_Bioprocess_Utils.virtual_lab import conduct_experiment

    bo = BO(
        obj_func=conduct_experiment,    # Positive titre values (maximisation)
        budget=15000,
        n_initial=7,
        phase1_cost=2100,               # Phase 1 low-fidelity budget cap
        n_hf_reserve=1,                 # Reserve 2 high-fidelity experiments for Phase 3
    )
    bo.run_optimization()

    print(f"\nbo.X shape: {bo.X.shape}")
    print(f"bo.Y shape: {bo.Y.shape}")
    print(f"Total cost: {bo.current_cost}")
