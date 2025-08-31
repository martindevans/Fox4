import argparse
import os
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
import numpy as np

# Local imports
from model import Fox4RLNetwork
import training
import run as rl_run


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model_files(model_dir: Path, base_name: str = "model") -> None:
    """Ensure both model.pth and model.onnx exist in model_dir.

    If .onnx is missing but .pth exists, rebuild the ONNX by loading the pth and calling save().
    """
    onnx_path = model_dir / f"{base_name}.onnx"
    pth_path = model_dir / f"{base_name}.pth"
    if onnx_path.exists():
        return
    if pth_path.exists():
        model = Fox4RLNetwork()
        state = torch.load(str(pth_path), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        # Use the network's save() which writes both .onnx and .pth
        model.save(str(onnx_path))

def collect_replays(sim_root: Path, sim_count: int, dest: Path) -> None:
    """Collect result.vtgr files from sim_* folders into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(sim_count):
        src = sim_root / f"sim_{i}" / "result.vtgr"
        if src.exists():
            shutil.copy(src, dest / f"sim_{i}.vtgr")

def ensure_gen0(path: str) -> None:
    """Ensure the base generation (0) exists with an initial random model."""
    gen0 = Path(path) / "generations" / "0"
    gen0.mkdir(parents=True, exist_ok=True)

    # If the base model doesn't exist, create it using the helper already present
    if not (gen0 / "model.onnx").exists() or not (gen0 / "model.pth").exists():
        margs = SimpleNamespace()
        margs.path = str(gen0 / "model.onnx")
        rl_run.create_model(margs)


def run_generation_sims(gen_dir: Path, sim_count: int, sim_parallel: int) -> None:
    """Run the csharp simulator to produce datasets for this generation."""
    rl_run.run_sims(str(gen_dir), sim_count, sim_parallel)

def evaluate_params(
    gen_dir: Path,
    df_in,
    df_out,
    df_extra,
    params: training.PPOParameters,
    candidate_dir: Path,
    val_sims: int = 0,
    val_parallel: int = 1,
) -> Tuple[float, Fox4RLNetwork]:
    """Train a model using provided dataset and hyperparameters, return avg reward.

    - Loads the base model from gen_dir/model.pth to align with
      the recorded old_log_probs in df_extra.
    - Trains with PPO and saves the trained model into candidate_dir.
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Load base model for this generation
    model = Fox4RLNetwork()
    model.load_state_dict(torch.load(str(gen_dir / "model.pth"), map_location=device))
    model.to(device)

    # Train with selected hyperparameters
    (trained_tuple, avg_reward) = training.train(
        model,
        df_in,
        df_out,
        df_extra,
        params,
        device,
    )
    trained_model, losses, entropies = trained_tuple

    # Save artifacts
    (candidate_dir / "artifacts").mkdir(exist_ok=True)
    trained_model.save(str((candidate_dir / "artifacts" / "trained.onnx")))
    with open(candidate_dir / "params.json", "w") as f:
        f.write(params.to_json())
    with open(candidate_dir / "metrics.json", "w") as f:
        json.dump({
            "avg_reward": float(avg_reward),
            "final_policy_loss": float(losses[-1][1]) if losses else None,
            "final_value_loss": float(losses[-1][0]) if losses else None,
            "final_entropy": float(entropies[-1]) if entropies else None,
        }, f, indent=2)

    #  Validate by running sims with the trained model to get a real reward
    if val_sims > 0:
        # Place trained model where run_sims expects it
        src_pth = candidate_dir / "artifacts" / "trained.pth"
        src_onnx = candidate_dir / "artifacts" / "trained.onnx"
        if src_pth.exists():
            shutil.copy(src_pth, candidate_dir / "model.pth")
        if src_onnx.exists():
            shutil.copy(src_onnx, candidate_dir / "model.onnx")
        # If only pth exists, rebuild onnx
        ensure_model_files(candidate_dir, base_name="model")

        # Generate new trajectories using this trained model
        rl_run.run_sims(str(candidate_dir), val_sims, val_parallel)

        # Load validation datasets and compute average reward
        v_in, v_out, v_extra = training.load_datasets(str(candidate_dir), val_sims)
        total_episodes = v_extra["done"].sum()
        if total_episodes == 0:
            eval_reward = float("-inf")
        else:
            eval_reward = float(v_extra["score"].sum() / total_episodes)

        # Collect replays produced by validation runs
        collect_replays(candidate_dir, val_sims, candidate_dir / "replays")
        return eval_reward, trained_model

    return float(avg_reward), trained_model


def select_top_k(results: List[Tuple[int, float, training.PPOParameters, Path]], k: int):
    """Select top-k candidates by reward.

    results is a list of tuples: (idx, avg_reward, params, candidate_dir)
    """
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted[:k]


def search(
    path: str,
    generations: int,
    population: int,
    top_k: int,
    mutation_std: float,
    sim_count: int,
    sim_parallel: int,
    seed: int | None,
    val_sims: int = 0,
    val_parallel: int = 1,
):
    """Run a population-based RL hyperparameter search.

    Flow per generation:
    1) Ensure generation folder and base model
    2) Run simulator to build dataset
    3) Evaluate a population of PPOParameters on the same dataset
    4) Pick top-k; copy the single best trained model forward as next gen base
    """

    rng = np.random.RandomState(seed)

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    leaderboard_path = root / "leaderboard.csv"
    if not leaderboard_path.exists():
        with open(leaderboard_path, "w") as f:
            f.write("generation,candidate,avg_reward,params_json\n")

    # Gen0 setup
    ensure_gen0(path)

    # Start from a baseline; updated each generation to the best found
    base_params = training.PPOParameters()

    # Track the global best across all generations
    global_best_reward = float("-inf")
    global_best_dir = root / "best_global"
    global_best_dir.mkdir(exist_ok=True)

    for gen in range(generations):
        gen_dir = root / "generations" / str(gen)
        next_gen_dir = root / "generations" / str(gen + 1)
        next_gen_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generation {gen} ===")
        print("Running simulations to collect dataset…")
        run_generation_sims(gen_dir, sim_count, sim_parallel)

        # Load combined dataset from sim folders
        print("Loading dataset…")
        df_in, df_out, df_extra = training.load_datasets(str(gen_dir), sim_count)
        total_rows = df_in.shape[0]
        print(f" - Dataset rows: {total_rows}")

        # Build population of hyperparameters
        params_list: List[training.PPOParameters] = []
        for i in range(population):
            if i == 0 and gen == 0:
                # Keep one unmutated baseline for reference in the first gen
                params_list.append(base_params)
            else:
                # Mutate around the baseline or best of previous gen
                params_list.append(base_params.mutate(stddev=mutation_std, seed=int(rng.randint(0, 1_000_000))))

        # Evaluate all candidates
        print("Evaluating population…")
        results = []  # (idx, reward, params, cand_dir)
        for i, params in enumerate(params_list):
            cand_dir = gen_dir / "candidates" / f"cand_{i}"
            avg_reward, _ = evaluate_params(
                gen_dir,
                df_in,
                df_out,
                df_extra,
                params,
                cand_dir,
                val_sims=val_sims,
                val_parallel=val_parallel if val_parallel else sim_parallel,
            )
            results.append((i, avg_reward, params, cand_dir))

            with open(leaderboard_path, "a") as f:
                f.write(f"{gen},{i},{avg_reward},{params.to_json()}\n")

        # Pick top-k
        top = select_top_k(results, min(top_k, len(results)))
        best_idx, best_reward, best_params, best_dir = top[0]
        print(f"Best candidate: cand_{best_idx} with avg_reward={best_reward:.3f}")

        # Persist best-of-gen artifacts
        (gen_dir / "best").mkdir(exist_ok=True)
        # Copy trained model from best candidate into gen_dir/best and next generation model
        src_trained_pth = best_dir / "artifacts" / "trained.pth"
        src_trained_onnx = best_dir / "artifacts" / "trained.onnx"
        if src_trained_pth.exists():
            shutil.copy(src_trained_pth, gen_dir / "best" / "trained.pth")
            shutil.copy(src_trained_pth, next_gen_dir / "model.pth")
        if src_trained_onnx.exists():
            shutil.copy(src_trained_onnx, gen_dir / "best" / "trained.onnx")
            shutil.copy(src_trained_onnx, next_gen_dir / "model.onnx")

        # Ensure next generation has a valid ONNX even if only .pth was produced
        ensure_model_files(next_gen_dir, base_name="model")

        with open(gen_dir / "best" / "params.json", "w") as f:
            f.write(best_params.to_json())

        # Copy best candidate replays into best-of-gen
        cand_replays = best_dir / "replays"
        if cand_replays.exists():
            dest_replays = gen_dir / "best" / "replays"
            dest_replays.mkdir(parents=True, exist_ok=True)
            for f in cand_replays.glob("*.vtgr"):
                shutil.copy(f, dest_replays / f.name)

        # Update global best if improved
        if best_reward > global_best_reward:
            global_best_reward = best_reward
            # Copy best artifacts to a stable location
            for fname in ("trained.pth", "trained.onnx"):
                src = gen_dir / "best" / fname
                if src.exists():
                    shutil.copy(src, global_best_dir / fname)

            with open(global_best_dir / "params.json", "w") as f:
                f.write(best_params.to_json())

            with open(global_best_dir / "summary.json", "w") as f:
                json.dump({
                    "generation": gen,
                    "candidate": int(best_idx),
                    "avg_reward": float(best_reward),
                }, f, indent=2)

            # Propagate replays to global best
            gen_best_replays = gen_dir / "best" / "replays"
            if gen_best_replays.exists():
                gb_replays = global_best_dir / "replays"
                gb_replays.mkdir(parents=True, exist_ok=True)
                # Clear old replays to keep current best-only
                for old in gb_replays.glob("*.vtgr"):
                    try:
                        old.unlink()
                    except OSError:
                        pass
                for f in gen_best_replays.glob("*.vtgr"):
                    shutil.copy(f, gb_replays / f.name)

        # For the next generation, center mutations around the best of this gen
        base_params = best_params


def main():
    parser = argparse.ArgumentParser(prog="RL Hyperparameter Search", description="Population-based RL search for PPO hyperparameters")
    parser.add_argument("--path", type=str, required=True, help="Output root path for the search run")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--mutation_std", type=float, default=0.2)
    parser.add_argument("--sim_count", type=int, default=1)
    parser.add_argument("--sim_parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_sims", type=int, default=0, help="Number of validation sims per candidate (0 = off)")
    parser.add_argument("--val_parallel", type=int, default=None, help="Parallelism for validation sims; defaults to sim_parallel")

    args = parser.parse_args()

    search(
        path=args.path,
        generations=args.generations,
        population=args.population,
        top_k=args.top_k,
        mutation_std=args.mutation_std,
        sim_count=args.sim_count,
        sim_parallel=args.sim_parallel,
        seed=args.seed,
        val_sims=args.val_sims,
        val_parallel=(args.val_parallel if args.val_parallel is not None else args.sim_parallel),
    )


if __name__ == "__main__":
    main()
