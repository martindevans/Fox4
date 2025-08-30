import argparse
import torch
from pathlib import Path
import os
from types import SimpleNamespace
import json
import uuid
import subprocess
import shutil
from tqdm import tqdm
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from score import StableAltitudeScore
from model import Fox4RLNetwork
import training

Fox4DllPath = os.path.abspath("../csharp/AIPProvider/bin/Debug/net6.0/AIPProvider.dll")
MapPath = os.path.abspath("../csharp/Map")
AIPilotPath = os.path.abspath("../csharp/AIPSim/AIPilot.exe")
HeadlessClientPath = os.path.abspath("../csharp/HeadlessClient/HeadlessClient.exe")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPOParameters = training.PPOParameters()
ScoreFunction = StableAltitudeScore()

def create_model(args):
    print("🎲 Creating random Fox4 RL model...")
    
    # Initialise model and save it (pth and onnx)
    model = Fox4RLNetwork()
    model.save(args.path)
    
    print(f"✅ Random model exported to: {args.path}")

def run_sims(path, count, parallel):
    # Setup runs
    for i in tqdm(range(0, count), desc="Creating Sim Folders", leave=False):

        # Create a folder for the run (delete and recreate if it exists)
        sim_folder = os.path.abspath(os.path.join(path, f"sim_{i}"))
        if os.path.exists(sim_folder):
            shutil.rmtree(sim_folder)
        os.mkdir(sim_folder)

        # Write config file into folder
        with open(os.path.join(sim_folder, f"simConfig.json"), mode="w") as f:
            f.write(json.dumps({
                "allied": Fox4DllPath,
                "enemy": Fox4DllPath,
                "noMap": True,
                "debugEnemy": True,
                "spawnDist": 0,
                "alliedArgs": [ f"--log-tensors --output-rand-dev 1 --runid {str(uuid.uuid4())}" ],
                "enemyArgs": [ f"--log-tensors --output-rand-dev 1 --runid {str(uuid.uuid4())}" ]
            }, indent=2))
        
        # Copy model into folder
        shutil.copyfile(os.path.join(path, "model.onnx"), os.path.join(sim_folder, "model.onnx"))
        
        # Write bat into folder
        with open(os.path.join(sim_folder, f"run.bat"), mode="w") as f:
            f.writelines([
                f"cd {sim_folder}\n",
                f"{AIPilotPath} \"simConfig.json\" > sim.log\n",
                f"{HeadlessClientPath} --convert --input recording.json --output result.vtgr --map \"{MapPath}\"\n"
            ])
            pass
    
    # Run all sims
    with tqdm(total=count, desc=f"Executing Simulations", leave=False) as pbar:
        for i in range(0, count, parallel):
            procs = []
            for j in range(i, min(count, i + parallel)):
                bat_path = Path(os.path.join(path, f"sim_{j}", "run.bat")).resolve()
                procs.append(subprocess.Popen(bat_path, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
            try:
                for p in procs:
                    p.wait()
                    pbar.update(1)
            except KeyboardInterrupt:
                for p in procs:
                    try:
                        p.terminate()
                    except OSError:
                        pass
                raise KeyboardInterrupt()
    
    # Cleanup
    for dirpath, _, filenames in os.walk(path):
        if dirpath == path:
            continue
        for f in filenames:
            if f.endswith(".onnx") or f.endswith(".json"):
                os.remove(os.path.join(dirpath, f))

    # Postprocessing datasets
    for i in tqdm(range(0, count), desc="Postprocessing Dataset", leave=False):
        sim_folder = os.path.abspath(os.path.join(path, f"sim_{i}"))

        for in_file in glob.glob(os.path.join(sim_folder, "input_tensors*")):

            # Load the version of the model we just generated data with
            # Do this again for every file, in case the model is stateful (e.g. recurrent layers)
            model = Fox4RLNetwork()
            model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
            model.to(device)
            model.eval()

            df_input = pd.read_csv(in_file)

            out_file = in_file.replace("input_", "output_")
            df_output = pd.read_csv(out_file)

            extra_file = in_file.replace("input_", "extra_")
            df_extra = pd.DataFrame()

            # Clean column names, removing spaces
            df_input.columns = df_input.columns.str.strip()
            df_output.columns = df_output.columns.str.strip()

            # Add output log probs and value

            # Convert entire DataFrame to tensor
            inputs_tensor = torch.tensor(df_input.to_numpy(dtype="float32"), device=device)
            outputs_tensor = torch.tensor(df_output.to_numpy(dtype="float32"), device=device)

            # Forward pass in batch
            value, distributions = model.forward(inputs_tensor)

            # Compute log probs in batch
            log_probs = distributions.log_prob(outputs_tensor).sum(dim=1)

            # Add columns to dataframe
            df_extra["log_prob"] = log_probs.cpu().detach().numpy()
            df_extra["value"] = value.cpu().detach().numpy()

            # Add "done" col, zero for all but last row
            df_extra["done"] = 0
            df_extra.loc[df_input.index[-1], "done"] = 1

            # Add "score" col, evaluating score function for each state
            ScoreFunction.attach_score(df_input, df_output, df_extra)
            df_extra["score_cumulative"] = df_extra["score"].cumsum()

            # Save all CSVs, rounding to 4 sig fig to save some space
            df_extra.to_csv(extra_file, index=False, float_format="%.4g")
            df_input.to_csv(in_file, index=False, float_format="%.4g")
            df_output.to_csv(out_file, index=False, float_format="%.4g")


def args_run_sims(args):
    run_sims(
        args.path,
        args.count,
        args.parallel
    )

def find_max_generation():
    return (1, "path")

def train_loop(path, generations, start_gen, sim_count, sim_parallel):
    print(f"♻️  Beginning training loop")

    # Create training folder
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "generations", "0"))

        margs = SimpleNamespace()
        margs.path = os.path.join(path, "generations", "0", "model.onnx")
        create_model(margs)

    gen_idx = 0
    for gen in range(start_gen, start_gen + generations):
        gen_idx += 1
        print(f"Starting Generation: {gen_idx}/{generations}")

        # Execute simulations in sub folders
        generation_dir = os.path.join(path, "generations", str(gen))
        run_sims(generation_dir, sim_count, sim_parallel)

        # Load one big dataset from all sim folders in generation
        (df_in, df_out, df_extra) = training.load_datasets(generation_dir, sim_count)
        print(f" - Loaded {df_in.shape[0]} rows")

        # Load model from gen folder
        model = Fox4RLNetwork()
        model.load_state_dict(torch.load(os.path.join(generation_dir, "model.pth")))
        model.to(device)

        # Do training and save it to the folder under a new name
        ((model2, losses, entropies), avg_reward) = training.train(model, df_in, df_out, df_extra, PPOParameters, device)
        model2.save(os.path.join(generation_dir, "trained.onnx"))

        # Write out training stats
        with open(os.path.join(path, f"loss_log.csv"), mode="a") as f:
            for _, ((vl, pl), en) in enumerate(zip(losses, entropies)):
                f.write(f"{vl},{pl},{en}\n")
        with open(os.path.join(path, f"reward_log.csv"), mode="a") as f:
            f.write(f"{avg_reward}\n")

        # Output loss graph
        plt.clf()
        df_log = pd.read_csv(os.path.join(path, f"loss_log.csv"))
        fig, ax1 = plt.subplots()

        # Plot first line on left y-axis
        ax1.plot(np.log(df_log.iloc[:, 0].clip(lower=1e-8)), color='blue', label='Value Loss')
        ax1.set_ylabel('log(Value Loss)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create right y-axis and plot second line
        ax2 = ax1.twinx()
        ax2.plot(df_log.iloc[:, 2], color='red', label='Entropy')
        ax2.set_ylabel('Entropy', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.grid(True)
        plt.savefig(os.path.join(path, "Loss.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # Output reward graph
        df_log = pd.read_csv(os.path.join(path, f"reward_log.csv"))
        plt.plot(df_log.iloc[:, 0], color='blue', label='Average Reward')
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.savefig(os.path.join(path, "Reward.png"), dpi=300, bbox_inches='tight')
        plt.clf()

        # Make next gen folder
        next_gen_dir = os.path.join(path, "generations", str(gen + 1))
        if os.path.exists(next_gen_dir):
            shutil.rmtree(next_gen_dir)
        os.mkdir(next_gen_dir)

        # Copy in trained model from this gen to next gen
        shutil.copy(os.path.join(generation_dir, "trained.onnx"), os.path.join(next_gen_dir, "model.onnx"))
        shutil.copy(os.path.join(generation_dir, "trained.pth"), os.path.join(next_gen_dir, "model.pth"))

def train_loop_args(args):
    train_loop(args.path, args.generations, args.resume, args.sim_count, args.sim_parallel)

def main():
    parser = argparse.ArgumentParser(prog='RL Runner', description='Runs various RL related actions')
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-model
    parser_create = subparsers.add_parser("create-model", help="Create a randomly initialised model")
    parser_create.add_argument("--path", type=str, required=True)
    parser_create.set_defaults(func=create_model)

    # run-sims
    parser_sim = subparsers.add_parser("run-sims", help="Run simulations and dump results")
    parser_sim.add_argument("--path", type=str, required=True)
    parser_sim.add_argument("--count", type=int, required=False, default=1)
    parser_sim.add_argument("--parallel", type=int, required=False, default=8)
    parser_sim.set_defaults(func=args_run_sims)

    # train-loop
    parser_train = subparsers.add_parser("train-loop", help="Begin running a training loop")
    parser_train.add_argument("--path", type=str, required=True)
    parser_train.add_argument("--generations", type=int, required=False, default=4)
    parser_train.add_argument("--resume", type=int, required=False, default=0)
    parser_train.add_argument("--sim_count", type=int, required=False, default=1)
    parser_train.add_argument("--sim_parallel", type=int, required=False, default=8)
    parser_train.set_defaults(func=train_loop_args)

    # 4. Parse and dispatch
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()