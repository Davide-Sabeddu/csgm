import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from csgm import NoiseScheduler, ConditionalScoreModel2D
from csgm.utils import get_seismic_dataset, make_grid

EXPERIMENTS = [
    "/home/da389032/csgm/data/checkpoints/seismic_imaging_long_dataset-seismic_batchsize-128_max_epochs-1000_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4",
    "/home/da389032/csgm/data/checkpoints/seismic_imaging_long_dataset-seismic_batchsize-128_max_epochs-10000_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4"
]

CHECKPOINTS = [
    list(range(0, 701, 50)),
    list(range(0, 7001, 500))
]

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TRAIN_IDX = 0
VAL_IDX = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "posterior_sampling_results"
os.makedirs(SAVE_DIR, exist_ok=True)

all_results = []

@torch.no_grad()
def run_all_experiments():
    global all_results

    dset_train, dset_val, _, _ = get_seismic_dataset()

    condition_sets = {
        "train": dset_train.tensors[0][TRAIN_IDX, 1, ...].unsqueeze(0),
        "val":   dset_val.tensors[0][VAL_IDX, 1, ...].unsqueeze(0)
    }

    for cond_name, y_cond_single in condition_sets.items():
        print(f"\n=== Running conditional: {cond_name} ===")
        y_cond_single = y_cond_single.to(DEVICE)

        for i, exp_dir in enumerate(EXPERIMENTS):
            print(f"\n--- Experiment {i}: {exp_dir} ---")
            results = {"epoch": [], "mean": [], "std": [], "energy": []}

            for ckpt_epoch in CHECKPOINTS[i]:
                ckpt_path = os.path.join(exp_dir, f"checkpoint_{ckpt_epoch}.pth")
                print(f"Loading checkpoint {ckpt_epoch}...")
                checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

                args = checkpoint["args"]

                model = ConditionalScoreModel2D(
                    modes=args.modes,
                    hidden_dim=args.hidden_dim,
                    nlayers=args.nlayers,
                    nt=args.nt
                ).to(DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                noise_scheduler = NoiseScheduler(
                    nt=args.nt,
                    beta_schedule=args.beta_schedule,
                    device=DEVICE
                )

                y_cond_batch = y_cond_single.repeat(BATCH_SIZE, 1, 1, 1)
                grid = make_grid(dset_train.tensors[0].shape[2:]).to(DEVICE)
                grid = grid.repeat(BATCH_SIZE,1,1,1)

                all_samples = []
                num_batches = int(np.ceil(NUM_SAMPLES / BATCH_SIZE))

                for _ in tqdm(range(num_batches), desc=f"{cond_name} ckpt {ckpt_epoch}"):
                    sample = torch.randn((BATCH_SIZE, *args.input_size, 1), device=DEVICE)
                    timesteps = list(range(len(noise_scheduler)))[::-1]

                    for t in timesteps:
                        t_batch = torch.full((BATCH_SIZE,), t, device=DEVICE, dtype=torch.long)
                        residual = model(sample, y_cond_batch, t_batch, grid)
                        sample = noise_scheduler.step(residual, t, sample)

                    remaining = NUM_SAMPLES - len(all_samples)
                    if remaining < BATCH_SIZE:
                        sample = sample[:remaining]

                    all_samples.append(sample.clone())

                all_samples = torch.cat(all_samples, dim=0)[...,0].cpu()

                mean_image = all_samples.mean(dim=0)
                std_image  = all_samples.std(dim=0)
                energy     = torch.norm(all_samples.view(all_samples.size(0), -1), dim=1)

                results["epoch"].append(ckpt_epoch)
                results["mean"].append(mean_image.numpy())
                results["std"].append(std_image.numpy())
                results["energy"].append(energy.numpy())

                save_path = os.path.join(SAVE_DIR, f"{cond_name}_exp{i}_ckpt{ckpt_epoch}.npz")
                np.savez_compressed(save_path,
                                    samples=all_samples.numpy(),
                                    mean=mean_image.numpy(),
                                    std=std_image.numpy(),
                                    energy=energy.numpy())
                print(f"Saved {save_path}")

            all_results.append((cond_name, results))


def load_existing_results():
    all_results = []
    labels = ["train", "val"]

    for cond_name in labels:
        for i in range(len(EXPERIMENTS)):
            results = {"epoch": [], "mean": [], "std": [], "energy": []}

            for ckpt_epoch in CHECKPOINTS[i]:
                path = os.path.join(SAVE_DIR, f"{cond_name}_exp{i}_ckpt{ckpt_epoch}.npz")
                if not os.path.exists(path):
                    continue
                data = np.load(path)
                results["epoch"].append(ckpt_epoch)
                results["mean"].append(data["mean"])
                results["std"].append(data["std"])
                results["energy"].append(data["energy"])

            all_results.append((cond_name, results))

    return all_results


def plot_statistics_evolution_train_mixed():
    plt.figure(figsize=(18,5))
    labels = ["full dataset", "10% subset"]

    
    plt.subplot(1, 3, 1)
    for i, (cond_name, results) in enumerate(all_results):
        if cond_name != "train":
            continue
        epochs = np.array(results["epoch"])
        mean_vals = np.array([m.mean() for m in results["mean"]])
        epochs_norm = epochs / epochs[-1]
        label = f"{labels[i % 2]} ({cond_name})"
        plt.plot(epochs_norm, mean_vals, label=label)
    plt.xlabel("Normalized epoch")
    plt.ylabel("Mean of posterior")
    plt.title("Posterior mean evolution")
    plt.grid(True)
    plt.legend()

 
    plt.subplot(1, 3, 2)
    for i, (cond_name, results) in enumerate(all_results):
        if cond_name != "train":
            continue
        epochs = np.array(results["epoch"])
        std_vals = np.array([s.mean() for s in results["std"]])
        epochs_norm = epochs / epochs[-1]

        cutoff = int(0.1 * len(epochs_norm))
        epochs_norm_cut = epochs_norm[cutoff:]
        std_vals_cut = std_vals[cutoff:]

        label = f"{labels[i % 2]} ({cond_name})"
        plt.plot(epochs_norm_cut, std_vals_cut, label=label)
    plt.xlabel("Normalized epoch (after 10%)")
    plt.ylabel("Posterior std")
    plt.title("Posterior std evolution (train)")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

 
    plt.subplot(1, 3, 3)
    for i, (cond_name, results) in enumerate(all_results):
        if cond_name != "train":
            continue
        epochs = np.array(results["epoch"])
        energy_vals = np.array([e.mean() for e in results["energy"]])
        epochs_norm = epochs / epochs[-1]

        cutoff = int(0.1 * len(epochs_norm))
        epochs_norm_cut = epochs_norm[cutoff:]
        energy_vals_cut = energy_vals[cutoff:]

        label = f"{labels[i % 2]} ({cond_name})"
        plt.plot(epochs_norm_cut, energy_vals_cut, label=label)
    plt.xlabel("Normalized epoch (after 10%)")
    plt.ylabel("Energy")
    plt.title("Posterior energy evolution (train)")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "posterior_statistics_evolution_train_mixed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(SAVE_DIR) and len(os.listdir(SAVE_DIR)) > 0:
        print("Loading existing sampling results...")
        all_results = load_existing_results()
    else:
        run_all_experiments()

    plot_statistics_evolution_train_mixed()