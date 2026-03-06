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
    list(range(0, 701, 100)),
    list(range(0, 7001, 1000))
]

NUM_CONDITIONALS = 64
train_indices = list(range(NUM_CONDITIONALS))  
BATCH_SIZE = 32
NUM_SAMPLES = 512
VAL_IDX = 0
val_indices = [VAL_IDX]  
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "posterior_sampling_results"
os.makedirs(SAVE_DIR, exist_ok=True)

all_results = []

@torch.no_grad()
def run_all_experiments():
    global all_results

    dset_train, dset_val, _, _ = get_seismic_dataset()

    condition_sets = []

    for idx in train_indices:
        y_cond_single = dset_train.tensors[0][idx, 1, ...].unsqueeze(0)
        condition_sets.append(("train", idx, y_cond_single))

    for idx in val_indices:
        y_cond_single = dset_val.tensors[0][idx, 1, ...].unsqueeze(0)
        condition_sets.append(("val", idx, y_cond_single))

    for cond_name, idx, y_cond_single in condition_sets:
        print(f"\n=== Running conditional: {cond_name} sample {idx} ===")
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

                for _ in tqdm(range(num_batches), desc=f"{cond_name} idx {idx} ckpt {ckpt_epoch}"):
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

                results["epoch"].append(ckpt_epoch)
                results["mean"].append(mean_image.numpy())
                results["std"].append(std_image.numpy())

                # Save separately for each conditioning index
                save_path = os.path.join(
                    SAVE_DIR,
                    f"{cond_name}_idx{idx}_exp{i}_ckpt{ckpt_epoch}.npz"
                )
                np.savez_compressed(save_path,
                                    samples=all_samples.numpy(),
                                    mean=mean_image.numpy(),
                                    std=std_image.numpy())
                print(f"Saved {save_path}")

            all_results.append((cond_name, idx, results))

def load_existing_results():
    all_results = []
    labels = ["train", "val"]

    for cond_name in labels:
        if cond_name == "train":
            idx_list = range(64) 
        else:
            idx_list = [0]  

        for idx in idx_list:
            for i in range(len(EXPERIMENTS)):
                results = {"epoch": [], "mean": [], "std": []}

                for ckpt_epoch in CHECKPOINTS[i]:
                    path = os.path.join(SAVE_DIR,
                                        f"{cond_name}_idx{idx}_exp{i}_ckpt{ckpt_epoch}.npz")
                    if not os.path.exists(path):
                        continue
                    data = np.load(path)
                    results["epoch"].append(ckpt_epoch)
                    results["mean"].append(data["mean"])
                    results["std"].append(data["std"])

                all_results.append((cond_name, idx, results))

    return all_results

# def plot_frobenius_norms_iterations():
#     plt.figure(figsize=(12,5))
#     labels = ["full dataset", "10% subset"]

#     # Set dataset sizes and batch size
#     dataset_sizes = [1000, 100]  # full dataset, 10%
#     batch_size = BATCH_SIZE

#     # ---- Frobenius norm of mean ----
#     plt.subplot(1, 2, 1)
#     for i, (cond_name, results) in enumerate(all_results):
#         if cond_name != "train":
#             continue
#         num_batches_per_epoch = int(np.ceil(dataset_sizes[i] / batch_size))
#         iterations = np.array(results["epoch"]) * num_batches_per_epoch
#         mean_vals = np.array([np.linalg.norm(m, 'fro') for m in results["mean"]])
#         plt.plot(iterations, mean_vals, label=labels[i])
#     plt.xlabel("Total iterations")
#     plt.ylabel("Frobenius norm of mean")
#     plt.title("Posterior mean Frobenius norm")
#     plt.grid(True)
#     plt.legend()

#     # ---- Frobenius norm of std ----
#     plt.subplot(1, 2, 2)
#     for i, (cond_name, results) in enumerate(all_results):
#         if cond_name != "train":
#             continue
#         num_batches_per_epoch = int(np.ceil(dataset_sizes[i] / batch_size))
#         iterations = np.array(results["epoch"]) * num_batches_per_epoch
#         std_vals = np.array([np.linalg.norm(s, 'fro') for s in results["std"]])
#         plt.plot(iterations, std_vals, label=labels[i])
#     plt.xlabel("Total iterations")
#     plt.ylabel("Frobenius norm of std")
#     plt.title("Posterior std Frobenius norm")
#     plt.grid(True)
#     plt.yscale("log")
#     plt.legend()

#     plt.tight_layout()
#     save_path = os.path.join(SAVE_DIR, "frobenius_norms_total_iterations.png")
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"Saved plot to {save_path}")
 

def plot_frobenius_norms_iterations():
    plt.figure(figsize=(12,5))
    labels = ["full dataset", "10% subset"]

    num_samples_list = [4752, 475]  # full dataset, 10% subset
    batch_size = 128

    # ---- Frobenius norm of mean ----
    plt.subplot(1, 2, 1)

    # group by experiment type (0=full,1=subset)
    for exp_idx in range(len(EXPERIMENTS)):
        mean_vals_list = []
        std_vals_list = []

        # loop over all conditionals for this experiment
        for cond_name, idx, results in all_results:
            # only consider training conditionals for this experiment
            if cond_name != "train":
                continue

            # only pick results for this experiment index
            # assuming ordering: exp0, exp1,... for each idx
            # adjust if needed
            exp_order = idx + exp_idx * 64
            res = all_results[exp_order][2]  # get results dict
            steps_per_epoch = int(np.ceil(num_samples_list[exp_idx] / batch_size))
            iterations = np.array(res["epoch"]) * steps_per_epoch
            mean_vals = np.array([np.linalg.norm(m, 'fro') for m in res["mean"]])
            std_vals = np.array([np.linalg.norm(s, 'fro') for s in res["std"]])

            mean_vals_list.append(mean_vals)
            std_vals_list.append(std_vals)

        # average across all conditionals
        mean_vals_avg = np.mean(mean_vals_list, axis=0)
        std_vals_avg = np.mean(std_vals_list, axis=0)

        label = f"{labels[exp_idx]}"
        plt.plot(iterations, mean_vals_avg, label=label)
        plt.fill_between(iterations,
                        mean_vals_avg - std_vals_avg,
                        mean_vals_avg + std_vals_avg,
                        alpha=0.3)  

    plt.xlabel("Total iterations")
    plt.ylabel("Frobenius norm of mean")
    plt.title("Frobenius norm of posterior mean")
    plt.grid(True)
    plt.legend()

    # ---- Frobenius norm of std ----
    plt.subplot(1, 2, 2)

    for exp_idx in range(len(EXPERIMENTS)):
        std_vals_list = []

        for cond_name, idx, results in all_results:
            if cond_name != "train":
                continue

            exp_order = idx + exp_idx * 64
            res = all_results[exp_order][2]
            steps_per_epoch = int(np.ceil(num_samples_list[exp_idx] / batch_size))
            iterations = np.array(res["epoch"]) * steps_per_epoch
            std_vals = np.array([np.linalg.norm(s, 'fro') for s in res["std"]])
            std_vals_list.append(std_vals)

        std_vals_avg = np.mean(std_vals_list, axis=0)
        label = f"{labels[exp_idx]}"
        plt.plot(iterations, std_vals_avg, label=label)
        plt.fill_between(iterations,
                        std_vals_avg - std_vals_avg,
                        std_vals_avg + std_vals_avg,
                        alpha=0.3)
    plt.xlabel("Total iterations")
    plt.ylabel("Frobenius norm of std")
    plt.title("Frobenius norm of posterior std")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "frobenius_norms_train_conditional_iterations.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}") 

def plot_statistics_evolution():
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
    plt.xlabel("Normalized epochs")
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
    plt.xlabel("Normalized epochs")
    plt.ylabel("Posterior std")
    plt.title("Posterior std evolution ")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

 
    plt.subplot(1, 3, 3)
    for i, (cond_name, results) in enumerate(all_results):
        if cond_name != "val":
            continue
        epochs = np.array(results["epoch"])
        energy_vals = np.array([e.mean() for e in results["energy"]])
        epochs_norm = epochs / epochs[-1]

        cutoff = int(0.1 * len(epochs_norm))
        epochs_norm_cut = epochs_norm[cutoff:]
        energy_vals_cut = energy_vals[cutoff:]

        label = f"{labels[i % 2]} ({cond_name})"
        plt.plot(epochs_norm_cut, energy_vals_cut, label=label)
    plt.xlabel("Normalized epoch ")
    plt.ylabel("Energy")
    plt.title("Posterior energy evolution ")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "posterior_statistics_evolution_val_conditional.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(SAVE_DIR) and len(os.listdir(SAVE_DIR)) > 0:
        print("Loading existing sampling results...")
        all_results = load_existing_results()
    else:
        run_all_experiments()

    # plot_statistics_evolution()
    plot_frobenius_norms_iterations()