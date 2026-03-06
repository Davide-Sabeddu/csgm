# plot_seismic_val_loss.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from csgm.utils import checkpointsdir, plotsdir

# =========================
# Experiments and labels
# =========================
EXPERIMENTS = {
    "full_dataset": {
        "path": "/home/da389032/csgm/data/checkpoints/seismic_imaging_long_dataset-seismic_batchsize-128_max_epochs-1000_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4",
        "label": "Full dataset",
        "color": "#1f77b4",
        "save_freq": 50  # checkpoint frequency (in epochs)
    },
    "subset_10": {
        "path": "/home/da389032/csgm/data/checkpoints/seismic_imaging_long_dataset-seismic_batchsize-128_max_epochs-10000_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4",
        "label": "10% subset",
        "color": "#d62728",
        "save_freq": 500
    }
}

SAVE_DIR = "posterior_sampling_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Plot validation loss
# =========================
def plot_val_loss():
    plt.figure(figsize=(8,5))

    for exp_name, cfg in EXPERIMENTS.items():
        ckpt_files = sorted([
            f for f in os.listdir(cfg["path"]) if f.startswith("checkpoint_") and f.endswith(".pth")
        ], key=lambda x: int(x.split("_")[-1].split(".")[0]))

        val_losses = []
        epochs = []

        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(cfg["path"], ckpt_file)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if "val_obj" in ckpt and len(ckpt["val_obj"]) > 0:
                val_losses.append(ckpt["val_obj"][-1])  # last val_obj in checkpoint
                epochs.append(int(ckpt_file.split("_")[-1].split(".")[0]))

        if not val_losses:
            print(f"No validation data found for {exp_name}")
            continue

        # Normalize x-axis to fraction of total epochs
        epochs = np.array(epochs)
        val_losses = np.array(val_losses)
        epochs_frac = epochs / epochs[-1]

        plt.plot(epochs_frac, val_losses, label=cfg["label"], color=cfg["color"], linewidth=1.5)

    plt.xlabel("Fraction of total epochs")
    plt.ylabel("Validation loss")
    plt.title("Seismic experiment validation loss")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "seismic_val_loss.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved validation loss plot to {save_path}")
    plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    plot_val_loss()