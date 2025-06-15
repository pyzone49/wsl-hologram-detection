import os
import numpy as np
import matplotlib.pyplot as plt

def parse_loss_file(filepath):
    steps = []
    losses = []
    with open(filepath, "r") as f:
        epoch = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                #index of epoch
                loss = float(parts[1])
                steps.append(epoch)
                losses.append(loss)
                epoch += 1
    return np.array(steps), np.array(losses)

def plot_val_and_train_loss(run_dir):
    val_loss_path = os.path.join(run_dir, "metrics", "val_loss")
    train_loss_path = os.path.join(run_dir, "metrics", "train_loss_epoch")

    if not (os.path.exists(val_loss_path) and os.path.exists(train_loss_path)):
        print("val_loss or train_loss_epoch file not found.")
        return

    val_steps, val_losses = parse_loss_file(val_loss_path)
    train_steps, train_losses = parse_loss_file(train_loss_path)

    # Align on common steps (epochs)
    common_steps = sorted(set(val_steps) & set(train_steps))
    val_dict = dict(zip(val_steps, val_losses))
    train_dict = dict(zip(train_steps, train_losses))
    val_aligned = [val_dict[s] for s in common_steps]
    train_aligned = [train_dict[s] for s in common_steps]

    plt.figure(figsize=(10, 6))
    plt.plot(common_steps, train_aligned, label="Train Loss (epoch)")
    plt.plot(common_steps, val_aligned, label="Val Loss")
    plt.xlabel("Epoch/Step")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set this to your run directory (not the parent of all runs)
    run_dir = "/home/diva/Documents/other/pouliquen.24.icdar/mlruns/670970938816456836/f38a7cdec67045038d338b6935cf2dcf"
    plot_val_and_train_loss(run_dir)