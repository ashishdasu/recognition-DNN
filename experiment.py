# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Systematic experiment exploring how three architectural dimensions affect CNN performance
# on the Fashion MNIST dataset. Evaluates ~75 configurations using a linear search strategy:
# optimize one dimension at a time while holding the others at their baseline values.

# import statements
import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# ============================================================================
# EXPERIMENT PLAN (D19)
#
# Dataset: Fashion MNIST (10 clothing categories, 60k train / 10k test, 28x28 grayscale)
# Chosen over MNIST digits because it is more challenging and provides more room
# to observe the effect of architectural changes.
#
# Three dimensions explored:
#   1. Number of convolution filters (conv1 / conv2 channel counts)
#   2. Dropout rate
#   3. Number of hidden nodes in the fully connected layer
#
# Strategy: linear search — hold two dimensions at baseline, sweep the third.
# Two rounds of linear search are used (23 configs per round, ~46 total).
# After sweeping all three dimensions, the best value from each becomes the
# new baseline for round two. This captures first-order interactions without
# the full combinatorial grid (which would be 9×8×6 = 432 configurations).
#
# Baseline: conv_filters=20, dropout=0.25, hidden_nodes=100
# (deliberately offset from the MNIST spec to give room in both directions)
#
# Metrics: test accuracy (%), training time (seconds)
# Training: 5 epochs per configuration (enough to see trends, fast enough for ~75 runs)
# ============================================================================

# ============================================================================
# HYPOTHESES (D20) — written before running experiments
#
# Dimension 1: Number of convolution filters
#   Values: [4, 8, 16, 32, 48, 64]
#   Hypothesis: Accuracy will increase with more filters up to a point (~32-48),
#   then plateau or slightly decrease due to overfitting on the relatively small
#   28x28 images. More filters means more learnable feature detectors, but
#   Fashion MNIST categories (t-shirt, trouser, etc.) have limited visual
#   complexity at this resolution. Training time should increase roughly linearly
#   with filter count.
#
# Dimension 2: Dropout rate
#   Values: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
#   Hypothesis: A moderate dropout (0.2-0.3) will perform best. Zero dropout will
#   overfit — training accuracy high but test accuracy lower. Very high dropout
#   (0.5-0.7) will underfit because too much information is discarded during
#   training, preventing the network from learning strong features. The optimal
#   rate balances regularization against information loss.
#
# Dimension 3: Number of hidden nodes in the FC layer
#   Values: [16, 32, 64, 128, 256, 512]
#   Hypothesis: Accuracy will improve as hidden nodes increase from 16 to ~128-256,
#   then plateau. Too few nodes bottleneck the classifier — the convolutional
#   features cannot be combined expressively enough. Too many nodes add parameters
#   without benefit since the upstream conv layers constrain the feature richness.
#   Training time will increase modestly since the FC layer is a small fraction
#   of total computation.
# ============================================================================


# Configurable CNN for Fashion MNIST. Architecture mirrors the MNIST network
# but with adjustable filter counts, dropout rate, and hidden layer size.
class ExperimentNetwork(nn.Module):

    # creates the network with parameterized architecture
    def __init__(self, conv1_filters=20, conv2_filters=40, dropout_rate=0.25, hidden_nodes=100):
        super(ExperimentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        # after two 5x5 convs and two 2x2 pools: 28 -> 24 -> 12 -> 8 -> 4
        fc_input = conv2_filters * 4 * 4
        self.fc1 = nn.Linear(fc_input, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 10)

        self.fc_input = fc_input

    # forward pass — same structure as the MNIST CNN
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.fc_input)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Trains and evaluates a single configuration. Returns test accuracy and training time.
def run_single_experiment(config, train_loader, test_loader, device, n_epochs=5):
    model = ExperimentNetwork(
        conv1_filters=config['conv1_filters'],
        conv2_filters=config['conv2_filters'],
        dropout_rate=config['dropout_rate'],
        hidden_nodes=config['hidden_nodes']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    # train
    for epoch in range(n_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    # evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)

    test_acc = 100.0 * correct / total
    return test_acc, train_time


# Runs a sweep over one dimension while holding others at baseline values.
# Returns a list of (param_value, accuracy, time) tuples.
def sweep_dimension(dim_name, dim_values, baseline, train_loader, test_loader, device, n_epochs=5):
    results = []
    for val in dim_values:
        config = baseline.copy()
        config[dim_name] = val

        # for filter count, scale conv2 to 2x conv1 (standard practice)
        if dim_name == 'conv1_filters':
            config['conv2_filters'] = val * 2

        acc, t = run_single_experiment(config, train_loader, test_loader, device, n_epochs)
        results.append((val, acc, t))
        print(f'  {dim_name}={val:<6}  Acc: {acc:.2f}%  Time: {t:.1f}s')

    return results


# Plots sweep results for one dimension: accuracy and training time vs parameter value.
def plot_sweep(dim_name, dim_label, results, round_num, output_dir):
    values = [r[0] for r in results]
    accs = [r[1] for r in results]
    times = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(values, accs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel(dim_label)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title(f'{dim_label} vs Accuracy (Round {round_num})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(values, times, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel(dim_label)
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title(f'{dim_label} vs Training Time (Round {round_num})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{output_dir}/experiment_{dim_name}_round{round_num}.png'
    plt.savefig(filename, dpi=150)
    print(f'  Plot saved to {filename}')


# main function — loads Fashion MNIST, runs two rounds of linear search across 3 dimensions
def main(argv):
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # select best available device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # load Fashion MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.2860,), (0.3530,))  # Fashion MNIST stats
    ])
    train_dataset = torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                                       transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                                      transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # dimension definitions
    dimensions = {
        'conv1_filters': {
            'values': [4, 8, 16, 24, 32, 48, 64, 96, 128],
            'label': 'Number of Conv Filters (conv1)',
        },
        'dropout_rate': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'label': 'Dropout Rate',
        },
        'hidden_nodes': {
            'values': [16, 32, 64, 128, 192, 256, 384, 512],
            'label': 'Hidden Nodes in FC Layer',
        },
    }

    # baseline configuration — starting point for sweeps
    baseline = {
        'conv1_filters': 20,
        'conv2_filters': 40,
        'dropout_rate': 0.25,
        'hidden_nodes': 100,
    }

    all_results = []  # collect every run for CSV export

    # two rounds of linear search — second round uses best values from first round
    for round_num in range(1, 3):
        print(f'\n{"="*60}')
        print(f'ROUND {round_num} — Baseline: {baseline}')
        print(f'{"="*60}')

        for dim_name, dim_info in dimensions.items():
            print(f'\nSweeping {dim_info["label"]}...')
            results = sweep_dimension(dim_name, dim_info['values'], baseline,
                                      train_loader, test_loader, device, n_epochs=5)

            # record results
            for val, acc, t in results:
                all_results.append({
                    'round': round_num,
                    'dimension': dim_name,
                    'value': val,
                    'accuracy': acc,
                    'time': t,
                    'baseline': str(baseline),
                })

            # plot this sweep
            plot_sweep(dim_name, dim_info['label'], results, round_num, output_dir)

            # update baseline with best value from this sweep
            best_val, best_acc, _ = max(results, key=lambda r: r[1])
            if dim_name == 'conv1_filters':
                baseline['conv1_filters'] = best_val
                baseline['conv2_filters'] = best_val * 2
            else:
                baseline[dim_name] = best_val
            print(f'  Best: {dim_name}={best_val} ({best_acc:.2f}%) — updated baseline')

    # save all results to CSV
    csv_path = f'{output_dir}/experiment_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'dimension', 'value',
                                                'accuracy', 'time', 'baseline'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\nAll results saved to {csv_path}')

    # save summary
    summary_path = f'{output_dir}/experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('EXPERIMENT SUMMARY\n')
        f.write(f'Total configurations tested: {len(all_results)}\n')
        f.write(f'Final optimized baseline: {baseline}\n\n')

        best_overall = max(all_results, key=lambda r: r['accuracy'])
        f.write(f'Best single result: {best_overall["accuracy"]:.2f}% '
                f'({best_overall["dimension"]}={best_overall["value"]}, '
                f'round {best_overall["round"]})\n\n')

        f.write('HYPOTHESES vs RESULTS\n')
        f.write('-' * 60 + '\n')
        f.write('See experiment.py header comments for hypotheses.\n')
        f.write('Compare against the plots in results/experiment_*.png\n')
    print(f'Summary saved to {summary_path}')

    print(f'\nFinal optimized configuration: {baseline}')
    print(f'Total experiments run: {len(all_results)}')


if __name__ == "__main__":
    main(sys.argv)
