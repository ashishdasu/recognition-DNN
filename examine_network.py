# Ashish Dasu (adasu)
# CS5330 — Project 5: Recognition using Deep Networks
# Examines a trained MNIST CNN by visualizing its first-layer filters
# and showing their effect on an example digit.

# import statements
import os
import sys
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from train_network import MyNetwork


# Prints the full model structure and the shape/values of conv1 filter weights.
# The conv1 layer has 10 filters of shape [1, 5, 5] — one channel, 5x5 each.
def print_model_and_weights(model):
    print('--- Model Structure ---')
    print(model)

    weights = model.conv1.weight
    print(f'\nconv1 weight shape: {weights.shape}')
    print(f'  {weights.shape[0]} filters, {weights.shape[1]} input channel, '
          f'{weights.shape[2]}x{weights.shape[3]} kernel')

    for i in range(weights.shape[0]):
        print(f'\nFilter {i}:')
        print(weights[i, 0].detach().numpy())


# Plots the 10 conv1 filters in a 3x4 grid (last 2 cells empty).
# Each 5x5 filter is displayed as a small heatmap showing learned edge/texture detectors.
def plot_filters(model):
    weights = model.conv1.weight

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    with torch.no_grad():
        for i in range(10):
            row, col = divmod(i, 4)
            ax = axes[row][col]
            ax.imshow(weights[i, 0].numpy(), cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.set_xticks([])
            ax.set_yticks([])

    # hide unused subplot cells
    for i in range(10, 12):
        row, col = divmod(i, 4)
        axes[row][col].axis('off')

    plt.suptitle('Conv1 Learned Filters (5x5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/conv1_filters.png', dpi=150)
    print('Filter visualization saved to results/conv1_filters.png')


# Applies each of the 10 conv1 filters to the first training image using cv2.filter2D.
# This shows the raw convolution output before any pooling or activation,
# revealing which spatial features each filter responds to.
def plot_filter_effects(model, train_dataset):
    image, label = train_dataset[0]
    image_np = image.squeeze().numpy()

    weights = model.conv1.weight

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # first cell: the original digit
    axes[0][0].imshow(image_np, cmap='gray')
    axes[0][0].set_title(f'Original (digit {label})')
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    # remaining cells: filtered versions
    with torch.no_grad():
        for i in range(10):
            kernel = weights[i, 0].numpy()
            filtered = cv2.filter2D(image_np, -1, kernel)

            # +1 offset because cell 0 is the original image
            row, col = divmod(i + 1, 4)
            ax = axes[row][col]
            ax.imshow(filtered, cmap='gray')
            ax.set_title(f'Filter {i}')
            ax.set_xticks([])
            ax.set_yticks([])

    # hide the last unused cell
    axes[2][3].axis('off')

    plt.suptitle('Conv1 Filters Applied to First Training Example', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/conv1_filter_effects.png', dpi=150)
    print('Filter effects visualization saved to results/conv1_filter_effects.png')


# main function — loads model, prints structure, visualizes filters and their effects
def main(argv):
    os.makedirs('results', exist_ok=True)

    # load trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()
    print('Loaded model from mnist_model.pth\n')

    # print model structure and conv1 weights
    print_model_and_weights(model)

    # save model structure and weights to file for the report
    with open('results/examine_output.txt', 'w') as f:
        f.write('--- Model Structure ---\n')
        f.write(str(model) + '\n')
        weights = model.conv1.weight
        f.write(f'\nconv1 weight shape: {weights.shape}\n')
        f.write(f'  {weights.shape[0]} filters, {weights.shape[1]} input channel, '
                f'{weights.shape[2]}x{weights.shape[3]} kernel\n')
        for i in range(weights.shape[0]):
            f.write(f'\nFilter {i}:\n')
            f.write(str(weights[i, 0].detach().numpy()) + '\n')
    print('Model structure and weights saved to results/examine_output.txt')

    # visualize the 10 learned filters
    plot_filters(model)

    # load first training example (unnormalized for cleaner filter2D output)
    train_dataset_raw = torchvision.datasets.MNIST('data', train=True, download=True,
                                                    transform=torchvision.transforms.ToTensor())

    # show what each filter does to a real digit
    plot_filter_effects(model, train_dataset_raw)

    print('\nDone. Check results/ for saved plots.')


if __name__ == "__main__":
    main(sys.argv)
