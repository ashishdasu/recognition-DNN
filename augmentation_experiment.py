# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Extension: data augmentation experiment.
# Trains identical CNN architectures with and without augmentation (random
# rotation and affine transforms) to measure how augmentation affects
# generalization on MNIST.

# import statements
import sys
import os
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from train_network import MyNetwork, train_network, test_network


# main function — trains baseline vs augmented models and compares
def main(argv):
    n_epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5

    os.makedirs('results', exist_ok=True)

    # baseline transform (same as train_network.py)
    baseline_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # augmented transform — random rotation and slight affine shift
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # test set uses baseline transform (no augmentation at test time)
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                               transform=baseline_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                               shuffle=False)

    # select device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    results = {}

    for name, transform in [('Baseline', baseline_transform), ('Augmented', augmented_transform)]:
        print(f'\n{"="*50}')
        print(f'Training {name} model')
        print(f'{"="*50}')

        train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True)

        model = MyNetwork().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        for epoch in range(1, n_epochs + 1):
            print(f'\nEpoch {epoch}/{n_epochs}')
            train_loss, train_acc, _ = train_network(model, train_loader, optimizer, epoch)
            test_loss, test_acc = test_network(model, test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f'  Train — Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'  Test  — Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

        results[name] = {
            'train_losses': train_losses, 'test_losses': test_losses,
            'train_accs': train_accs, 'test_accs': test_accs
        }

    # plot comparison
    epochs = range(1, n_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # accuracy comparison
    ax1.plot(epochs, results['Baseline']['test_accs'], 'b-o', label='Baseline')
    ax1.plot(epochs, results['Augmented']['test_accs'], 'r-o', label='Augmented')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy: Baseline vs Augmented')
    ax1.legend()

    # loss comparison
    ax2.plot(epochs, results['Baseline']['test_losses'], 'b-o', label='Baseline')
    ax2.plot(epochs, results['Augmented']['test_losses'], 'r-o', label='Augmented')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss: Baseline vs Augmented')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/augmentation_comparison.png', dpi=150)
    plt.savefig('report_images/augmentation_comparison.png', dpi=150)
    print('\nPlot saved')

    # save metrics
    with open('results/augmentation_metrics.txt', 'w') as f:
        f.write('Data Augmentation Experiment\n')
        f.write(f'Augmentation: RandomRotation(15) + RandomAffine(translate=0.1)\n\n')
        for name in ['Baseline', 'Augmented']:
            r = results[name]
            f.write(f'{name}:\n')
            f.write(f'  Final test accuracy: {r["test_accs"][-1]:.2f}%\n')
            f.write(f'  Best test accuracy:  {max(r["test_accs"]):.2f}% (epoch {r["test_accs"].index(max(r["test_accs"])) + 1})\n')
            f.write(f'  Final train accuracy: {r["train_accs"][-1]:.2f}%\n')
            f.write(f'  Train-test gap: {r["train_accs"][-1] - r["test_accs"][-1]:.2f}%\n\n')
        f.write('Epoch-by-epoch test accuracy:\n')
        f.write('Epoch  Baseline  Augmented\n')
        for i in range(n_epochs):
            f.write(f'{i+1:5d}  {results["Baseline"]["test_accs"][i]:.2f}%   {results["Augmented"]["test_accs"][i]:.2f}%\n')

    print('Metrics saved to results/augmentation_metrics.txt')
    print('Done.')


if __name__ == "__main__":
    main(sys.argv)
