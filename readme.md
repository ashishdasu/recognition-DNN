# Recognition using Deep Networks

**Author:** Ashish Dasu

## Links

- **Live digit recognition demo:** https://drive.google.com/file/d/1ZG766VyhJh46lZKns5g2FsrzX5kfNx_j/view?usp=share_link

- **Greek letter additional examples:** https://drive.google.com/file/d/1lHB5EN0bwnRl7Tm4SNbaN5l9Mrj1877j/view?usp=sharing

  - Contains both the provided course examples (alpha, beta, gamma) and my drawn examples (all 6 classes). The provided images were preprocessed with PIL MinFilter(5) to thicken their strokes for better domain matching with user-drawn test images.


## How to Run

```bash
python train_network.py          # Train CNN on MNIST, saves mnist_model.pth
python evaluate_network.py       # Test set eval + handwritten digit classification
python examine_network.py        # Filter visualization and application
python greek_letters.py          # Transfer learning on Greek letters (6 classes)
python transformer_network.py    # Vision transformer on MNIST
python experiment.py             # Fashion MNIST architecture experiment (50 configs)
python gabor_experiment.py       # Gabor filter bank comparison
python live_digit_recognition.py # Webcam digit recognition (press Q to quit)
python greek_tuner.py            # Greek letter hyperparameter tuning (10 configs)
python confusion_tsne.py        # Confusion matrix and t-SNE visualization
python augmentation_experiment.py # Data augmentation comparison
```

## Dependencies

- Done on M1 MacBook Pro Tahoe 26.4
- Python 3.13
- PyTorch, torchvision
- OpenCV (cv2)
- matplotlib, numpy, PIL
- scikit-learn (for confusion matrix and t-SNE)
