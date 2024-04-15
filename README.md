Here's a detailed README file for the project MineNet:

# MineNet

MineNet is a PyTorch-based deep learning project for satellite image classification. It provides an implementation of various state-of-the-art convolutional neural network (CNN) architectures and Transformers for classifying satellite images into different categories. The project supports multiple input bands and allows for easy configuration of the model, dataset, and training parameters.

## Features

- Support for various CNN architectures: ResNet50, DenseNet (121, 169, 201), EfficientNet (B0, B7)
- Support for Vision Transformers (ViT) and Swin Transformers
- Flexible configuration of input bands and image size
- Data augmentation techniques (random horizontal and vertical flips) for training
- Evaluation metrics: loss, accuracy, F1-score, precision, and recall
- TensorBoard integration for visualizing training and validation metrics
- Model checkpointing and saving of best-performing models
- Plotting of training and validation metrics

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/MineNet.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset by placing the satellite images and corresponding labels in the `data/train/train` directory. The labels should be provided in a CSV file named `answer.csv` with each line containing the image filename and label separated by a comma.

2. Run the training script with appropriate arguments:

```
python main.py --data_dir path/to/data --bands 0,1,2 --model resnet50 --num_classes 2 --epochs 100 --batch_size 32 --learning_rate 0.001 --output_dir path/to/output
```

Here's a brief description of the available arguments:

- `--data_dir`: Directory containing the dataset
- `--bands`: Comma-separated list of bands to use for training (e.g., `0,1,2`)
- `--model`: Model architecture (e.g., `resnet50`, `densenet121`, `vit`)
- `--num_classes`: Number of classes in the dataset
- `--image_size`: Input image size (default: 500)
- `--patch_size`: Patch size for Vision Transformer (default: 16)
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for the optimizer
- `--epochs`: Number of training epochs
- `--output_dir`: Directory to save the trained models and plots

3. During training, the script will print the loss, accuracy, F1-score, precision, and recall for the validation and test sets after each epoch. Additionally, TensorBoard logs will be saved in the specified `output_dir`, allowing you to visualize the training and validation metrics.

4. The trained models will be saved in the `output_dir` directory every 10 epochs, with a filename in the format `{model}_epoch{epoch}_bands{bands}_lr{learning_rate}_batchsize{batch_size}.pth`. The final model will be saved as `{model}_final_bands{bands}_lr{learning_rate}_batchsize{batch_size}.pth`.

5. Plots showing the training and validation metrics will be saved in the `output_dir` directory as `metrics_plot.png` after every 10 epochs and at the end of training.

## Project Structure

```
MineNet/
├── data/
│   └── train/
│       ├── train/
│       └── answer.csv
├── main.py
├── load_dataset.py
├── nets.py
├── requirements.txt
└── README.md
```

- `data/`: Directory for storing the dataset
  - `train/`: Directory containing the training images
  - `answer.csv`: CSV file with image filenames and labels
- `main.py`: Main script for training and evaluation
- `load_dataset.py`: Module for loading and preprocessing the dataset
- `nets.py`: Module containing the model architectures


