import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from load_dataset import SatelliteDataset
from nets import (
    VisionTransformer, SwinTransformer, ConvNeXt, ResNet50, DenseNet121, DenseNet169,
    DenseNet201, EfficientNetB0, EfficientNetB7
)
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    accuracy = correct.double() / total
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return running_loss / len(dataloader.dataset), accuracy, f1, precision, recall

def plot_metrics(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'b', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1_scores, 'b', label='Validation F1 Score')
    plt.plot(epochs, val_precisions, 'g', label='Validation Precision')
    plt.plot(epochs, val_recalls, 'r', label='Validation Recall')
    plt.title('Validation F1 Score, Precision, and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset = SatelliteDataset(args.data_dir, "train", args.bands)
    val_dataset = SatelliteDataset(args.data_dir, "val", args.bands)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create the model
    if args.model == "vit":
        model = VisionTransformer(num_classes=args.num_classes, num_channels=len(args.bands), image_size=args.image_size, patch_size=args.patch_size)
    elif args.model == "swin":
        model = SwinTransformer(num_classes=args.num_classes, num_channels=len(args.bands), image_size=args.image_size, patch_size=args.patch_size)
    elif args.model == "convnext":
        model = ConvNeXt(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "resnet50":
        model = ResNet50(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "densenet121":
        model = DenseNet121(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "densenet169":
        model = DenseNet169(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "densenet201":
        model = DenseNet201(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "efficientnet_b0":
        model = EfficientNetB0(num_classes=args.num_classes, num_channels=len(args.bands))
    elif args.model == "efficientnet_b7":
        model = EfficientNetB7(num_classes=args.num_classes, num_channels=len(args.bands))
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    # Create output directory for saving models and plots
    os.makedirs(args.output_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []

    # Train the model
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

        # Record metrics in TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("F1 Score/Validation", val_f1, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_name = f"{args.model}_epoch{epoch+1}_bands{'_'.join(map(str, args.bands))}_lr{args.learning_rate}_batchsize{args.batch_size}.pth"
            torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

            # Plot metrics every 10 epochs
            plot_metrics(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, args.output_dir)

    # Save the final model
    model_name = f"{args.model}_final_bands{'_'.join(map(str, args.bands))}_lr{args.learning_rate}_batchsize{args.batch_size}.pth"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

    # Plot the final metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, args.output_dir)

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Image Classification")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--bands", type=str, default="0,1,2", help="Comma-separated list of bands to use for training")
    parser.add_argument("--model", type=str, default="vit", help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for Vision Transformer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the trained models and plots")
    args = parser.parse_args()
    main(args)