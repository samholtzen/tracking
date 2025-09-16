import os
import zarr
import torch
import torch.optim as optim
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
import numpy as np

torch.cuda.set_device(5)

import sys
sys.path.append('utils/')

from training_utils import *
from training_utils import ResNetWithContrastive, ContrastiveLoss

def train_model(model, train_loader, val_loader, num_classes, compartments, num_epochs=10, learning_rate=0.001, contrastive_weight=0.5, checkpoint_dir='checkpoints'):
    # Initialize wandb
    wandb.init(project="barcodes", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "contrastive_weight": contrastive_weight,
        "model": model.model_name,
        "num_classes": num_classes
    })

    # Create a subdirectory for this run using the wandb run name
    run_checkpoint_dir = os.path.join(checkpoint_dir, wandb.run.name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device

    # Set a seed for reproducibility
    set_seed(42)

    contrastive_criterion = ContrastiveLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.9
    )

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_class_loss = 0.0
        train_contrastive_loss = 0.0
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            
            # Compute class weights for this batch
            class_weights = compute_class_weights(labels, num_classes).to(device)

            classification_criterion = focal_loss(alpha=class_weights, gamma=2, device='cuda')

            optimizer.zero_grad()
            
            features, outputs = model(inputs)
            class_loss = classification_criterion(outputs, labels)
            contrastive_loss = contrastive_criterion(features, labels)
            total_loss = class_loss + contrastive_weight * contrastive_loss

            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            
            train_class_loss += class_loss.item() * inputs.size(0)
            train_contrastive_loss += contrastive_loss.item() * inputs.size(0)
            train_total_loss += total_loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            running_class_loss = train_class_loss / train_total
            running_contrastive_loss = train_contrastive_loss / train_total
            running_total_loss = train_total_loss / train_total
            running_accuracy = 100. * train_correct / train_total

            train_pbar.set_postfix({
                'class_loss': f'{running_class_loss:.4f}',
                'contr_loss': f'{running_contrastive_loss:.4f}',
                'total_loss': f'{running_total_loss:.4f}',
                'acc': f'{running_accuracy:.2f}%'
            })

            if batch_idx % 5 == 0:
                wandb.log({
                    'running_loss': running_total_loss,
                    'running_class_loss': running_class_loss,
                    'running_contr_loss': running_contrastive_loss,
                    'running_train_acc': running_accuracy
                })

        train_class_loss /= len(train_loader.dataset)
        train_contrastive_loss /= len(train_loader.dataset)
        train_total_loss /= len(train_loader.dataset)
        train_accuracy = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_class_loss = 0.0
        val_contrastive_loss = 0.0
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        class_names = [str(i) for i in compartments]

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                class_weights = compute_class_weights(labels, num_classes).to(device)
                classification_criterion = focal_loss(alpha=class_weights, gamma=2, device='cuda')

                features, outputs = model(inputs)
                class_loss = classification_criterion(outputs, labels)
                contrastive_loss = contrastive_criterion(features, labels)
                total_loss = class_loss + contrastive_weight * contrastive_loss

                val_class_loss += class_loss.item() * inputs.size(0)
                val_contrastive_loss += contrastive_loss.item() * inputs.size(0)
                val_total_loss += total_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_class_loss /= len(val_loader.dataset)
        val_contrastive_loss /= len(val_loader.dataset)
        val_total_loss /= len(val_loader.dataset)
        val_accuracy = 100. * val_correct / val_total

        lr_scheduler.step(val_total_loss)

        # Compute and plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm = cm / cm.sum(axis=1)[:,np.newaxis]
        cm_plot = wandb.Image(plot_confusion_matrix(cm, class_names))

        plt_pred_hist = plot_prediction_histogram(model, val_loader)
        score_hist_plot = wandb.Image(plt_pred_hist)

        cm_80 = evaluate_model_confusion_matrix_v2(model, val_loader, len(class_names), class_names, threshold=0.80)
        cm80_plot = wandb.Image(cm_80[-1]) if len(cm_80) == 4 else None
        
        cm_90 = evaluate_model_confusion_matrix_v2(model, val_loader, len(class_names), class_names, threshold=0.90)
        cm90_plot = wandb.Image(cm_90[-1]) if len(cm_90) == 4 else None

        # Generate embeddings and create NCA plot
        embedding_vectors, class_scores, true_labels = generate_embeddings(model, val_loader)
        plt_nca = plot_nca_scatter(embedding_vectors, class_scores, true_labels, threshold=0.8, compartments=compartments)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
           # "lr": lr_scheduler._last_lr()[0],
            "train_class_loss": train_class_loss,
            "train_contrastive_loss": train_contrastive_loss,
            "train_total_loss": train_total_loss,
            "train_accuracy": train_accuracy,
            "val_class_loss": val_class_loss,
            "val_contrastive_loss": val_contrastive_loss,
            "val_total_loss": val_total_loss,
            "val_accuracy": val_accuracy,
            "confusion_matrix": cm_plot,
            "confusion_matrix_80": cm80_plot,
            "confusion_matrix_90": cm90_plot,
            "score_histogram": score_hist_plot
            })
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_filename = os.path.join(run_checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            save_checkpoint(accelerator, model, optimizer, epoch, val_accuracy, checkpoint_filename)
            print(f"Saved new best model with validation accuracy: {val_accuracy:.2f}%")
            
            # Save model to wandb model registry
            artifact = wandb.Artifact(f"best_model_epoch_{epoch+1}", type="model")
            artifact.add_file(checkpoint_filename) # Should change to artifact.add_dir instead to save on space
            wandb.log_artifact(artifact)
        
        print("--------------------")

    return accelerator.unwrap_model(model)

def main():
    # Data paths
    path = '/data/sholtzen/barcode_dynamics/20250725/fish/training.zarr'
    compartments = ['D', 'scr', 'G5', 'V']
    
    # Load data
    z = zarr.open(path)
    all_compartments = list(set(z['train'].y[:]))
    print(all_compartments)
    print(compartments)
    print(len(compartments))
    
    X_train, y_train, mask_train = load_split(z, 'train', compartments)
    X_val, y_val, mask_val = load_split(z, 'val', compartments)
    X_test, y_test, mask_test = load_split(z, 'test', compartments)
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180)
    ])
    transform_test = transforms.Compose([
        transforms.RandomRotation(degrees=0),
    ])
    
    # Create datasets and dataloaders    
    BATCH_SIZE = 32*8
    SHUFFLE = True
    NUM_WORKERS = 4
    
    val_loader = create_dataloader(X_val, y_val, mask_val, transform_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    test_loader = create_dataloader(X_test, y_test, mask_test, transform_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    train_loader = create_dataloader(X_train, y_train, mask_train, transform_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    
    # Define model
    num_classes = len(compartments)
    model = ResNetWithContrastive(num_classes)
    
    # Train model
    trained_model = train_model(model,
                                train_loader,
                                val_loader, 
                                num_classes,
                                compartments,
                                num_epochs=32*4, 
                                learning_rate=2e-3)
    
    return trained_model

if __name__ == "__main__":
    model = main()
