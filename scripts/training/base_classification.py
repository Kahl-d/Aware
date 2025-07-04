import os
import gc
import random
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from skmultilearn.model_selection import IterativeStratification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss, classification_report
from tqdm import tqdm

# --- 1. Configuration Management ---
class TrainingConfig:
    """
    Manages all configuration parameters for the training process.
    Centralizes hyperparameters, file paths, and advanced training settings.
    """
    MODEL_NAME = 'microsoft/deberta-v3-large'
    TRAIN_FILE_PATH = 'train.csv' # This will be split into 3 folds for cross-validation
    TEST_FILE_PATH = 'test.csv'   # This is the final hold-out set for evaluation
    OUTPUT_DIR = './proto'        # Directory to save models and logs
    LOG_FILE = os.path.join(OUTPUT_DIR, 'training_proto.log')

    # Cross-Validation settings
    N_SPLITS = 5 # Updated: Number of folds for Iterative Stratification

    # Core Training Parameters
    SEED = 42
    N_EPOCHS = 20 # Updated: Number of training epochs per fold
    BATCH_SIZE = 32
    MAX_LENGTH = 512 # Max sequence length for tokenizer

    # Advanced Learning Techniques
    # Changed to ensure balanced sampling is active for most epochs with N_EPOCHS=6.
    # Starting from epoch 1 means it will be used from the second epoch (index 1).
    BALANCED_SAMPLING_START_EPOCH = 5 
    EARLY_STOPPING_PATIENCE = 3 # Number of epochs to wait for improvement before stopping
    LABEL_SMOOTHING_FACTOR = 0.1 # Factor for label smoothing to prevent overconfidence
    FOCAL_LOSS_GAMMA = 2.5 # Gamma parameter for Focal Loss, controls focus on hard examples

    # Advanced Optimizer Parameters
    ENCODER_LR = 1e-5     # Learning rate for the base encoder (e.g., DeBERTa layers)
    DECODER_LR = 1e-4     # Learning rate for the classification head (decoder)
    WEIGHT_DECAY = 0.01   # L2 regularization to prevent overfitting
    GRADIENT_CLIPPING = 1.0 # Max gradient norm to prevent exploding gradients
    WARMUP_RATIO = 0.1    # Proportion of total steps for learning rate warm-up

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory if it doesn't exist
os.makedirs(TrainingConfig.OUTPUT_DIR, exist_ok=True)

# --- 2. Logging & Reproducibility ---
# Configure logging to output to both a file and the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                    handlers=[logging.FileHandler(TrainingConfig.LOG_FILE), logging.StreamHandler()])

def set_seed(seed):
    """
    Sets the random seed for reproducibility across multiple libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN, potentially sacrificing some performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Apply the global seed setting
set_seed(TrainingConfig.SEED)
logging.info(f"Seed set to {TrainingConfig.SEED}")

# --- 3. Custom Dataset and Loss Functions ---
class ThemeDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text sentences and their multi-label themes.
    Tokenizes input sentences and prepares them for model input.
    """
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.sentences = df['sentence'].values
        # Labels are all columns except 'sentence'
        self.labels = df.drop('sentence', axis=1).values

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a single item (sentence and its labels) by index.
        Tokenizes the sentence and returns input_ids, attention_mask, and labels.
        """
        sentence = self.sentences[index]
        labels = self.labels[index]

        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,    # Add [CLS] and [SEP]
            max_length=self.max_len,    # Pad/truncate to max_length
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate if longer than max_length
            return_tensors='pt'         # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float) # Labels as float for BCEWithLogitsLoss
        }

class FocalLoss(nn.Module):
    """
    Implements Focal Loss, which is particularly effective for highly imbalanced datasets.
    It down-weights easy examples and focuses training on hard, misclassified examples.
    Includes optional label smoothing.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for positive/negative examples
        self.gamma = gamma  # Focusing parameter to down-weight easy examples
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') # Base BCE loss

    def forward(self, inputs, targets):
        """
        Calculates the Focal Loss.
        Args:
            inputs (torch.Tensor): Raw logits from the model.
            targets (torch.Tensor): True labels (0 or 1).
        Returns:
            torch.Tensor: The scalar Focal Loss.
        """
        # Apply label smoothing if configured
        if TrainingConfig.LABEL_SMOOTHING_FACTOR > 0:
            targets = targets * (1 - TrainingConfig.LABEL_SMOOTHING_FACTOR) + 0.5 * TrainingConfig.LABEL_SMOOTHING_FACTOR
        
        # Calculate BCE loss per element
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate pt (probability of correct classification)
        pt = torch.exp(-bce_loss)
        
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean() # Return mean loss across all elements

class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Saves the best model weights.
    """
    def __init__(self, patience=5, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last validation score improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_score, model):
        """
        Checks current validation score against the best observed score.
        If improved, saves the model; otherwise, increments counter and checks for early stop.
        """
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            # Save model state dictionary for later loading
            torch.save(model.state_dict(), self.path)
            logging.info(f"Validation score improved. Saving model to {self.path}")
            self.counter = 0 # Reset counter on improvement
        else:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True # Trigger early stopping

# --- 4. Core Training and Evaluation Functions ---
def train_fn(data_loader, model, optimizer, device, scheduler, scaler):
    """
    Performs one epoch of training.
    Args:
        data_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): Device (CPU/GPU) to run training on.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        scaler (GradScaler): For mixed-precision training.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0

    # Iterate over batches with a progress bar
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Clear gradients from previous step

        # Enable mixed precision training
        with autocast():
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # Loss is calculated internally by the model if labels are provided,
            # but we use our custom FocalLoss assigned to model.loss_fct
            loss = model.loss_fct(outputs.logits, labels)
            
        # Scale loss for mixed precision and backpropagate
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping (important for AMP)
        scaler.unscale_(optimizer)
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIPPING)
        
        # Optimizer step and scheduler step for mixed precision
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item() # Accumulate loss

    return total_loss / len(data_loader) # Return average loss

def eval_fn(data_loader, model, device):
    """
    Evaluates the model on a given data loader.
    Args:
        data_loader (DataLoader): DataLoader for evaluation data.
        model (nn.Module): The model to evaluate.
        device (torch.device): Device (CPU/GPU) to run evaluation on.
    Returns:
        tuple: (numpy.ndarray of predictions, numpy.ndarray of true targets)
    """
    model.eval() # Set model to evaluation mode
    final_targets, final_outputs = [], []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # Keep labels for consistency, though not used in forward pass here

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Apply sigmoid to logits to get probabilities, then move to CPU and convert to numpy
            final_outputs.extend(torch.sigmoid(outputs.logits).cpu().numpy())
            final_targets.extend(labels.cpu().numpy()) # Store true labels

    return np.array(final_outputs), np.array(final_targets)

# --- 5. Single Fold Training Function ---
def train_fold(fold, train_df, val_df, tokenizer, label_columns):
    """
    Trains and evaluates the model for a single cross-validation fold.
    Args:
        fold (int): Current fold number (0-indexed).
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
        label_columns (list): List of column names representing labels.
    Returns:
        tuple: (numpy.ndarray of validation predictions, numpy.ndarray of validation targets)
    """
    logging.info(f"========== FOLD {fold + 1}/{TrainingConfig.N_SPLITS} ==========")
    
    train_dataset = ThemeDataset(train_df, tokenizer, TrainingConfig.MAX_LENGTH)
    val_dataset = ThemeDataset(val_df, tokenizer, TrainingConfig.MAX_LENGTH)

    # Setup for Class-Balanced Sampler
    # Calculate inverse frequency weights for each sample based on its labels.
    # Samples with rarer labels will get higher weights.
    label_frequencies = train_df[label_columns].sum() # Sum occurrences of each label
    weights = train_df[label_columns].apply(
        lambda r: 1.0 / label_frequencies[r.astype(bool)].min() if r.sum() > 0 else 1.0 / len(train_df),
        axis=1
    )
    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(weights), # Number of samples to draw in an epoch
        replacement=True # Sample with replacement
    )
    
    # Define data loaders: one standard, one with balanced sampling
    train_loader_standard = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    train_loader_balanced = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)
    
    # Load the pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        TrainingConfig.MODEL_NAME, 
        num_labels=len(label_columns), 
        problem_type="multi_label_classification"
    )
    model.to(TrainingConfig.DEVICE) # Move model to appropriate device

    # Assign custom Focal Loss to the model
    model.loss_fct = FocalLoss(gamma=TrainingConfig.FOCAL_LOSS_GAMMA).to(TrainingConfig.DEVICE)
    
    # Define optimizer with separate learning rates for encoder and decoder parts
    encoder_params = [p for n, p in model.named_parameters() if n.startswith('deberta.')]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith('deberta.')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': TrainingConfig.ENCODER_LR}, 
        {'params': decoder_params, 'lr': TrainingConfig.DECODER_LR}
    ], weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # Calculate total training steps for the learning rate scheduler
    num_training_steps = len(train_loader_standard) * TrainingConfig.N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(TrainingConfig.WARMUP_RATIO * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler() # Initialize GradScaler for Automatic Mixed Precision (AMP)
    
    # Setup Early Stopping
    best_model_path = os.path.join(TrainingConfig.OUTPUT_DIR, f'best_model_fold_{fold}.pth')
    early_stopper = EarlyStopping(patience=TrainingConfig.EARLY_STOPPING_PATIENCE, path=best_model_path)
    
    # Training loop for the current fold
    for epoch in range(TrainingConfig.N_EPOCHS):
        logging.info(f"--- Epoch {epoch + 1}/{TrainingConfig.N_EPOCHS} ---")
        
        # Switch to balanced sampler after the specified epoch
        current_train_loader = train_loader_balanced if epoch >= TrainingConfig.BALANCED_SAMPLING_START_EPOCH else train_loader_standard
        if epoch >= TrainingConfig.BALANCED_SAMPLING_START_EPOCH: 
            logging.info("Using class-balanced sampler.")
        
        # Train and evaluate
        avg_train_loss = train_fn(current_train_loader, model, optimizer, TrainingConfig.DEVICE, scheduler, scaler)
        logging.info(f"Fold {fold+1} Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")
        
        outputs, targets = eval_fn(val_loader, model, TrainingConfig.DEVICE)
        # Calculate Macro F1-score on validation set
        macro_f1 = f1_score(targets, outputs >= 0.5, average='macro', zero_division=0)
        logging.info(f"Fold {fold+1} Epoch {epoch+1} Validation Macro F1 (at 0.5 thresh): {macro_f1:.4f}")
        
        # Check for early stopping
        early_stopper(macro_f1, model)
        if early_stopper.early_stop:
            logging.info("Early stopping triggered."); 
            break # Exit epoch loop if early stopping is triggered
    
    # After training epochs, load the best saved model for final evaluation on this fold's validation set
    model.load_state_dict(torch.load(best_model_path))
    outputs, targets = eval_fn(val_loader, model, TrainingConfig.DEVICE)
    
    # Return raw outputs (probabilities) and true targets for overall analysis (OOF predictions)
    return outputs, targets

# --- 6. Main Execution Block ---
def main():
    """
    Main function to run the entire multi-label text classification pipeline.
    Handles data loading, cross-validation, training, ensembling, and final evaluation.
    """
    logging.info("--- Starting 5-Fold Cross-Validation Ensemble Training ---")
    
    # Load training and test data
    full_train_df = pd.read_csv(TrainingConfig.TRAIN_FILE_PATH)
    test_df = pd.read_csv(TrainingConfig.TEST_FILE_PATH)
    
    # Identify label columns (all columns except 'sentence')
    label_columns = [col for col in full_train_df.columns if col != 'sentence']
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
    
    # Cross-validation setup using Iterative Stratification for multi-label data
    kfold = IterativeStratification(n_splits=TrainingConfig.N_SPLITS, order=1)
    
    oof_outputs = [] # To store out-of-fold predictions
    oof_targets = [] # To store out-of-fold true targets

    # Iterate through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X=full_train_df, y=full_train_df[label_columns].values)):
        train_fold_df = full_train_df.iloc[train_indices]
        val_fold_df = full_train_df.iloc[val_indices]
        
        # Train the model for the current fold and get OOF predictions
        fold_outputs, fold_targets = train_fold(fold, train_fold_df, val_fold_df, tokenizer, label_columns)
        
        oof_outputs.append(fold_outputs)
        oof_targets.append(fold_targets)

    # --- 7. Overall Cross-Validation Performance ---
    logging.info("--- Overall Cross-Validation Results ---")
    # Concatenate all OOF predictions and targets
    all_oof_outputs = np.concatenate(oof_outputs)
    all_oof_targets = np.concatenate(oof_targets)
    
    # Calculate overall Macro F1-score from OOF predictions using default 0.5 threshold
    cv_macro_f1 = f1_score(all_oof_targets, all_oof_outputs >= 0.5, average='macro', zero_division=0)
    logging.info(f"Overall CV Macro F1-Score (at 0.5 thresh): {cv_macro_f1:.4f}")

    # --- 8. Final Ensemble Evaluation on Test Set ---
    logging.info("--- Evaluating Ensemble on Hold-Out Test Set ---")
    
    # Load all trained models (one from each fold) for ensembling
    models = []
    for fold in range(TrainingConfig.N_SPLITS):
        model = AutoModelForSequenceClassification.from_pretrained(
            TrainingConfig.MODEL_NAME, 
            num_labels=len(label_columns), 
            problem_type="multi_label_classification"
        )
        # Load the best state dictionary saved during training for each fold
        model.load_state_dict(torch.load(os.path.join(TrainingConfig.OUTPUT_DIR, f'best_model_fold_{fold}.pth')))
        model.to(TrainingConfig.DEVICE)
        model.eval() # Set to evaluation mode
        models.append(model)
        
    test_dataset = ThemeDataset(test_df, tokenizer, TrainingConfig.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    # Get predictions by averaging model outputs from the ensemble
    all_avg_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Predicting"):
            input_ids = batch['input_ids'].to(TrainingConfig.DEVICE)
            attention_mask = batch['attention_mask'].to(TrainingConfig.DEVICE)
            
            # Collect predictions from all models in the ensemble
            batch_preds = [torch.sigmoid(model(input_ids, attention_mask=attention_mask).logits) for model in models]
            # Average the probabilities across all models for the current batch
            avg_preds = torch.stack(batch_preds).mean(dim=0)
            all_avg_preds.append(avg_preds.cpu().numpy())
            
    final_test_outputs = np.concatenate(all_avg_preds) # Final ensemble probabilities on test set
    final_test_targets = test_df[label_columns].values # True labels for the test set

    # Find optimal thresholds based on the OOF predictions
    logging.info("--- Finding Optimal Thresholds on OOF Predictions ---")
    optimal_thresholds = {label: 0.5 for label in label_columns} # Initialize with default 0.5
    for i, label in enumerate(label_columns):
        best_f1, best_thresh = 0, 0.5
        # Iterate through a range of thresholds to find the best F1-score for each label
        for thresh in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(all_oof_targets[:, i], all_oof_outputs[:, i] >= thresh, zero_division=0)
            if f1 > best_f1: 
                best_f1, best_thresh = f1, thresh
        optimal_thresholds[label] = best_thresh
        logging.info(f"Optimal threshold for {label}: {best_thresh:.2f} (OOF F1: {best_f1:.4f})")

    # Apply optimal thresholds to ensemble test predictions
    final_predictions = np.zeros_like(final_test_outputs)
    for i, label in enumerate(label_columns):
        # Apply the optimal threshold found for each label
        final_predictions[:, i] = (final_test_outputs[:, i] >= optimal_thresholds[label]).astype(int)

    logging.info("--- Final ENSEMBLE Performance on Test Set ---")
    # Calculate and log overall ensemble performance metrics
    logging.info(f"Final Ensemble Macro F1-Score: {f1_score(final_test_targets, final_predictions, average='macro', zero_division=0):.4f}")
    logging.info(f"Final Ensemble Micro F1-Score: {f1_score(final_test_targets, final_predictions, average='micro', zero_division=0):.4f}")
    logging.info(f"Final Ensemble Hamming Loss: {hamming_loss(final_test_targets, final_predictions):.4f}")
    
    # Generate and log a detailed classification report for each label
    logging.info("Final Ensemble Test Set Classification Report:\n" + 
                 classification_report(final_test_targets, final_predictions, target_names=label_columns, zero_division=0))

if __name__ == '__main__':
    main()
