import os
import gc
import math
import random
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast # Keep autocast here for potential future use or explicit call
from skmultilearn.model_selection import IterativeStratification
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss, classification_report
from tqdm import tqdm

# --- 1. Configuration Management ---
class TrainingConfig:
    """
    Manages all configuration parameters for the training process.
    Centralizes hyperparameters, file paths, and advanced training settings.
    """
    MODEL_NAME = 'microsoft/deberta-v3-large'
    TRAIN_FILE_PATH = 'train_essay_aware.csv' # Dataset for essay-aware training
    TEST_FILE_PATH = 'test_essay_aware.csv'   # Dataset for essay-aware testing
    OUTPUT_DIR = './models_essay_aware'       # Output directory for essay-aware models and logs
    LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log_essay_aware.log')

    # Cross-Validation settings
    N_SPLITS = 5 # Consistent with base script: Number of folds for Iterative Stratification

    # Core Training Parameters
    SEED = 42 # Consistent with base script
    N_EPOCHS = 20 # Consistent with base script: Number of training epochs per fold
    # WARNING: Using a larger batch size (32) for essay-aware models might require more GPU memory,
    # especially with MAX_LENGTH=1024. If OOM errors occur, reduce this.
    BATCH_SIZE = 4 # Consistent with base script
    MAX_LENGTH = 1024 # Max sequence length for essay tokenizer (larger for full essays)
    
    # Advanced Learning Techniques
    # Adjusted to ensure balanced sampling is active for most epochs with N_EPOCHS=6.
    # Starting from epoch 1 means it will be used from the second epoch (index 1).
    BALANCED_SAMPLING_START_EPOCH = 5 # Consistent with base script
    EARLY_STOPPING_PATIENCE = 3 # Consistent with base script
    LABEL_SMOOTHING_FACTOR = 0.1 # Consistent with base script
    FOCAL_LOSS_GAMMA = 2.5 # Consistent with base script

    # Advanced Optimizer Parameters
    ENCODER_LR = 1e-5     # Consistent with base script
    DECODER_LR = 1e-4     # Consistent with base script
    WEIGHT_DECAY = 0.01   # Consistent with base script
    GRADIENT_CLIPPING = 1.0 # Consistent with base script
    WARMUP_RATIO = 0.1    # Consistent with base script
    
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
logging.info("Starting Essay-Aware Multi-Label Classification Pipeline")


# --- 3. Essay-Aware Data Structures & Model ---

def build_essay_data(df: pd.DataFrame, label_columns: list) -> list:
    """
    Groups sentences from the dataframe into essays, tracking sentence character spans.
    This function is crucial for the essay-aware approach, as it reconstructs essays
    from individual sentences and their labels.
    """
    essays = []
    # Ensure 'sentence_id' exists, as it's used for sorting within an essay
    if 'sentence_id' not in df.columns:
        df['sentence_id'] = df.groupby('essay_id').cumcount()
    # Sort to ensure sentences are in correct order within each essay
    df = df.sort_values(by=["essay_id", "sentence_id"])
    
    for essay_id, group in tqdm(df.groupby('essay_id'), desc="Building Essay Data"):
        essay_sentences = group['sentence'].astype(str).tolist()
        essay_text = " ".join(essay_sentences) # Reconstruct full essay text
        
        sentence_spans = []
        offset = 0
        for sentence in essay_sentences:
            start_char = offset
            end_char = start_char + len(sentence)
            sentence_spans.append((start_char, end_char))
            offset = end_char + 1 # Account for the space separator between sentences
            
        essays.append({
            "essay_id": essay_id,
            "essay_text": essay_text,
            "sentence_spans": sentence_spans,
            "sentence_labels": group[label_columns].values, # Labels for each sentence in the essay
        })
    return essays

class EssayDataset(Dataset):
    """
    Custom PyTorch Dataset for essay-level data.
    Tokenizes full essays and maps sentence character spans to token spans,
    which is essential for extracting sentence embeddings from the transformer's output.
    """
    def __init__(self, essays: list, tokenizer, max_length: int):
        self.essays = essays
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        item = self.essays[idx]
        # Tokenize the entire essay, returning offset_mapping for span extraction
        encoding = self.tokenizer(
            item["essay_text"],
            return_offsets_mapping=True, # Critical for mapping char spans to token spans
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Pop offset_mapping to prevent it being passed directly to the model
        offset_mapping = encoding.pop("offset_mapping").squeeze(0)
        
        sentence_token_spans = []
        # Map character-level sentence spans to token-level spans
        for sent_char_start, sent_char_end in item["sentence_spans"]:
            token_indices = []
            for i, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                # Check for overlap between token character span and sentence character span
                if max(tok_char_start, sent_char_start) < min(tok_char_end, sent_char_end):
                    token_indices.append(i)
            
            if token_indices:
                # If tokens are found for the sentence, get the start and end token indices
                start_tok_idx, end_tok_idx = token_indices[0], token_indices[-1] + 1
            else: 
                # Handle cases where a sentence might be entirely truncated or very short
                # by assigning a dummy span (0,0) which will be masked later
                start_tok_idx, end_tok_idx = 0, 0 
            sentence_token_spans.append((start_tok_idx, end_tok_idx))
            
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentence_token_spans": sentence_token_spans, # List of (start_token, end_token)
            "sentence_labels": torch.tensor(item["sentence_labels"], dtype=torch.float),
            "num_sentences": len(item["sentence_labels"]) # Number of sentences in the original essay
        }

def collate_fn(batch: list) -> dict:
    """
    Custom collate function for the DataLoader to handle variable numbers of sentences per essay.
    Pads sentence-level labels and token spans to the maximum number of sentences in the batch.
    """
    max_num_sentences = max(x["num_sentences"] for x in batch)
    num_labels = batch[0]["sentence_labels"].shape[1] # Number of classification labels
    
    # Stack input_ids and attention_mask as they are already padded to MAX_LENGTH
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    
    # Initialize padded tensors for labels, token spans, and a valid mask
    # -100.0 for labels indicates padded values that will be ignored by loss
    padded_labels = torch.full((len(batch), max_num_sentences, num_labels), -100.0, dtype=torch.float)
    padded_spans = torch.zeros((len(batch), max_num_sentences, 2), dtype=torch.long)
    valid_mask = torch.zeros((len(batch), max_num_sentences), dtype=torch.bool) # Mask for valid sentences
    
    for i, item in enumerate(batch):
        num_sentences = item["num_sentences"]
        # Fill padded tensors with actual data for valid sentences
        padded_labels[i, :num_sentences, :] = item["sentence_labels"]
        padded_spans[i, :num_sentences] = torch.tensor(item["sentence_token_spans"], dtype=torch.long)
        valid_mask[i, :num_sentences] = True # Mark valid sentences
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentence_token_spans": padded_spans,
        "labels": padded_labels,
        "valid_mask": valid_mask # This mask is crucial for ignoring padded sentences during loss/evaluation
    }

class AttentionPooling(nn.Module):
    """
    Numerically stable Attention Pooling layer.
    Computes a weighted sum of token embeddings, where weights are learned via attention.
    Used to aggregate token embeddings into a single sentence embedding.
    """
    def __init__(self, in_features):
        super().__init__()
        # Attention network: Linear -> Tanh -> Linear (to scalar score)
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features), 
            nn.Tanh(), 
            nn.Linear(in_features, 1, bias=False) # Output a single attention score per token
        )

    def forward(self, token_embeddings, attention_mask):
        # Calculate raw attention logits
        attn_logits = self.attention_net(token_embeddings).squeeze(-1)
        # Mask padded tokens by setting their attention logits to a very small number
        attn_logits.masked_fill_(~attention_mask, torch.finfo(attn_logits.dtype).min)
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_logits, dim=1)
        # Compute weighted sum of token embeddings
        return torch.bmm(attn_weights.unsqueeze(1), token_embeddings).squeeze(1)

class EssayAwareClassifier(nn.Module):
    """
    The main essay-aware model architecture.
    It processes entire essays, extracts sentence embeddings via attention pooling,
    and then uses a BiLSTM to capture contextual relationships between sentences.
    """
    def __init__(self, model_name, num_labels, lstm_dim=256, dropout_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Load the base transformer model (e.g., DeBERTa-v3-large)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooler = AttentionPooling(self.config.hidden_size) # Attention pooling for sentence embeddings
        # BiLSTM layer to capture inter-sentence dependencies
        self.context_layer = nn.LSTM(self.config.hidden_size, lstm_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        # Classifier head
        self.classifier = nn.Linear(lstm_dim * 2, num_labels) # *2 because of bidirectional LSTM output

    def forward(self, input_ids, attention_mask, sentence_token_spans, valid_mask, labels=None):
        # Pass the entire essay through the transformer to get token hidden states
        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        B, L_max, H = hidden_states.shape # Batch size, Max sequence length, Hidden size
        S_max = sentence_token_spans.shape[1] # Max number of sentences in the batch
        
        # Create a mask for tokens belonging to each sentence
        token_indices = torch.arange(L_max, device=hidden_states.device).view(1, 1, L_max)
        start_tokens, end_tokens = sentence_token_spans[..., 0].unsqueeze(-1), sentence_token_spans[..., 1].unsqueeze(-1)
        
        # `sentence_token_mask` is B x S_max x L_max, indicating which tokens belong to which sentence
        sentence_token_mask = (token_indices >= start_tokens) & (token_indices < end_tokens) & (start_tokens < end_tokens)
        # Combine with valid_mask from collate_fn to ignore padded sentences
        sentence_token_mask &= valid_mask.unsqueeze(-1)

        # Reshape hidden_states and masks for vectorized attention pooling
        # This expands hidden states to apply pooling across each sentence
        expanded_hidden = hidden_states.unsqueeze(1).expand(-1, S_max, -1, -1).reshape(B * S_max, L_max, H)
        expanded_mask = sentence_token_mask.view(B * S_max, L_max)
        
        # Apply attention pooling to get a single embedding for each sentence
        pooled_embeddings = self.pooler(expanded_hidden, expanded_mask)
        # Reshape back to (Batch_size, Max_sentences_in_batch, Hidden_size)
        sentence_embeddings = pooled_embeddings.view(B, S_max, H)
        
        # Get actual number of sentences per essay for packing sequence
        num_sentences = valid_mask.sum(dim=1).cpu() 
        # Pack padded sequence to handle variable sentence lengths in BiLSTM efficiently
        packed_input = nn.utils.rnn.pack_padded_sequence(sentence_embeddings, num_sentences, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.context_layer(packed_input)
        # Pad back to original S_max length
        contextual_embeddings, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=S_max)
        
        # Apply dropout and pass through classifier
        logits = self.classifier(self.dropout(contextual_embeddings))
        
        loss = None
        if labels is not None:
            # The FocalLoss here already handles `valid_mask` and label smoothing internally
            loss_fct = FocalLoss(gamma=TrainingConfig.FOCAL_LOSS_GAMMA)
            loss = loss_fct(logits, labels, valid_mask) # Pass valid_mask to FocalLoss

        return {'loss': loss, 'logits': logits}

class FocalLoss(nn.Module):
    """
    Focal Loss adapted for padded sequences using a valid_mask.
    This version correctly applies the loss only to the valid (non-padded) sentences.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, valid_mask):
        # Flatten inputs and targets and apply the valid_mask
        inputs_flat = inputs[valid_mask]
        targets_flat = targets[valid_mask]
        
        # If there are no valid elements (e.g., an empty essay after truncation), return 0 loss
        if targets_flat.numel() == 0: 
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
        # Apply label smoothing if configured
        if TrainingConfig.LABEL_SMOOTHING_FACTOR > 0:
            targets_flat = targets_flat * (1 - TrainingConfig.LABEL_SMOOTHING_FACTOR) + 0.5 * TrainingConfig.LABEL_SMOOTHING_FACTOR
        
        # Calculate BCE loss per element
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none')
        
        # Calculate pt (probability of correct classification)
        pt = torch.exp(-bce_loss)
        
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean() # Return mean loss across all valid elements

class EarlyStopping:
    """Standard EarlyStopping implementation."""
    def __init__(self, patience=5, path='checkpoint.pt'):
        self.patience=patience
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.path=path

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            # Save the model state dict
            torch.save(model.state_dict(), self.path)
            logging.info(f"Validation score improved ({self.best_score:.4f}). Saving model to {self.path}")
            self.counter = 0 # Reset counter on improvement
        else:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: 
                self.early_stop = True # Trigger early stopping

# --- 4. Core Training & Evaluation ---
def train_fn(data_loader, model, optimizer, device, scheduler, scaler):
    """
    Performs one epoch of training for the Essay-Aware model.
    Args:
        data_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The EssayAwareClassifier model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): Device (CPU/GPU) to run training on.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        scaler (GradScaler): For mixed-precision training.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        # Move all batch components to the correct device
        batch_args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        # Use mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(**batch_args)
            loss = outputs['loss']

        # Check for NaN or infinite loss to prevent silent failures
        if not math.isfinite(loss.item()): 
            logging.error("Loss is NaN or infinite. Stopping training.")
            raise ValueError("Loss is NaN or infinite.")

        scaler.scale(loss).backward() # Scale loss and backpropagate
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIPPING)
        
        scaler.step(optimizer) # Optimizer step
        scaler.update() # Update scaler for next iteration
        scheduler.step() # Learning rate scheduler step
        total_loss += loss.item() # Accumulate loss

    return total_loss / len(data_loader) # Return average loss for the epoch

def eval_fn(data_loader, model, device):
    """
    Evaluates the Essay-Aware model on a given data loader.
    Args:
        data_loader (DataLoader): DataLoader for evaluation data.
        model (nn.Module): The EssayAwareClassifier model to evaluate.
        device (torch.device): Device (CPU/GPU) to run evaluation on.
    Returns:
        tuple: (numpy.ndarray of predictions, numpy.ndarray of true targets)
    """
    model.eval() # Set model to evaluation mode
    final_targets, final_outputs = [], []

    with torch.no_grad(): # Disable gradient calculation
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Prepare batch arguments, excluding 'labels' for evaluation forward pass
            batch_args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'labels'}
            
            with torch.cuda.amp.autocast():
                outputs = model(**batch_args)

            logits = outputs['logits']
            valid_mask = batch['valid_mask'].to(device) # Ensure mask is on device

            # Collect targets and sigmoid probabilities only for valid (non-padded) sentences
            final_targets.append(batch['labels'][valid_mask.cpu()].numpy()) # Targets are on CPU
            final_outputs.append(torch.sigmoid(logits[valid_mask]).cpu().numpy()) # Logits on device, then sigmoid, then CPU

    return np.concatenate(final_outputs), np.concatenate(final_targets)

# --- 5. Main Training Flow ---
def train_fold(fold, train_essays, val_essays, tokenizer, label_columns):
    """
    Trains and evaluates the Essay-Aware model for a single cross-validation fold.
    Args:
        fold (int): Current fold number (0-indexed).
        train_essays (list): List of processed essay data for training in this fold.
        val_essays (list): List of processed essay data for validation in this fold.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
        label_columns (list): List of column names representing labels.
    Returns:
        tuple: (numpy.ndarray of validation predictions, numpy.ndarray of validation targets)
    """
    logging.info(f"========== FOLD {fold + 1}/{TrainingConfig.N_SPLITS} ==========")
    train_dataset = EssayDataset(train_essays, tokenizer, TrainingConfig.MAX_LENGTH)
    val_dataset = EssayDataset(val_essays, tokenizer, TrainingConfig.MAX_LENGTH)
    
    # Per-essay weighted sampler: weights essays based on the frequency of labels within their sentences.
    # This helps ensure essays containing rarer labels are sampled more frequently.
    all_sentence_labels = np.vstack([e['sentence_labels'] for e in train_essays])
    label_frequencies = all_sentence_labels.sum(axis=0) # Sum occurrences of each label across all sentences
    essay_weights = []
    for essay in train_essays:
        essay_labels_sum = essay['sentence_labels'].sum(axis=0) # Sum of labels present in current essay's sentences
        if essay_labels_sum.sum() == 0: 
            # If an essay has no positive labels, assign a baseline weight
            weight = 1.0 / len(train_essays)
        else:
            # Find the minimum frequency among the labels present in this essay.
            # This ensures that essays with even one rare label get a boost.
            min_freq = label_frequencies[essay_labels_sum > 0].min()
            weight = 1.0 / min_freq if min_freq > 0 else 1.0 / len(train_essays) # Avoid division by zero
        essay_weights.append(weight)
        
    sampler = WeightedRandomSampler(weights=essay_weights, num_samples=len(essay_weights), replacement=True)
    
    # DataLoaders using the custom collate_fn
    train_loader_standard = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train_loader_balanced = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize the EssayAwareClassifier model
    model = EssayAwareClassifier(TrainingConfig.MODEL_NAME, num_labels=len(label_columns)).to(TrainingConfig.DEVICE)
    
    # Define optimizer with separate learning rates for transformer (encoder) and other layers (decoder)
    encoder_params = [p for n, p in model.named_parameters() if n.startswith('transformer.')]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith('transformer.')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': TrainingConfig.ENCODER_LR}, 
        {'params': decoder_params, 'lr': TrainingConfig.DECODER_LR}
    ], weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # Setup learning rate scheduler
    num_training_steps = len(train_loader_standard) * TrainingConfig.N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(TrainingConfig.WARMUP_RATIO * num_training_steps), 
        num_training_steps=num_training_steps
    )
    scaler = GradScaler() # For Automatic Mixed Precision
    
    # Setup Early Stopping
    best_model_path = os.path.join(TrainingConfig.OUTPUT_DIR, f'best_model_fold_{fold}.pth')
    early_stopper = EarlyStopping(patience=TrainingConfig.EARLY_STOPPING_PATIENCE, path=best_model_path)
    
    # Training loop for the fold
    for epoch in range(TrainingConfig.N_EPOCHS):
        logging.info(f"--- Epoch {epoch + 1}/{TrainingConfig.N_EPOCHS} ---")
        
        # Switch between standard and balanced loader based on configured epoch
        current_train_loader = train_loader_balanced if epoch >= TrainingConfig.BALANCED_SAMPLING_START_EPOCH else train_loader_standard
        if epoch >= TrainingConfig.BALANCED_SAMPLING_START_EPOCH: 
            logging.info("Using essay-level class-balanced sampler.")
        
        # Perform training and evaluation
        avg_train_loss = train_fn(current_train_loader, model, optimizer, TrainingConfig.DEVICE, scheduler, scaler)
        logging.info(f"Fold {fold+1} Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")
        
        outputs, targets = eval_fn(val_loader, model, TrainingConfig.DEVICE)
        macro_f1 = f1_score(targets, outputs >= 0.5, average='macro', zero_division=0)
        logging.info(f"Fold {fold+1} Epoch {epoch+1} Validation Macro F1 (at 0.5 thresh): {macro_f1:.4f}")
        
        early_stopper(macro_f1, model) # Check for early stopping
        if early_stopper.early_stop:
            logging.info("Early stopping triggered."); 
            break # Break out of the epoch loop
    
    logging.info(f"Loading best model for fold {fold+1} from {best_model_path}")
    # SECURITY FIX: Use weights_only=True to prevent loading malicious code.
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return eval_fn(val_loader, model, TrainingConfig.DEVICE) # Return evaluation on best model

def main():
    """
    Main function to orchestrate the entire essay-aware multi-label classification pipeline.
    Handles data loading, cross-validation, training, ensembling, and final evaluation.
    """
    logging.info("--- Starting Essay-Aware Cross-Validation Ensemble Training ---") # Updated log message
    
    # Load training data and identify label columns
    full_train_df = pd.read_csv(TrainingConfig.TRAIN_FILE_PATH)
    label_columns = [col for col in full_train_df.columns if col not in ['essay_id', 'sentence_id', 'sentence']]
    
    # Build essay-level data structures
    essays = build_essay_data(full_train_df, label_columns)
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
    
    # Prepare stratification labels: True if essay has at least one positive label for a given class
    # This ensures balanced distribution of essays across folds, especially for multi-label data.
    y_stratify = np.array([np.any(e['sentence_labels'], axis=0) for e in essays], dtype=int)
    kfold = IterativeStratification(n_splits=TrainingConfig.N_SPLITS, order=1)
    
    oof_outputs, oof_targets = [], [] # Lists to store Out-Of-Fold predictions and targets
    essays_np = np.array(essays, dtype=object) # Convert to numpy array for easier indexing
    
    # Perform cross-validation
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X=np.zeros(len(essays_np)), y=y_stratify)):
        train_essays, val_essays = essays_np[train_indices].tolist(), essays_np[val_indices].tolist()
        # Train fold and get OOF predictions
        fold_outputs, fold_targets = train_fold(fold, train_essays, val_essays, tokenizer, label_columns)
        oof_outputs.append(fold_outputs)
        oof_targets.append(fold_targets)
        # Clean up GPU memory after each fold
        gc.collect(); torch.cuda.empty_cache()

    logging.info("--- Overall Cross-Validation Results ---")
    all_oof_outputs = np.concatenate(oof_outputs)
    all_oof_targets = np.concatenate(oof_targets)
    cv_macro_f1 = f1_score(all_oof_targets, all_oof_outputs >= 0.5, average='macro', zero_division=0)
    logging.info(f"Overall CV Macro F1-Score (at 0.5 thresh): {cv_macro_f1:.4f}")

    logging.info("--- Finding Optimal Thresholds on OOF Predictions ---")
    optimal_thresholds = {}
    # Determine best threshold for each label based on OOF predictions
    for i, label in enumerate(label_columns):
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(all_oof_targets[:, i], all_oof_outputs[:, i] >= thresh, zero_division=0)
            if f1 > best_f1: 
                best_f1, best_thresh = f1, thresh
        optimal_thresholds[label] = round(best_thresh, 2) # Round threshold for cleaner logging
        logging.info(f"Optimal threshold for {label}: {optimal_thresholds[label]:.2f} (OOF F1: {best_f1:.4f})")

    logging.info("--- Evaluating Ensemble on Hold-Out Test Set ---")
    test_df = pd.read_csv(TrainingConfig.TEST_FILE_PATH)
    test_essays = build_essay_data(test_df, label_columns)
    test_dataset = EssayDataset(test_essays, tokenizer, TrainingConfig.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    models = []
    # Load all best models from each fold for ensembling
    for fold in range(TrainingConfig.N_SPLITS):
        model = EssayAwareClassifier(TrainingConfig.MODEL_NAME, num_labels=len(label_columns))
        model_path = os.path.join(TrainingConfig.OUTPUT_DIR, f'best_model_fold_{fold}.pth')
        # SECURITY FIX: Use weights_only=True to prevent loading malicious code.
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(TrainingConfig.DEVICE); model.eval() # Move to device and set to eval mode
        models.append(model)
        
    final_test_outputs, final_test_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Predicting"):
            # Prepare batch arguments, excluding 'labels' for inference
            batch_args = {k: v.to(TrainingConfig.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'labels'}
            
            # Get predictions from each model and average them
            batch_preds = [torch.sigmoid(model(**batch_args)['logits']) for model in models]
            avg_preds = torch.stack(batch_preds).mean(dim=0)
            
            valid_mask = batch['valid_mask'].to(TrainingConfig.DEVICE) # Ensure mask is on device
            
            # Collect true labels and predictions only for valid sentences
            final_test_targets.append(batch['labels'][valid_mask.cpu()].numpy())
            final_test_outputs.append(avg_preds[valid_mask].cpu().numpy())
    
    # Concatenate all predictions and targets for final evaluation
    final_test_outputs = np.concatenate(final_test_outputs)
    final_test_targets = np.concatenate(final_test_targets)
    
    # Apply optimal thresholds to generate final binary predictions
    final_predictions = np.zeros_like(final_test_outputs)
    for i, label in enumerate(label_columns):
        final_predictions[:, i] = (final_test_outputs[:, i] >= optimal_thresholds[label]).astype(int)

    logging.info("--- Final ENSEMBLE Performance on Test Set ---")
    # Log final performance metrics
    logging.info(f"Final Ensemble Macro F1-Score: {f1_score(final_test_targets, final_predictions, average='macro', zero_division=0):.4f}")
    logging.info(f"Final Ensemble Micro F1-Score: {f1_score(final_test_targets, final_predictions, average='micro', zero_division=0):.4f}")
    logging.info(f"Final Ensemble Hamming Loss: {hamming_loss(final_test_targets, final_predictions):.4f}")
    logging.info("\n" + classification_report(final_test_targets, final_predictions, target_names=label_columns, zero_division=0))

if __name__ == '__main__':
    main()
