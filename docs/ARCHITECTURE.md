# 🏗️ Aware Architecture Documentation

This document provides a detailed technical overview of the Aware framework's architecture, components, and implementation details.

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Domain-Adaptive Pre-training (DAPT)](#domain-adaptive-pre-training-dapt)
3. [Essay-Aware Architecture](#essay-aware-architecture)
4. [Multi-Label Classification](#multi-label-classification)
5. [Data Flow](#data-flow)
6. [Model Components](#model-components)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation Framework](#evaluation-framework)

## 🎯 System Overview

The Aware framework is designed as a three-stage pipeline that progressively enhances the model's understanding of Cultural Capital Themes (CCTs) in student essays:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Stage 1:      │    │   Stage 2:       │    │   Stage 3:      │
│   Domain-Adapt  │───▶│  Essay-Aware     │───▶│ Multi-Label     │
│   Pre-training  │    │  Architecture    │    │ Classification  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Design Principles

1. **Domain Awareness**: Adapt to student essay language patterns
2. **Context Preservation**: Maintain essay-level narrative flow
3. **Multi-Label Handling**: Address theme overlap and co-occurrence
4. **Scalability**: Support large-scale educational applications

## 🔄 Domain-Adaptive Pre-training (DAPT)

### Purpose
Adapt the base language model (DeBERTa-v3-large) to the specific linguistic patterns found in student essays from STEM classrooms.

### Implementation Details

#### Model Configuration
```python
MODEL_NAME = 'microsoft/deberta-v3-large'
OUTPUT_DIR = './deberta-v3-large-essays-adapted-final'
VALIDATION_SPLIT_PERCENTAGE = 5
```

#### Training Parameters
```python
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=torch.cuda.is_bf16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

#### Data Processing
- **Input**: Raw student essays from STEM classrooms
- **Tokenization**: SentencePiece tokenizer with 512 token limit
- **Masking**: 15% MLM probability for masked language modeling
- **Validation**: 5% holdout for perplexity evaluation

### Evaluation Metrics
- **Perplexity**: Measures model's understanding of domain language
- **Qualitative Analysis**: Masked word prediction quality
- **Domain-Specific Vocabulary**: Adaptation to educational terminology

## 🧠 Essay-Aware Architecture

### Core Innovation
The essay-aware architecture addresses the context dependency challenge by processing entire essays rather than isolated sentences.

### Component Breakdown

#### 1. Essay Reconstruction
```python
def build_essay_data(df: pd.DataFrame, label_columns: list) -> list:
    """
    Groups sentences into essays with character span tracking.
    """
    essays = []
    for essay_id, group in df.groupby('essay_id'):
        essay_text = " ".join(group['sentence'].tolist())
        sentence_spans = calculate_character_spans(essay_text, group['sentence'])
        essays.append({
            "essay_id": essay_id,
            "essay_text": essay_text,
            "sentence_spans": sentence_spans,
            "sentence_labels": group[label_columns].values
        })
    return essays
```

#### 2. Character-to-Token Mapping
```python
def map_char_spans_to_token_spans(offset_mapping, sentence_spans):
    """
    Maps character-level sentence spans to token-level spans.
    """
    sentence_token_spans = []
    for sent_char_start, sent_char_end in sentence_spans:
        token_indices = []
        for i, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            if max(tok_char_start, sent_char_start) < min(tok_char_end, sent_char_end):
                token_indices.append(i)
        
        if token_indices:
            start_tok_idx, end_tok_idx = token_indices[0], token_indices[-1] + 1
        else:
            start_tok_idx, end_tok_idx = 0, 0
        sentence_token_spans.append((start_tok_idx, end_tok_idx))
    
    return sentence_token_spans
```

#### 3. Attention Pooling Mechanism
```python
class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Linear(in_features, 1)
    
    def forward(self, token_embeddings, attention_mask):
        # Calculate attention scores
        attention_scores = self.attention(token_embeddings).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted average of token embeddings
        sentence_embedding = torch.sum(attention_weights.unsqueeze(-1) * token_embeddings, dim=1)
        return sentence_embedding
```

#### 4. BiLSTM Context Layer
```python
class EssayAwareClassifier(nn.Module):
    def __init__(self, model_name, num_labels, lstm_dim=256, dropout_prob=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.attention_pooling = AttentionPooling(self.transformer.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=lstm_dim,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(lstm_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout_prob)
```

### Data Flow in Essay-Aware Model

```
Input Essay: "I am here because... [full essay text]"
    ↓
Tokenization: [CLS] I am here because... [SEP]
    ↓
Transformer Encoding: Hidden states for all tokens
    ↓
Sentence Span Extraction: Map character spans to token spans
    ↓
Attention Pooling: Weighted token embeddings per sentence
    ↓
BiLSTM Processing: Contextual sentence representations
    ↓
Classification: Multi-label predictions per sentence
```

## 🏷️ Multi-Label Classification

### Problem Formulation
Given a sentence $s$ from essay $e$, predict a binary vector $y \in \{0,1\}^{12}$ where each element indicates the presence of a Cultural Capital Theme.

### Loss Function: Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets, valid_mask):
        # Apply label smoothing
        if TrainingConfig.LABEL_SMOOTHING_FACTOR > 0:
            targets = targets * (1 - TrainingConfig.LABEL_SMOOTHING_FACTOR) + 0.5 * TrainingConfig.LABEL_SMOOTHING_FACTOR
        
        # Calculate focal loss
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        # Apply valid mask
        focal_loss = focal_loss * valid_mask.unsqueeze(-1)
        return focal_loss.mean()
```

### Class Imbalance Handling

#### 1. Weighted Sampling
```python
def create_weighted_sampler(dataset, labels):
    """
    Creates weighted sampler to handle class imbalance.
    """
    class_counts = labels.sum(axis=0)
    class_weights = 1.0 / class_counts
    sample_weights = labels.dot(class_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights))
```

#### 2. Optimal Threshold Tuning
```python
def find_optimal_thresholds(oof_predictions, oof_labels):
    """
    Finds optimal classification threshold for each theme.
    """
    optimal_thresholds = {}
    for i, theme_name in enumerate(label_columns):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            predictions = (oof_predictions[:, i] > threshold).astype(int)
            f1 = f1_score(oof_labels[:, i], predictions, average='binary')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[theme_name] = best_threshold
    return optimal_thresholds
```

## 📊 Data Flow

### Training Pipeline
```
Raw Essays → Preprocessing → DAPT → Essay Reconstruction → 
Transformer Encoding → Attention Pooling → BiLSTM → 
Multi-Label Classification → Ensemble → Evaluation
```

### Inference Pipeline
```
New Essay → Essay Reconstruction → Transformer Encoding → 
Attention Pooling → BiLSTM → Multi-Label Classification → 
Threshold Application → Theme Predictions
```

## 🔧 Model Components

### Configuration Management
```python
class TrainingConfig:
    MODEL_NAME = 'microsoft/deberta-v3-large'
    N_SPLITS = 5
    N_EPOCHS = 20
    BATCH_SIZE = 32  # Base model
    BATCH_SIZE = 4   # Essay-aware model
    MAX_LENGTH = 512  # Base model
    MAX_LENGTH = 1024 # Essay-aware model
    EARLY_STOPPING_PATIENCE = 3
    LABEL_SMOOTHING_FACTOR = 0.1
    FOCAL_LOSS_GAMMA = 2.5
```

### Dataset Classes
```python
class ThemeDataset(Dataset):
    """Standard sentence-level dataset for base model."""
    
class EssayDataset(Dataset):
    """Essay-level dataset with character-to-token mapping."""
```

### Custom Collate Functions
```python
def collate_fn(batch: list) -> dict:
    """
    Handles variable-length essays and sentence counts.
    """
    max_num_sentences = max(x["num_sentences"] for x in batch)
    num_labels = batch[0]["sentence_labels"].shape[1]
    
    # Pad to maximum sentences in batch
    padded_labels = torch.full((len(batch), max_num_sentences, num_labels), -100.0)
    # ... padding logic
```

## 🚀 Training Pipeline

### Cross-Validation Strategy
```python
def train_with_cross_validation(train_df, tokenizer, label_columns):
    """
    Implements 5-fold iterative stratification for balanced splits.
    """
    kfold = IterativeStratification(n_splits=5, order=1)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df, train_df[label_columns])):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        # Train model for this fold
        model = train_fold(fold, train_fold_df, val_fold_df, tokenizer, label_columns)
```

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
    
    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
```

## 📈 Evaluation Framework

### Metrics
1. **Macro F1-Score**: Average F1 across all themes
2. **Micro F1-Score**: Overall F1 considering all predictions
3. **Hamming Loss**: Fraction of incorrectly predicted labels
4. **Per-Theme Performance**: Individual theme precision, recall, F1

### Ensemble Strategy
```python
def ensemble_predictions(fold_predictions, optimal_thresholds):
    """
    Combines predictions from all folds using optimal thresholds.
    """
    ensemble_preds = np.mean(fold_predictions, axis=0)
    
    final_predictions = np.zeros_like(ensemble_preds)
    for i, theme_name in enumerate(label_columns):
        threshold = optimal_thresholds[theme_name]
        final_predictions[:, i] = (ensemble_preds[:, i] > threshold).astype(int)
    
    return final_predictions
```

## 🔍 Performance Analysis

### Model Comparison
| Aspect | Base Model | Essay-Aware Model |
|--------|------------|-------------------|
| **Context Handling** | Sentence-level | Essay-level |
| **Memory Usage** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Macro F1** | 0.5135 | 0.5329 |
| **Micro F1** | 0.8050 | 0.8255 |

### Theme-Specific Performance
- **Strong Themes**: Aspirational (F1: 0.83), Filial Piety (F1: 0.74)
- **Challenging Themes**: Perseverance (F1: 0.00), Resistance (F1: 0.10)
- **Balanced Themes**: Attainment (F1: 0.59), Social (F1: 0.48)

## 🛠️ Implementation Notes

### Memory Optimization
- **Gradient Accumulation**: Effective batch size management
- **Mixed Precision**: BF16 training for A100 GPUs
- **Dynamic Padding**: Efficient handling of variable-length essays

### Reproducibility
- **Seed Setting**: Consistent random states across libraries
- **Deterministic Training**: CUDA deterministic mode
- **Checkpointing**: Model state preservation

### Scalability Considerations
- **Distributed Training**: Multi-GPU support via PyTorch DDP
- **Data Loading**: Optimized DataLoader configurations
- **Model Parallelism**: Support for large model architectures

---

This architecture documentation provides the technical foundation for understanding and extending the Aware framework. For implementation details, refer to the source code in the respective model directories. 