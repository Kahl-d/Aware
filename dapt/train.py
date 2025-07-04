# train.py (Updated DAPT Script)
import logging
import math
import os
import sys
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Set up logging for a clear and informative training process
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    # Note: The manual torch.distributed initialization block has been removed.
    # The `accelerate` library handles all device placement and initialization
    # automatically when using `accelerate launch`. Manually setting it up
    # was causing the "Duplicate GPU" error.

    # --- 1. Hardcoded Configuration ---
    # All parameters are set here directly in the script for non-interactive HPC jobs.

    # --- Paths and Model Config ---
    # IMPORTANT: You must change this to the actual, full path of your dataset on the cluster's filesystem.
    DATA_FILE = "combined_essays.csv"
    MODEL_NAME_OR_PATH = "microsoft/deberta-v3-large"
    OUTPUT_DIR = "./deberta-v3-large-essays-adapted-final" # Final model will be saved here.

    # --- Dataset Config ---
    VALIDATION_SPLIT_PERCENTAGE = 5 # Using 5% of ~3800 essays (~190 essays) for validation.

    logger.info("DAPT Script starting with final hardcoded configurations.")
    logger.info(f"Data file: {DATA_FILE}")
    logger.info(f"Base model: {MODEL_NAME_OR_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")


    # --- 2. Load and Prepare the Dataset ---
    logger.info(f"Loading data from {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        logger.error(f"FATAL: Data file not found at {DATA_FILE}. Please make sure this file exists in the same directory as the script, or provide a full path.")
        sys.exit(1)
    raw_datasets = load_dataset('csv', data_files={'train': DATA_FILE})

    logger.info("Creating train/validation split...")
    # The `train_test_split` function expects a float between 0.0 and 1.0.
    split_datasets = raw_datasets["train"].train_test_split(
        test_size=(VALIDATION_SPLIT_PERCENTAGE / 100), seed=42
    )
    split_datasets["validation"] = split_datasets.pop("test")
    logger.info(f"Dataset splits created: {split_datasets}")


    # --- 3. Load Model and Tokenizer ---
    logger.info(f"Loading model and tokenizer for '{MODEL_NAME_OR_PATH}'...")
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = DebertaV2ForMaskedLM.from_pretrained(MODEL_NAME_OR_PATH, config=config)
    
    # IMPORTANT: Resize token embeddings to potentially include new tokens from your domain
    # If the tokenizer's vocabulary length changes due to new tokens in your combined_essays.csv,
    # this line will update the model's embedding layer accordingly.
    # New embeddings for new tokens will be randomly initialized.
    model.resize_token_embeddings(len(tokenizer)) 
    logger.info(f"Model embeddings resized to match tokenizer length: {len(tokenizer)}")


    # --- 4. Preprocess the Data (Ensuring Document Integrity) ---
    column_names = raw_datasets["train"].column_names
    text_column_name = "essay" if "essay" in column_names else column_names[1]

    def tokenize_function(examples):
        # Using tokenizer.encode_plus allows for more explicit control,
        # but tokenizer() call (as-is) is fine for most cases.
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    logger.info("Tokenizing datasets...")
    tokenized_datasets = split_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=column_names,
    )

    # --- 5. Configure the Data Collator (for Whole Word Masking) ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # ===================================================================================
    # --- 6. Set Up Training Arguments for a Short, High-Quality Run ---
    # These settings are specifically optimized for a smaller dataset (~3-4k essays)
    # and use the latest, correct argument names to avoid errors.
    # ===================================================================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        # Data-centric strategy
        num_train_epochs=5, # Run for 5 full epochs to find the sweet spot.
        # Performance optimizations for A100s
        per_device_train_batch_size=4,  # Smaller batch size for more gradient updates on a small dataset.
        gradient_accumulation_steps=4,  # Effective batch size = 4(batch)*4(accum)*num_gpus = 64 with 4 GPUs
        bf16=torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_bf16_supported(),
        optim="adamw_torch_fused",
        # Logging and saving strategy to get the BEST model from the run
        eval_strategy="epoch",          # CORRECTED: Use 'eval_strategy' instead of 'evaluation_strategy'.
        logging_strategy="epoch",       # Log metrics at the end of each epoch.
        save_strategy="epoch",          # Save a checkpoint at the end of each epoch.
        save_total_limit=1,             # Only keep the single best checkpoint, saving disk space.
        load_best_model_at_end=True,    # CRITICAL: This ensures the final model is the best one.
        metric_for_best_model="eval_loss",# Use validation loss to determine the best model.
        greater_is_better=False,        # For loss/perplexity, lower is better.
        report_to="tensorboard",
    )

    # Sanity check for A100-specific optimizations
    if training_args.bf16:
        logger.info("bfloat16 mixed-precision training is enabled for A100s.")


    # --- 7. Initialize and Run the Trainer ---
    # CRITICAL FIX: Pass the tokenizer to the Trainer constructor.
    # This ensures that trainer.save_model() will save the tokenizer
    # alongside the model weights and config, creating a complete package.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer, # <--- THIS IS THE KEY ADDITION!
    )

    logger.info("Starting Domain-Adaptive Pre-training...")
    # This run is short, so we start fresh. If it fails, `resume_from_checkpoint=True` can be used.
    train_result = trainer.train()

    # --- 8. Save Final Model, Metrics, and State ---
    # Because `load_best_model_at_end=True`, the trainer has automatically loaded the
    # best model's weights before this step. We are saving the optimal model.
    # With `tokenizer=tokenizer` passed to Trainer, this will now also save
    # tokenizer.json, vocab.json, etc., making the output directory self-contained.
    logger.info("DAPT complete. Saving the best model found during training...")
    trainer.save_model()
    trainer.save_state()

    # Log and save final metrics for record-keeping
    metrics = train_result.metrics
    try:
        logger.info("Evaluating the final best model on the validation set...")
        eval_metrics = trainer.evaluate()
        final_perplexity = math.exp(eval_metrics["eval_loss"])
        metrics["final_eval_perplexity"] = final_perplexity
        logger.info(f"Final Perplexity of the best model: {final_perplexity:.2f}")
    except Exception as e:
        logger.error(f"Could not calculate final perplexity: {e}")

    try:
        with open(os.path.join(OUTPUT_DIR, "final_training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logger.error(f"Could not save training metrics to JSON: {e}")

    logger.info(f"The best domain-adapted model has been saved to {OUTPUT_DIR}")
    logger.info("This model is now ready for Phase 2: fine-tuning on a specific classification task.")


if __name__ == "__main__":
    main()