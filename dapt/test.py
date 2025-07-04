import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import logging
import sys
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Model Paths ---
ORIGINAL_MODEL_PATH = "microsoft/deberta-v3-large"
ADAPTED_MODEL_PATH = "./deberta-v3-large-essays-adapted-final"

# --- Test Sentences ---
test_texts = [
    "As a first-generation student, I had to learn how to navigate the university's [MASK] all by myself.",
    "My parents always taught me that the most important thing in life was my [MASK].",
    "Despite the financial hardship, my family's emotional [MASK] was a constant source of strength.",
    "I hope that by graduating I can be a role [MASK] for my younger siblings.",
    "The professor said my essay on cultural [MASK] was insightful and well-argued."
]

def get_predictions(model_path, tokenizer_path, model_label):
    """
    Loads a model and tokenizer from specified paths and returns predictions.
    This allows us to "mix and match" components.
    """
    try:
        logger.info(f"--- Predictions from ({model_label}):")
        
        # Load the model and tokenizer. The from_pretrained function will handle
        # both local paths and Hugging Face model IDs automatically.
        # The faulty os.path.exists check has been removed.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        
        # Create the pipeline with the loaded components
        fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        predictions = fill_masker(text_with_mask, top_k=5)
        
        for pred in predictions:
            logger.info(f"    - Predicted token: {pred['token_str']:<15} | Confidence: {pred['score']:.4f}")

    except Exception as e:
        logger.error(f"Could not get predictions for model '{model_label}'. Error: {e}")


def main():
    """
    Main function to loop through all test texts and compare model outputs.
    """
    for text in test_texts:
        global text_with_mask
        text_with_mask = text

        logger.info("\n" + "="*80)
        logger.info(f"Input Text: '{text}'")
        logger.info("="*80)
        
        # --- 1. Original Model ---
        get_predictions(
            model_path=ORIGINAL_MODEL_PATH, 
            tokenizer_path=ORIGINAL_MODEL_PATH, 
            model_label="Original Model"
        )
        
        # --- 2. Adapted Model ---
        get_predictions(
            model_path=ADAPTED_MODEL_PATH, 
            tokenizer_path=ORIGINAL_MODEL_PATH,
            model_label="Your Adapted Model"
        )

    logger.info("\n" + "="*80)
    logger.info("Comparison complete.")


if __name__ == "__main__":
    main()