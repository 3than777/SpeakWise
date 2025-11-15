"""
SpeakWise: AI-Assisted Limited Repair Platform
Fine-tuning LLM for controllable speech repair using LLaMA-Factory

This template implements the training pipeline for a limited repair model
that helps children with language disorders improve speech intelligibility
while preserving their speaking style.
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import numpy as np
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import textstat


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model and training"""
    # Model settings
    model_name: str = "microsoft/phi-2"  # Using Phi-2 (smaller, faster, fully open)
    model_max_length: int = 512

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Training settings
    output_dir: str = "./output/limited_repair_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Repair intensity parameter (λ)
    repair_intensity_min: float = 0.0  # No repair
    repair_intensity_max: float = 4.0  # Maximum repair

    # Data settings
    train_data_path: str = "./data/train.json"
    eval_data_path: str = "./data/eval.json"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100

    # Device settings
    use_8bit: bool = True  # Use 8-bit quantization for memory efficiency


# ============================================================================
# System Prompt Engineering
# ============================================================================

SYSTEM_PROMPT = """You are an expert AI assistant specialized in helping children with language disorders improve their speech clarity. Your role is to perform "limited repair" on their speech or text input.

Guidelines:
1. Preserve the child's original speaking style and voice as much as possible
2. Correct only what is necessary for clarity based on the repair intensity level (λ)
3. Do not over-correct or make the speech too formal
4. Maintain the child's vocabulary level and sentence structure when appropriate
5. Focus on fixing: stuttering repetitions, word omissions, grammatical errors, and unclear expressions

Repair Intensity Levels (λ):
- λ=0: No correction, return input as-is
- λ=1: Minimal repair (fix only severe stuttering and major omissions)
- λ=2: Light repair (fix stuttering and basic grammar)
- λ=3: Moderate repair (improve clarity while keeping informal tone)
- λ=4: Full repair (maximize clarity and grammatical correctness)

Always maintain empathy and respect for the child's communication style."""


# Few-shot examples for prompt design
FEW_SHOT_EXAMPLES = [
    {
        "lambda": 1,
        "input": "I-I-I want to go to the p-p-park today",
        "output": "I want to go to the park today"
    },
    {
        "lambda": 2,
        "input": "Me and my friend we goed to store yesterday",
        "output": "Me and my friend went to the store yesterday"
    },
    {
        "lambda": 3,
        "input": "The dog was so so big and and it runned really really fast",
        "output": "The dog was so big and it ran really fast"
    },
    {
        "lambda": 4,
        "input": "I seed a big truck it was red and loud the driver he waved",
        "output": "I saw a big truck. It was red and loud. The driver waved at me."
    },
    {
        "lambda": 0,
        "input": "I like ice cream lots",
        "output": "I like ice cream lots"
    },
    {
        "lambda": 2,
        "input": "Can I have some some water please I thirsty",
        "output": "Can I have some water please? I'm thirsty."
    },
]


# ============================================================================
# Data Preparation
# ============================================================================

def create_prompt(input_text: str, repair_intensity: float, include_output: bool = True, output_text: str = "") -> str:
    """
    Create a formatted prompt for the model

    Args:
        input_text: Original speech/text from child
        repair_intensity: Lambda value (0-4) controlling repair degree
        include_output: Whether to include the expected output (for training)
        output_text: The corrected text (for training only)

    Returns:
        Formatted prompt string
    """
    # Few-shot context
    few_shot_context = "\n\n".join([
        f"Example {i+1} (λ={ex['lambda']}):\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES[:3])  # Use first 3 examples
    ])

    prompt = f"""{SYSTEM_PROMPT}

{few_shot_context}

Now, perform limited repair on the following input:

Repair Intensity (λ): {repair_intensity}
Input: {input_text}
Output:"""

    if include_output:
        prompt += f" {output_text}"

    return prompt


def prepare_dataset(data_path: str, tokenizer, config: ModelConfig) -> Dataset:
    """
    Prepare dataset for training

    Expected data format (JSON):
    [
        {
            "input": "original text with errors",
            "output": "corrected text",
            "lambda": 2.0
        },
        ...
    ]
    """
    # Load raw data
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Prepare training examples
    formatted_data = []
    for item in raw_data:
        input_text = item['input']
        output_text = item['output']
        lambda_val = item.get('lambda', 2.0)  # Default to moderate repair

        # Create prompt with output for training
        full_prompt = create_prompt(input_text, lambda_val, include_output=True, output_text=output_text)

        formatted_data.append({
            'text': full_prompt,
            'input': input_text,
            'output': output_text,
            'lambda': lambda_val
        })

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    def tokenize_function(examples):
        # Tokenize the full prompt
        model_inputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.model_max_length,
            padding='max_length',
        )

        # Create labels (same as input_ids for causal LM)
        model_inputs['labels'] = model_inputs['input_ids'].copy()

        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    return tokenized_dataset


# ============================================================================
# Model Setup
# ============================================================================

def setup_model(config: ModelConfig):
    """
    Load and configure model with LoRA for efficient fine-tuning
    """
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=False
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        load_in_8bit=config.use_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    if config.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def train_model(config: ModelConfig):
    """
    Main training function
    """
    # Setup model and tokenizer
    model, tokenizer = setup_model(config)

    # Prepare datasets
    print("Preparing training dataset...")
    train_dataset = prepare_dataset(config.train_data_path, tokenizer, config)

    print("Preparing evaluation dataset...")
    eval_dataset = prepare_dataset(config.eval_data_path, tokenizer, config)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        report_to="tensorboard",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


# ============================================================================
# Inference with Controllable Repair
# ============================================================================

def repair_speech(
    input_text: str,
    model,
    tokenizer,
    repair_intensity: float = 2.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Perform limited repair on input text with controllable intensity

    Args:
        input_text: Original speech/text
        model: Fine-tuned model
        tokenizer: Tokenizer
        repair_intensity: Lambda value (0-4)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Repaired text
    """
    # Clamp repair intensity to valid range
    repair_intensity = max(0.0, min(4.0, repair_intensity))

    # Create prompt without output
    prompt = create_prompt(input_text, repair_intensity, include_output=False)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (after "Output:")
    if "Output:" in full_output:
        repaired_text = full_output.split("Output:")[-1].strip()
    else:
        repaired_text = full_output

    return repaired_text


# ============================================================================
# Evaluation Metrics
# ============================================================================

class RepairEvaluator:
    """
    Evaluate repair quality using multiple metrics
    """
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore between predictions and references"""
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for pred, ref in zip(predictions, references):
            rouge_scores = self.rouge_scorer.score(ref, pred)
            scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            scores['rougeL'].append(rouge_scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }

    def compute_readability(self, texts: List[str]) -> Dict[str, float]:
        """Compute readability scores"""
        flesch_scores = [textstat.flesch_reading_ease(text) for text in texts]
        grade_levels = [textstat.flesch_kincaid_grade(text) for text in texts]

        return {
            'flesch_reading_ease': np.mean(flesch_scores),
            'flesch_kincaid_grade': np.mean(grade_levels)
        }

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Run all evaluation metrics"""
        metrics = {}

        # BERTScore (semantic similarity)
        metrics.update(self.compute_bertscore(predictions, references))

        # ROUGE (n-gram overlap)
        metrics.update(self.compute_rouge(predictions, references))

        # Readability
        pred_readability = self.compute_readability(predictions)
        metrics.update({f'pred_{k}': v for k, v in pred_readability.items()})

        ref_readability = self.compute_readability(references)
        metrics.update({f'ref_{k}': v for k, v in ref_readability.items()})

        return metrics


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """
    Example usage of the training pipeline
    """
    # Configuration
    config = ModelConfig(
        model_name="microsoft/phi-2",  # Using Phi-2 (smaller, faster, fully open)
        output_dir="./output/limited_repair_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Train model
    print("="*80)
    print("SpeakWise: Training Limited Repair Model")
    print("="*80)

    model, tokenizer = train_model(config)

    # Example inference
    print("\n" + "="*80)
    print("Testing Model with Different Repair Intensities")
    print("="*80)

    test_input = "I-I-I want to go to the p-p-park and and play with my my friend"

    for lambda_val in [0, 1, 2, 3, 4]:
        output = repair_speech(test_input, model, tokenizer, repair_intensity=lambda_val)
        print(f"\nλ={lambda_val}: {output}")

    # Evaluation example
    print("\n" + "="*80)
    print("Evaluation Metrics")
    print("="*80)

    evaluator = RepairEvaluator()

    # Example predictions and references
    predictions = [
        "I want to go to the park and play with my friend",
        "The dog ran very fast",
    ]
    references = [
        "I want to go to the park and play with my friend",
        "The dog was running really fast",
    ]

    metrics = evaluator.evaluate(predictions, references)

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
