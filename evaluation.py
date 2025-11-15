"""
Evaluation Module for SpeakWise

Comprehensive evaluation metrics for assessing repair quality:
1. BERTScore - Semantic similarity
2. ROUGE - N-gram overlap
3. Readability - Flesch Reading Ease, Flesch-Kincaid Grade
4. Word Error Rate (WER) - For ASR evaluation
5. Style Preservation - Measure how much the style is preserved
"""

import numpy as np
from typing import List, Dict, Tuple
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import textstat
from jiwer import wer, cer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


# ============================================================================
# Comprehensive Evaluator
# ============================================================================

class ComprehensiveEvaluator:
    """
    Evaluate repair quality across multiple dimensions
    """

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    # ========================================================================
    # Semantic Similarity Metrics
    # ========================================================================

    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore between predictions and references

        BERTScore measures semantic similarity using BERT embeddings
        High scores indicate good semantic preservation
        """
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }

    # ========================================================================
    # N-gram Overlap Metrics
    # ========================================================================

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores

        ROUGE measures n-gram overlap between predictions and references
        """
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

    # ========================================================================
    # Readability Metrics
    # ========================================================================

    def compute_readability(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute readability scores

        - Flesch Reading Ease: Higher = easier to read (0-100)
        - Flesch-Kincaid Grade: U.S. grade level needed to understand
        """
        flesch_scores = []
        grade_levels = []

        for text in texts:
            try:
                flesch_scores.append(textstat.flesch_reading_ease(text))
                grade_levels.append(textstat.flesch_kincaid_grade(text))
            except:
                # Skip texts that are too short or invalid
                continue

        return {
            'flesch_reading_ease': np.mean(flesch_scores) if flesch_scores else 0.0,
            'flesch_kincaid_grade': np.mean(grade_levels) if grade_levels else 0.0
        }

    # ========================================================================
    # Word/Character Error Rate
    # ========================================================================

    def compute_error_rates(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute Word Error Rate (WER) and Character Error Rate (CER)

        Lower is better (0 = perfect match)
        Useful for ASR evaluation and measuring edit distance
        """
        # Concatenate all texts for overall metrics
        pred_concat = ' '.join(predictions)
        ref_concat = ' '.join(references)

        wer_score = wer(ref_concat, pred_concat)
        cer_score = cer(ref_concat, pred_concat)

        return {
            'word_error_rate': wer_score,
            'character_error_rate': cer_score
        }

    # ========================================================================
    # Style Preservation Metrics
    # ========================================================================

    def compute_length_preservation(
        self,
        predictions: List[str],
        originals: List[str]
    ) -> Dict[str, float]:
        """
        Measure length preservation

        Good repair should not dramatically change sentence length
        """
        pred_lengths = [len(p.split()) for p in predictions]
        orig_lengths = [len(o.split()) for o in originals]

        length_ratios = [p/o if o > 0 else 1.0 for p, o in zip(pred_lengths, orig_lengths)]

        return {
            'avg_length_ratio': np.mean(length_ratios),
            'length_ratio_std': np.std(length_ratios)
        }

    def compute_vocabulary_overlap(
        self,
        predictions: List[str],
        originals: List[str]
    ) -> Dict[str, float]:
        """
        Measure vocabulary overlap between original and repaired text

        Higher overlap = better style preservation
        """
        overlaps = []

        for pred, orig in zip(predictions, originals):
            pred_words = set(pred.lower().split())
            orig_words = set(orig.lower().split())

            if len(orig_words) == 0:
                overlaps.append(1.0)
            else:
                overlap = len(pred_words & orig_words) / len(orig_words)
                overlaps.append(overlap)

        return {
            'vocabulary_overlap': np.mean(overlaps),
            'vocabulary_overlap_std': np.std(overlaps)
        }

    # ========================================================================
    # Comprehensive Evaluation
    # ========================================================================

    def evaluate_all(
        self,
        originals: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Run all evaluation metrics

        Args:
            originals: Original disfluent text
            predictions: Model-repaired text
            references: Gold standard corrections

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        print("Computing BERTScore...")
        metrics.update(self.compute_bertscore(predictions, references))

        print("Computing ROUGE...")
        metrics.update(self.compute_rouge(predictions, references))

        print("Computing readability...")
        pred_readability = self.compute_readability(predictions)
        metrics.update({f'pred_{k}': v for k, v in pred_readability.items()})

        ref_readability = self.compute_readability(references)
        metrics.update({f'ref_{k}': v for k, v in ref_readability.items()})

        print("Computing error rates...")
        metrics.update(self.compute_error_rates(predictions, references))

        print("Computing style preservation...")
        metrics.update(self.compute_length_preservation(predictions, originals))
        metrics.update(self.compute_vocabulary_overlap(predictions, originals))

        return metrics


# ============================================================================
# Lambda-Specific Evaluation
# ============================================================================

class LambdaEvaluator:
    """
    Evaluate repair quality at different lambda (repair intensity) levels
    """

    def __init__(self):
        self.evaluator = ComprehensiveEvaluator()

    def evaluate_by_lambda(
        self,
        test_data: List[Dict]
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate metrics grouped by lambda level

        Args:
            test_data: List of dicts with keys: 'original', 'prediction', 'reference', 'lambda'

        Returns:
            Dictionary mapping lambda level to metrics
        """
        # Group by lambda
        lambda_groups = {}
        for item in test_data:
            lambda_val = item['lambda']
            if lambda_val not in lambda_groups:
                lambda_groups[lambda_val] = {
                    'originals': [],
                    'predictions': [],
                    'references': []
                }

            lambda_groups[lambda_val]['originals'].append(item['original'])
            lambda_groups[lambda_val]['predictions'].append(item['prediction'])
            lambda_groups[lambda_val]['references'].append(item['reference'])

        # Evaluate each group
        results = {}
        for lambda_val, data in lambda_groups.items():
            print(f"\nEvaluating λ={lambda_val}...")
            metrics = self.evaluator.evaluate_all(
                data['originals'],
                data['predictions'],
                data['references']
            )
            results[lambda_val] = metrics

        return results

    def plot_lambda_comparison(
        self,
        lambda_metrics: Dict[int, Dict[str, float]],
        save_path: str = './lambda_comparison.png'
    ):
        """
        Create visualizations comparing metrics across lambda levels
        """
        # Select key metrics to plot
        key_metrics = [
            'bertscore_f1',
            'rouge1',
            'word_error_rate',
            'vocabulary_overlap',
            'pred_flesch_reading_ease'
        ]

        lambda_levels = sorted(lambda_metrics.keys())
        num_metrics = len(key_metrics)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(key_metrics):
            values = [lambda_metrics[l].get(metric, 0) for l in lambda_levels]

            axes[i].plot(lambda_levels, values, marker='o', linewidth=2, markersize=8)
            axes[i].set_xlabel('Repair Intensity (λ)', fontsize=12)
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[i].set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(lambda_levels)

        # Hide extra subplot
        axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")


# ============================================================================
# ASR Confidence Evaluator
# ============================================================================

class ASRConfidenceEvaluator:
    """
    Evaluate ASR confidence before and after repair

    This helps validate that repaired text is clearer for ASR systems
    """

    def __init__(self, whisper_model=None):
        """
        Args:
            whisper_model: Loaded Whisper model (optional)
        """
        self.whisper_model = whisper_model

    def compute_asr_confidence(self, audio_path: str) -> float:
        """
        Compute ASR confidence on audio file

        Requires Whisper model to be loaded
        """
        if self.whisper_model is None:
            raise ValueError("Whisper model not loaded")

        # Transcribe with Whisper
        result = self.whisper_model.transcribe(
            audio_path,
            fp16=False,
            word_timestamps=True
        )

        # Get average word-level confidence if available
        # Note: Whisper doesn't directly provide confidence scores
        # You may need to use log probabilities or other methods

        return result

    def compare_repair_impact(
        self,
        original_texts: List[str],
        repaired_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Compare the impact of repair on text quality

        This simulates ASR improvement by measuring text clarity
        """
        # Use WER as proxy for ASR performance
        evaluator = ComprehensiveEvaluator()

        original_metrics = evaluator.compute_error_rates(original_texts, reference_texts)
        repaired_metrics = evaluator.compute_error_rates(repaired_texts, reference_texts)

        improvement = {
            'wer_improvement': original_metrics['word_error_rate'] - repaired_metrics['word_error_rate'],
            'cer_improvement': original_metrics['character_error_rate'] - repaired_metrics['character_error_rate'],
            'original_wer': original_metrics['word_error_rate'],
            'repaired_wer': repaired_metrics['word_error_rate']
        }

        return improvement


# ============================================================================
# Example Usage
# ============================================================================

def example_evaluation():
    """
    Example of how to use the evaluation modules
    """
    # Sample data
    originals = [
        "I-I-I want to go to the p-p-park",
        "Me and my friend we goed to store",
        "The dog was so so big and and it runned fast"
    ]

    predictions = [
        "I want to go to the park",
        "Me and my friend went to the store",
        "The dog was so big and it ran fast"
    ]

    references = [
        "I want to go to the park",
        "My friend and I went to the store",
        "The dog was very big and it ran fast"
    ]

    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate_all(originals, predictions, references)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name:30s}: {value:.4f}")

    # Lambda-specific evaluation example
    test_data = [
        {
            'original': originals[0],
            'prediction': predictions[0],
            'reference': references[0],
            'lambda': 2
        },
        {
            'original': originals[1],
            'prediction': predictions[1],
            'reference': references[1],
            'lambda': 3
        },
        {
            'original': originals[2],
            'prediction': predictions[2],
            'reference': references[2],
            'lambda': 3
        },
    ]

    lambda_eval = LambdaEvaluator()
    lambda_metrics = lambda_eval.evaluate_by_lambda(test_data)

    print("\n" + "="*80)
    print("LAMBDA-SPECIFIC RESULTS")
    print("="*80)

    for lambda_val, metrics in lambda_metrics.items():
        print(f"\nλ={lambda_val}:")
        for metric_name, value in sorted(metrics.items()):
            print(f"  {metric_name:28s}: {value:.4f}")


if __name__ == "__main__":
    example_evaluation()
