"""
Data Preparation Module for SpeakWise

This module handles:
1. Loading and preprocessing speech data from FluencyBank
2. Generating synthetic stuttering and disfluency data
3. Creating training/validation splits
4. Data augmentation techniques
"""

import json
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic speech repair training data
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Common stuttering patterns
        self.stutter_words = ['I', 'the', 'a', 'to', 'and', 'we', 'you', 'he', 'she', 'it']

        # Common grammar errors children make
        self.grammar_patterns = [
            ('went', 'goed'),
            ('ran', 'runned'),
            ('saw', 'seed'),
            ('were', 'was'),
            ('I am', 'I'),
            ('I have', 'I'),
        ]

    def add_stuttering(self, text: str, intensity: float = 0.3) -> str:
        """
        Add stuttering repetitions to text

        Args:
            text: Original clean text
            intensity: Probability of stuttering (0-1)

        Returns:
            Text with stuttering added
        """
        words = text.split()
        stuttered_words = []

        for word in words:
            # Add word repetition
            if random.random() < intensity and word.lower() in self.stutter_words:
                repetitions = random.randint(2, 4)
                stuttered_words.append('-'.join([word] * repetitions))
            # Add syllable repetition (first syllable)
            elif random.random() < intensity * 0.5 and len(word) > 3:
                first_part = word[:2]
                rest = word[2:]
                repetitions = random.randint(2, 3)
                stuttered_words.append('-'.join([first_part] * repetitions) + rest)
            else:
                stuttered_words.append(word)

        return ' '.join(stuttered_words)

    def add_word_repetitions(self, text: str, intensity: float = 0.2) -> str:
        """Add word-level repetitions (not stuttering)"""
        words = text.split()
        result = []

        for word in words:
            result.append(word)
            if random.random() < intensity:
                result.append(word)  # Repeat the word

        return ' '.join(result)

    def add_filler_words(self, text: str, intensity: float = 0.2) -> str:
        """Add filler words like 'um', 'uh', 'like'"""
        fillers = ['um', 'uh', 'like', 'you know', 'I mean']
        words = text.split()
        result = []

        for i, word in enumerate(words):
            if random.random() < intensity and i > 0:
                result.append(random.choice(fillers))
            result.append(word)

        return ' '.join(result)

    def add_grammar_errors(self, text: str, intensity: float = 0.3) -> str:
        """Add common children's grammar errors"""
        for correct, error in self.grammar_patterns:
            if random.random() < intensity:
                text = text.replace(correct, error)

        return text

    def omit_words(self, text: str, intensity: float = 0.15) -> str:
        """Randomly omit function words"""
        words = text.split()
        omit_candidates = ['is', 'are', 'am', 'the', 'a', 'an', 'to']

        result = []
        for word in words:
            if word.lower() in omit_candidates and random.random() < intensity:
                continue  # Skip this word
            result.append(word)

        return ' '.join(result)

    def generate_disfluent_version(
        self,
        clean_text: str,
        lambda_level: int = 2
    ) -> str:
        """
        Generate a disfluent version of clean text based on lambda level

        Higher lambda means more severe disfluencies

        Args:
            clean_text: Original grammatically correct text
            lambda_level: Disfluency level (0-4)

        Returns:
            Disfluent text
        """
        if lambda_level == 0:
            return clean_text

        # Intensity increases with lambda level
        intensity_map = {
            1: 0.1,   # Very minimal
            2: 0.25,  # Light
            3: 0.4,   # Moderate
            4: 0.6,   # Heavy
        }

        intensity = intensity_map.get(lambda_level, 0.25)

        disfluent_text = clean_text

        # Apply transformations based on lambda level
        if lambda_level >= 1:
            disfluent_text = self.add_stuttering(disfluent_text, intensity * 0.5)

        if lambda_level >= 2:
            disfluent_text = self.add_grammar_errors(disfluent_text, intensity)
            disfluent_text = self.omit_words(disfluent_text, intensity * 0.5)

        if lambda_level >= 3:
            disfluent_text = self.add_word_repetitions(disfluent_text, intensity * 0.7)

        if lambda_level >= 4:
            disfluent_text = self.add_filler_words(disfluent_text, intensity * 0.6)

        return disfluent_text

    def generate_training_pairs(
        self,
        clean_sentences: List[str],
        num_variations: int = 3
    ) -> List[Dict[str, any]]:
        """
        Generate training pairs from clean sentences

        Args:
            clean_sentences: List of grammatically correct sentences
            num_variations: Number of disfluent variations per sentence

        Returns:
            List of training examples with input, output, and lambda
        """
        training_data = []

        for sentence in clean_sentences:
            for _ in range(num_variations):
                # Random lambda level
                lambda_level = random.randint(1, 4)

                # Generate disfluent version
                disfluent = self.generate_disfluent_version(sentence, lambda_level)

                training_data.append({
                    'input': disfluent,
                    'output': sentence,
                    'lambda': lambda_level
                })

        return training_data


# ============================================================================
# FluencyBank Data Loader
# ============================================================================

class FluencyBankLoader:
    """
    Load and process data from FluencyBank corpus
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_chat_files(self, file_paths: List[str]) -> List[Dict[str, str]]:
        """
        Load CHAT format files from FluencyBank

        CHAT format includes transcribed speech with disfluency markers
        This is a simplified parser - actual CHAT files have complex formatting

        Returns:
            List of utterances with original and cleaned versions
        """
        utterances = []

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Parse CHAT format (simplified)
                    # Format: *CHI: utterance text [disfluency markers]
                    if line.startswith('*CHI:'):
                        original = self._extract_utterance(line)
                        cleaned = self._clean_utterance(original)

                        if original and cleaned and original != cleaned:
                            utterances.append({
                                'input': original,
                                'output': cleaned,
                                'source': 'fluencybank'
                            })

        return utterances

    def _extract_utterance(self, line: str) -> str:
        """Extract utterance text from CHAT line"""
        # Remove speaker tag
        text = line.split(':', 1)[1].strip()
        return text

    def _clean_utterance(self, text: str) -> str:
        """
        Remove disfluency markers and clean up text

        Common CHAT markers:
        - [/] : retracing
        - [//] : repetition
        - & : phonological fragment
        - ‹ › : overlap markers
        """
        # Remove CHAT markers (simplified)
        cleaned = re.sub(r'\[/+\]', '', text)  # Remove retracing markers
        cleaned = re.sub(r'&\w+', '', cleaned)  # Remove fragments
        cleaned = re.sub(r'[‹›]', '', cleaned)  # Remove overlap markers
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        return cleaned


# ============================================================================
# Dataset Builder
# ============================================================================

def build_dataset(
    output_dir: str = './data',
    num_synthetic_samples: int = 1000,
    train_ratio: float = 0.8
):
    """
    Build complete training and evaluation datasets

    Args:
        output_dir: Directory to save datasets
        num_synthetic_samples: Number of synthetic examples to generate
        train_ratio: Ratio of training vs evaluation data
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Example clean sentences for synthetic data generation
    # In practice, collect these from children's books, educational materials, etc.
    clean_sentences = [
        "I want to go to the park today",
        "My favorite color is blue",
        "Can I have some water please",
        "I like to play with my toys",
        "The dog is running in the yard",
        "I saw a big truck yesterday",
        "We went to the store with mom",
        "I am hungry and want lunch",
        "My teacher read us a story",
        "I have a new red backpack",
        "The cat is sleeping on the couch",
        "I can ride my bike really fast",
        "We are going to visit grandma",
        "I helped my dad wash the car",
        "The bird is singing in the tree",
        "I like to eat pizza and ice cream",
        "My friend has a cool video game",
        "I finished all my homework today",
        "The movie was really exciting",
        "I want to be a firefighter when I grow up",
        # Add more diverse sentences...
    ]

    # Generate synthetic data
    print("Generating synthetic training data...")
    generator = SyntheticDataGenerator(seed=42)
    training_pairs = generator.generate_training_pairs(
        clean_sentences,
        num_variations=num_synthetic_samples // len(clean_sentences)
    )

    print(f"Generated {len(training_pairs)} training pairs")

    # TODO: Load FluencyBank data if available
    # fluency_loader = FluencyBankLoader('./fluencybank_data')
    # fluency_data = fluency_loader.load_chat_files([...])
    # training_pairs.extend(fluency_data)

    # Shuffle and split
    random.shuffle(training_pairs)
    split_idx = int(len(training_pairs) * train_ratio)

    train_data = training_pairs[:split_idx]
    eval_data = training_pairs[split_idx:]

    # Save datasets
    train_path = output_path / 'train.json'
    eval_path = output_path / 'eval.json'

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved:")
    print(f"  Training: {train_path} ({len(train_data)} examples)")
    print(f"  Evaluation: {eval_path} ({len(eval_data)} examples)")

    # Print sample
    print("\nSample training examples:")
    for i, example in enumerate(train_data[:3]):
        print(f"\nExample {i+1} (λ={example['lambda']}):")
        print(f"  Input:  {example['input']}")
        print(f"  Output: {example['output']}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Build dataset
    build_dataset(
        output_dir='./data',
        num_synthetic_samples=1000,
        train_ratio=0.8
    )
