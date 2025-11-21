"""
Whisper Integration Module for SpeakWise

Integrates OpenAI Whisper for speech-to-text transcription
with the limited repair model for end-to-end speech processing
"""

import whisper
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import soundfile as sf
import librosa
from pathlib import Path


# ============================================================================
# Whisper ASR Module
# ============================================================================

class WhisperASR:
    """
    Wrapper for OpenAI Whisper automatic speech recognition
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: str = "en"
    ):
        """
        Initialize Whisper model

        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
                       - tiny: ~39M params, fastest
                       - base: ~74M params, good balance
                       - small: ~244M params, better accuracy
                       - medium: ~769M params, high accuracy
                       - large: ~1550M params, best accuracy
            device: Device to run on (cuda/cpu), auto-detected if None
            language: Language code (e.g., 'en' for English)
        """
        self.model_size = model_size
        self.language = language

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded successfully")

    def transcribe_audio(
        self,
        audio_path: str,
        return_timestamps: bool = False,
        return_confidence: bool = False
    ) -> Dict:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            return_timestamps: Include word-level timestamps
            return_confidence: Attempt to compute confidence scores

        Returns:
            Dictionary with transcription results
        """
        # Load audio using soundfile (doesn't require ffmpeg)
        try:
            audio_array, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        except Exception as e:
            # Fallback: try librosa (may work for some formats)
            try:
                audio_array, sr = librosa.load(audio_path, sr=16000)
            except:
                raise RuntimeError(f"Could not load audio file. Please ensure it's in a supported format (WAV, FLAC, OGG). Error: {e}")

        # Transcribe with Whisper using audio array
        result = self.model.transcribe(
            audio_array,
            language=self.language,
            fp16=(self.device == "cuda"),
            word_timestamps=return_timestamps,
            verbose=False
        )

        output = {
            'text': result['text'].strip(),
            'language': result['language'],
        }

        if return_timestamps and 'segments' in result:
            output['segments'] = result['segments']

        # Compute average log probability as confidence proxy
        if return_confidence:
            avg_logprob = np.mean([seg['avg_logprob'] for seg in result['segments']])
            # Convert log probability to rough confidence score (0-1)
            confidence = np.exp(avg_logprob)
            output['confidence'] = confidence

        return output

    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio from numpy array

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Transcribed text
        """
        # Whisper expects 16kHz audio
        if sample_rate != 16000:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=16000
            )

        # Pad/trim to 30 seconds (Whisper's input length)
        # Not necessary for transcribe() but useful for consistent processing
        result = self.model.transcribe(
            audio_array,
            language=self.language,
            fp16=(self.device == "cuda")
        )

        return result['text'].strip()


# ============================================================================
# End-to-End Speech Repair Pipeline
# ============================================================================

class SpeechRepairPipeline:
    """
    Complete pipeline: Speech -> Text -> Repair -> Output
    """

    def __init__(
        self,
        repair_model,
        repair_tokenizer,
        whisper_model_size: str = "base",
        device: Optional[str] = None
    ):
        """
        Initialize the complete pipeline

        Args:
            repair_model: Fine-tuned repair model
            repair_tokenizer: Tokenizer for repair model
            whisper_model_size: Whisper model size
            device: Device to use
        """
        # Initialize Whisper ASR
        self.asr = WhisperASR(
            model_size=whisper_model_size,
            device=device
        )

        # Store repair model
        self.repair_model = repair_model
        self.repair_tokenizer = repair_tokenizer

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def process_audio_file(
        self,
        audio_path: str,
        repair_intensity: float = 2.0,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Process audio file through complete pipeline

        Args:
            audio_path: Path to audio file
            repair_intensity: Lambda value for repair (0-4)
            return_intermediate: Return intermediate results

        Returns:
            Dictionary with results
        """
        # Step 1: Transcribe audio
        print(f"Transcribing audio: {audio_path}")
        transcription_result = self.asr.transcribe_audio(
            audio_path,
            return_confidence=True
        )

        transcribed_text = transcription_result['text']
        asr_confidence = transcription_result.get('confidence', 0.0)

        print(f"Transcription: {transcribed_text}")
        print(f"ASR Confidence: {asr_confidence:.3f}")

        # Step 2: Repair text
        print(f"Repairing text with λ={repair_intensity}")
        from train_limited_repair import repair_speech

        repaired_text = repair_speech(
            transcribed_text,
            self.repair_model,
            self.repair_tokenizer,
            repair_intensity=repair_intensity
        )

        print(f"Repaired: {repaired_text}")

        # Prepare output
        output = {
            'repaired_text': repaired_text,
            'audio_path': audio_path
        }

        if return_intermediate:
            output['transcribed_text'] = transcribed_text
            output['asr_confidence'] = asr_confidence
            output['repair_intensity'] = repair_intensity

        return output

    def process_live_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        repair_intensity: float = 2.0
    ) -> str:
        """
        Process live audio input (e.g., from microphone)

        Args:
            audio_array: Audio samples
            sample_rate: Sample rate
            repair_intensity: Lambda value

        Returns:
            Repaired text
        """
        # Transcribe
        transcribed = self.asr.transcribe_from_array(audio_array, sample_rate)

        # Repair
        from train_limited_repair import repair_speech
        repaired = repair_speech(
            transcribed,
            self.repair_model,
            self.repair_tokenizer,
            repair_intensity=repair_intensity
        )

        return repaired


# ============================================================================
# Audio Processing Utilities
# ============================================================================

class AudioProcessor:
    """
    Utility functions for audio preprocessing
    """

    @staticmethod
    def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Returns:
            (audio_array, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr

    @staticmethod
    def save_audio(audio_array: np.ndarray, sample_rate: int, output_path: str):
        """Save audio to file"""
        sf.write(output_path, audio_array, sample_rate)

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    @staticmethod
    def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simple noise reduction using spectral gating

        For more advanced noise reduction, consider using noisereduce library
        """
        # This is a placeholder - implement noise reduction if needed
        # pip install noisereduce
        # import noisereduce as nr
        # return nr.reduce_noise(audio, sr)
        return audio

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sr: int,
        top_db: int = 30
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio

        Args:
            audio: Audio array
            sr: Sample rate
            top_db: Threshold for silence detection (in dB)

        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of how to use the Whisper integration
    """
    # Initialize Whisper ASR only
    print("="*80)
    print("Example 1: Whisper ASR Only")
    print("="*80)

    asr = WhisperASR(model_size="base")

    # Example audio file (you need to provide actual audio)
    # audio_path = "./test_audio.wav"
    # result = asr.transcribe_audio(audio_path, return_confidence=True)
    # print(f"Transcription: {result['text']}")
    # print(f"Confidence: {result.get('confidence', 'N/A')}")

    print("\nWhisper ASR initialized successfully")
    print("To use: result = asr.transcribe_audio('your_audio.wav')")

    # Example: Full pipeline (requires trained model)
    print("\n" + "="*80)
    print("Example 2: Full Pipeline Setup")
    print("="*80)
    print("""
To use the full pipeline:

1. Train your repair model first:
   python train_limited_repair.py

2. Load the trained model:
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel

   base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
   model = PeftModel.from_pretrained(base_model, './output/limited_repair_model')
   tokenizer = AutoTokenizer.from_pretrained('./output/limited_repair_model')

3. Initialize pipeline:
   pipeline = SpeechRepairPipeline(model, tokenizer)

4. Process audio:
   result = pipeline.process_audio_file('audio.wav', repair_intensity=2.0)
   print(result['repaired_text'])
    """)


if __name__ == "__main__":
    example_usage()
