# SpeakWise: AI-Assisted Limited Repair Platform

An AI-powered platform for helping children with language disorders improve speech intelligibility while preserving their natural speaking style.

## Overview

SpeakWise uses advanced Large Language Models (LLMs) and speech recognition to perform "limited repair" on children's speech. The system can:

- Transcribe speech using OpenAI Whisper
- Apply controllable repair with adjustable intensity (λ parameter)
- Preserve the child's speaking style and vocabulary
- Provide real-time feedback through an interactive web interface

## Features

- **Demo Mode**: Start using immediately without training - uses OpenAI GPT API or rule-based repairs
- **Controllable Repair Intensity**: Adjust repair level from 0 (no correction) to 4 (maximum clarity)
- **Speech-to-Text Integration**: Built-in Whisper ASR for audio processing (independent of repair model)
- **LoRA Fine-tuning**: Efficient model training with Parameter-Efficient Fine-Tuning
- **Comprehensive Evaluation**: Multiple metrics including BERTScore, ROUGE, readability, and WER
- **Interactive UI**: User-friendly Streamlit interface with graceful degradation
- **Batch Processing**: Process multiple samples at once
- **Multiple Operation Modes**: Works with trained model, GPT API, or rule-based fallback

## Project Structure

```
SpeakWise/
├── setup_environment.bat          # Windows conda environment setup
├── requirements.txt               # Python dependencies
├── train_limited_repair.py        # Main training script with LoRA
├── data_preparation.py            # Data generation and preprocessing
├── evaluation.py                  # Evaluation metrics and analysis
├── whisper_integration.py         # Whisper ASR integration
├── app.py                         # Streamlit web application
├── data/                          # Training and evaluation data
│   ├── train.json
│   └── eval.json
└── output/                        # Trained models
    └── limited_repair_model/
```

## Installation

### Step 1: Set Up Conda Environment

**Windows:**
```bash
# Run the setup script
setup_environment.bat

# Or manually:
conda create -n speakwise python=3.10 -y
conda activate speakwise
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
conda create -n speakwise python=3.10 -y
conda activate speakwise

# Install PyTorch with CUDA (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers; import whisper; print('All packages installed successfully')"
```

## Quick Start

### Option 1: Demo Mode (Start Immediately)

You can start using SpeakWise right away without training:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` in **demo mode**:
- **Audio transcription**: Uses Whisper (if installed)
- **Text repair**: Uses rule-based repair by default
- **Enhanced demo**: Add OpenAI API key in sidebar for GPT-powered repairs

### Option 2: Full Mode (Train Custom Model)

For best results, train your own model:

#### 1. Generate Training Data

Create synthetic training data:

```bash
python data_preparation.py
```

This generates:
- `data/train.json`: Training dataset (800 examples)
- `data/eval.json`: Evaluation dataset (200 examples)

#### 2. Train the Model

Train the limited repair model with LoRA:

```bash
python train_limited_repair.py
```

**Training Configuration:**
```python
config = ModelConfig(
    model_name="microsoft/phi-2",  # Efficient, fully open model
    output_dir="./output/limited_repair_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    use_8bit=True  # 8-bit quantization for memory efficiency
)
```

**Expected Output:**
- Trained model saved to `./output/limited_repair_model/`
- Training logs in TensorBoard
- Example repairs at different λ levels

**Hardware Requirements:**
- GPU: NVIDIA with 6GB+ VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 10GB free space

#### 3. Run with Trained Model

Launch the app - it will automatically detect and use your trained model:

```bash
streamlit run app.py
```

## Usage Guide

### Getting Started

The app operates in two modes:
- **🎭 Demo Mode**: No trained model required - uses GPT API or rule-based repair
- **✅ Full Mode**: Uses your trained model for best results

### Text Input Mode

1. Open the **Text Input** tab
2. Enter text with disfluencies:
   ```
   I-I-I want to go to the p-p-park today
   ```
3. Adjust repair intensity (λ) in sidebar (0-4)
4. Click "Repair Text"
5. View original vs. repaired comparison with metrics

### Audio Input Mode

1. Open the **Audio Input** tab
2. Upload an audio file (WAV, MP3, M4A, FLAC, OGG)
3. Click "Process Audio"
4. View:
   - Original audio playback
   - Whisper transcription with confidence score
   - Repaired text

**Note:** Requires Whisper to be installed. The app will show a warning if unavailable.

### Batch Processing

1. Open the **Batch Processing** tab
2. Enter multiple samples (one per line)
3. Click "Process Batch"
4. View all results with expandable comparisons
5. Progress bar shows processing status

### Demo Mode Settings

To enhance demo mode quality:
1. Open sidebar
2. Find "Demo Mode Settings"
3. Enter your OpenAI API key
4. Demo repairs will now use GPT-3.5-turbo for better results

### Repair Intensity Levels (λ)

| λ | Description | Use Case |
|---|-------------|----------|
| 0 | No correction | Preserve original exactly |
| 1 | Minimal repair | Fix severe stuttering only |
| 2 | Light repair | Stuttering + basic grammar |
| 3 | Moderate repair | Improve clarity, maintain informality |
| 4 | Full repair | Maximum clarity and correctness |

## Data Preparation

### Using FluencyBank Data

1. Download data from [TalkBank](https://talkbank.org/fluency/)
2. Place CHAT files in `./fluencybank_data/`
3. Update `data_preparation.py`:

```python
fluency_loader = FluencyBankLoader('./fluencybank_data')
fluency_data = fluency_loader.load_chat_files([...])
```

### Custom Data Format

Create JSON files with this format:

```json
[
  {
    "input": "I-I-I want to go to the park",
    "output": "I want to go to the park",
    "lambda": 2.0
  },
  ...
]
```

### Synthetic Data Generation

The `SyntheticDataGenerator` class provides:
- Stuttering simulation
- Word repetition
- Grammar errors
- Word omissions
- Filler words

Example:
```python
from data_preparation import SyntheticDataGenerator

generator = SyntheticDataGenerator()
disfluent = generator.generate_disfluent_version(
    "I want to play with my friend",
    lambda_level=3
)
```

## Model Training Details

### Architecture

- **Base Model**: Microsoft Phi-2 (2.7B parameters)
  - Smaller, faster, and fully open-source
  - Lower VRAM requirements (6GB vs 12GB+)
  - Faster training and inference
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Only ~0.05% of parameters are trained

### Few-Shot Prompting

The model uses few-shot learning with examples:

```python
FEW_SHOT_EXAMPLES = [
    {
        "lambda": 1,
        "input": "I-I-I want to go to the p-p-park today",
        "output": "I want to go to the park today"
    },
    ...
]
```

### Training Hyperparameters

```python
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
weight_decay = 0.01
warmup_ratio = 0.03
```

### Memory Optimization

- 8-bit quantization (reduces VRAM by ~50%)
- Gradient checkpointing
- Mixed precision training (FP16)
- Phi-2's efficient architecture

**Minimum Requirements:**
- **For Demo Mode**: Any CPU (GPU optional for faster Whisper)
- **For Training**:
  - GPU: NVIDIA with 6GB+ VRAM (e.g., RTX 3060, GTX 1660 Ti)
  - RAM: 16GB
  - Storage: 10GB free space
- **For Inference Only**:
  - GPU: 4GB VRAM or CPU
  - RAM: 8GB

## Evaluation

### Metrics

1. **BERTScore**: Semantic similarity (0-1, higher is better)
2. **ROUGE**: N-gram overlap (0-1, higher is better)
3. **Readability**: Flesch Reading Ease and Grade Level
4. **Word Error Rate**: Edit distance (0-1, lower is better)
5. **Style Preservation**: Vocabulary overlap and length ratio

### Running Evaluation

```bash
python evaluation.py
```

Or programmatically:

```python
from evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
metrics = evaluator.evaluate_all(
    originals=["I-I-I want pizza"],
    predictions=["I want pizza"],
    references=["I want pizza"]
)
```

### Lambda-Specific Evaluation

```python
from evaluation import LambdaEvaluator

lambda_eval = LambdaEvaluator()
results = lambda_eval.evaluate_by_lambda(test_data)
lambda_eval.plot_lambda_comparison(results, save_path='./comparison.png')
```

## Advanced Usage

### Custom Model Configuration

```python
from train_limited_repair import ModelConfig, train_model

config = ModelConfig(
    model_name="microsoft/phi-2",  # Current default (recommended)
    # model_name="google/gemma-7b",  # Alternative: larger but needs more VRAM
    # model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Alternative: smaller/faster
    lora_r=16,                     # Increase LoRA rank
    num_train_epochs=5,            # More epochs
    learning_rate=1e-4,            # Lower learning rate
)

model, tokenizer = train_model(config)
```

### Inference with Custom Intensity

```python
from train_limited_repair import repair_speech

# Load your trained model
model, tokenizer = ...

# Repair with custom settings
result = repair_speech(
    input_text="I-I-I want to play",
    model=model,
    tokenizer=tokenizer,
    repair_intensity=2.5,  # Custom λ
    temperature=0.7,
    top_p=0.9
)
```

### Full Pipeline (Speech → Repair)

```python
from whisper_integration import SpeechRepairPipeline

# Initialize pipeline
pipeline = SpeechRepairPipeline(
    repair_model=model,
    repair_tokenizer=tokenizer,
    whisper_model_size="base"
)

# Process audio file
result = pipeline.process_audio_file(
    audio_path="./test.wav",
    repair_intensity=2.0,
    return_intermediate=True
)

print(f"Transcribed: {result['transcribed_text']}")
print(f"Repaired: {result['repaired_text']}")
print(f"ASR Confidence: {result['asr_confidence']}")
```

## Using LLaMA-Factory (Alternative Training Method)

If you prefer using LLaMA-Factory directly:

1. **Install LLaMA-Factory:**
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

2. **Prepare data in LLaMA-Factory format:**
```json
[
  {
    "instruction": "Perform limited repair with λ=2",
    "input": "I-I-I want to go park",
    "output": "I want to go to the park"
  }
]
```

3. **Train using LLaMA-Factory CLI:**
```bash
llamafactory-cli train \
    --model_name_or_path meta-llama/Llama-3-8B \
    --do_train \
    --dataset your_data \
    --finetuning_type lora \
    --output_dir ./output/llama_factory_model \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
per_device_train_batch_size = 2

# Enable 8-bit quantization (should already be enabled)
use_8bit = True

# Phi-2 is already efficient - if still issues, try:
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Even smaller
```

**2. Whisper Model Not Loading**
```bash
# Install ffmpeg (required for audio processing)
# Windows: Download from https://ffmpeg.org/
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
```

**3. Import Errors**
```bash
# Make sure you're in the correct directory
cd SpeakWise

# Activate conda environment
conda activate speakwise

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

**4. Model Download Issues**
```bash
# If HuggingFace downloads fail, try:
export HF_ENDPOINT=https://hf-mirror.com

# Or use offline mode with cached models
transformers-cli env
```

**5. App Starts in Demo Mode**
- This is normal if you haven't trained a model yet
- The app will automatically detect and use trained models
- Add OpenAI API key in sidebar for better demo mode results
- Or train your own model: `python train_limited_repair.py`

## Research & Citations

This project is based on research in:
- Controllable text generation
- Speech language pathology
- Assistive AI technology

### Related Papers
- "FUDGE: Controlled Text Generation With Future Discriminators" (Yang & Klein, 2021)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

## Safety & Ethics

**Important Considerations:**

1. **Not a Medical Device**: This is a research tool, not a replacement for professional speech therapy
2. **Data Privacy**: All audio and text data should be anonymized
3. **Parental Consent**: Obtain proper consent when working with children's data
4. **Bias Awareness**: Model may reflect biases in training data
5. **Supervision Required**: Always use under adult/therapist supervision

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional language support
- [ ] Real-time audio streaming
- [ ] Mobile app integration
- [ ] More sophisticated noise reduction
- [ ] Multi-speaker diarization
- [ ] Child-specific voice synthesis

## License

This project is for educational and research purposes. Check individual model licenses:
- Llama 3: [Meta License](https://ai.meta.com/llama/)
- Whisper: MIT License
- Gemma: [Google License](https://ai.google.dev/gemma/terms)

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Check GPU/CUDA compatibility
4. Verify all dependencies are installed

## Acknowledgments

- OpenAI for Whisper and GPT API (demo mode)
- Microsoft for Phi-2
- HuggingFace for Transformers and PEFT libraries
- TalkBank for FluencyBank dataset
- Research advisors and mentors

## Future Work

- [ ] Integration with additional ASR systems
- [ ] Child-specific voice cloning for TTS
- [ ] Gamification for therapy engagement
- [ ] Long-term progress tracking
- [ ] Multi-modal feedback (visual + audio)
- [ ] Mobile/tablet deployment

---

**Built with care for children with language disorders** 🎤❤️
