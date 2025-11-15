"""
SpeakWise Streamlit Application

Interactive web interface for the AI-assisted limited repair platform
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path
import tempfile
import re
import difflib

# Import local modules
from whisper_integration import WhisperASR, SpeechRepairPipeline, AudioProcessor
from train_limited_repair import repair_speech
from evaluation import ComprehensiveEvaluator

# Try to import OpenAI (optional for demo mode)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import gTTS for text-to-speech
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="SpeakWise - AI Speech Repair",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Model Loading (with caching)
# ============================================================================

@st.cache_resource
def load_repair_model(model_path: str):
    """
    Load repair model (cached)
    """
    # Check if model path exists and has valid files
    if not os.path.exists(model_path):
        return None, None

    # Check if it's a valid model directory (has config.json or pytorch_model.bin)
    config_exists = os.path.exists(os.path.join(model_path, "config.json"))
    model_exists = os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(model_path, "model.safetensors")) or \
                   os.path.exists(os.path.join(model_path, "adapter_model.bin"))

    if not (config_exists or model_exists):
        # Directory exists but no valid model files
        return None, None

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        return model, tokenizer

    except Exception:
        # Silently fail and return None - demo mode will be used
        return None, None


@st.cache_resource
def load_whisper_model(whisper_size: str = "base"):
    """
    Load Whisper ASR model (cached and independent of repair model)
    """
    try:
        asr = WhisperASR(model_size=whisper_size)
        return asr
    except Exception as e:
        st.warning(f"Could not load Whisper model: {e}")
        return None


def demo_repair_with_gpt(input_text: str, repair_intensity: float, api_key: str = None) -> str:
    """
    Use OpenAI GPT API for demo mode text repair
    """
    if not OPENAI_AVAILABLE:
        # Fallback to rule-based repair
        return rule_based_repair(input_text)

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # No API key, use rule-based fallback
        return rule_based_repair(input_text)

    try:
        client = OpenAI(api_key=api_key)

        # Create prompt based on repair intensity
        intensity_descriptions = {
            0.0: "Return the text exactly as-is with no changes.",
            1.0: "Fix only severe stuttering (e.g., I-I-I → I). Make minimal changes.",
            2.0: "Fix stuttering and basic grammar errors. Keep the speaking style informal.",
            3.0: "Improve clarity and grammar while maintaining a conversational tone.",
            4.0: "Maximize clarity and grammatical correctness while preserving meaning."
        }

        # Find closest intensity description
        closest_intensity = min(intensity_descriptions.keys(), key=lambda x: abs(x - repair_intensity))
        instruction = intensity_descriptions[closest_intensity]

        prompt = f"""You are helping a child with language disorders improve their speech clarity.

Instructions: {instruction}

Important guidelines:
- Preserve the child's voice and vocabulary
- Do not make the text too formal or adult-like
- Only fix what's necessary based on the repair intensity
- Maintain the original meaning and intent

Original text: "{input_text}"

Repaired text:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in helping children with language disorders improve speech clarity."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip().strip('"')

    except Exception as e:
        st.warning(f"GPT API call failed: {e}. Using rule-based repair.")
        return rule_based_repair(input_text)


def rule_based_repair(input_text: str) -> str:
    """
    Simple rule-based text repair as fallback
    """
    output_text = input_text

    # Remove stuttering (repeated characters with hyphens)
    output_text = re.sub(r'(\w)-\1+-\1+', r'\1', output_text)
    output_text = re.sub(r'(\w)-\1+', r'\1', output_text)

    # Remove word repetitions
    words = output_text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            cleaned_words.append(word)
    output_text = ' '.join(cleaned_words)

    # Basic grammar fixes for demo
    output_text = output_text.replace(' it ', ' ')
    output_text = output_text.replace(' he ', ' ')
    output_text = re.sub(r'\s+', ' ', output_text).strip()

    return output_text


def text_to_speech(text: str, lang: str = 'en') -> bytes:
    """
    Convert text to speech using gTTS

    Args:
        text: Text to convert to speech
        lang: Language code (default: 'en' for English)

    Returns:
        Audio bytes in MP3 format
    """
    if not GTTS_AVAILABLE:
        return None

    try:
        # Create TTS object
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            tmp_file.seek(0)

            # Read the audio bytes
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()

            # Clean up
            os.remove(tmp_file.name)

        return audio_bytes

    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


def generate_diff_html(original: str, repaired: str) -> str:
    """
    Generate HTML diff highlighting changes between original and repaired text

    Args:
        original: Original text
        repaired: Repaired text

    Returns:
        HTML string with highlighted differences
    """
    # Split into words for better diff granularity
    original_words = original.split()
    repaired_words = repaired.split()

    # Compute diff
    diff = difflib.ndiff(original_words, repaired_words)

    # Build HTML with color coding
    html_parts = []

    for item in diff:
        code = item[0]
        word = item[2:]

        if code == ' ':
            # Unchanged
            html_parts.append(f'<span style="color: white;">{word}</span>')
        elif code == '-':
            # Removed (show in red)
            html_parts.append(f'<span style="color: #ff4444; text-decoration: line-through;">{word}</span>')
        elif code == '+':
            # Added (show in green)
            html_parts.append(f'<span style="color: #44ff44; font-weight: bold;">{word}</span>')
        elif code == '?':
            # Skip the hint lines from ndiff
            continue

    html = ' '.join(html_parts)

    return f'<div style="background-color: #1e1e1e; padding: 15px; border-radius: 5px; font-family: monospace; line-height: 1.8;">{html}</div>'


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render application header"""
    st.title("SpeakWise: AI-Assisted Speech Repair")
    st.markdown("""
    **Helping children with language disorders communicate more clearly**

    This platform uses AI to perform "limited repair" on speech, improving clarity
    while preserving the speaker's natural style and voice.
    """)


def render_sidebar():
    """Render sidebar with settings"""
    st.sidebar.header("Settings")

    # Model settings
    st.sidebar.subheader("Model Configuration")

    model_path = st.sidebar.text_input(
        "Repair Model Path",
        value="./output/limited_repair_model",
        help="Path to fine-tuned repair model"
    )

    whisper_size = st.sidebar.selectbox(
        "Whisper Model Size",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )

    # Repair intensity slider
    st.sidebar.subheader("Repair Intensity (λ)")

    repair_intensity = st.sidebar.slider(
        "Intensity Level",
        min_value=0.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
        help="""
        - λ=0: No correction
        - λ=1: Minimal repair (severe stuttering only)
        - λ=2: Light repair (stuttering + basic grammar)
        - λ=3: Moderate repair (improve clarity)
        - λ=4: Full repair (maximum clarity)
        """
    )

    st.sidebar.markdown(f"""
    **Current Setting:** λ={repair_intensity}

    **What this means:**
    """)

    intensity_descriptions = {
        0.0: "No correction applied",
        0.5: "Minimal intervention",
        1.0: "Fix severe stuttering only",
        1.5: "Light stuttering correction",
        2.0: "Stuttering + basic grammar",
        2.5: "Enhanced grammar correction",
        3.0: "Moderate clarity improvement",
        3.5: "Strong clarity focus",
        4.0: "Maximum grammatical correction"
    }

    st.sidebar.info(intensity_descriptions.get(repair_intensity, "Custom intensity"))

    # API Key for demo mode
    st.sidebar.subheader("Demo Mode Settings")
    api_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Provide API key for GPT-powered demo repairs. Leave empty for rule-based repairs."
    )

    return model_path, whisper_size, repair_intensity, api_key


def render_text_input_tab(model, tokenizer, repair_intensity, api_key=None):
    """Render text input interface"""
    st.header("Text Input Mode")

    demo_mode = (model is None)
    if demo_mode and api_key:
        st.info("**DEMO MODE** - Using OpenAI GPT API for repairs")
    elif demo_mode:
        st.info("**DEMO MODE** - Using rule-based repair (add OpenAI API key in sidebar for better results)")

    st.markdown("Enter text directly for repair (useful for testing).")

    input_text = st.text_area(
        "Input Text",
        placeholder="Example: I-I-I want to go to the p-p-park today",
        height=100,
        help="Enter text with disfluencies, stuttering, or grammar errors"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        repair_button = st.button("Repair Text", type="primary")

    with col2:
        if st.button("Clear"):
            st.rerun()

    if repair_button and input_text:
        if model is None:
            # Demo mode - use GPT API or rule-based repair
            with st.spinner("Repairing text..."):
                output_text = demo_repair_with_gpt(input_text, repair_intensity, api_key)
        else:
            with st.spinner("Repairing text..."):
                # Perform repair with actual model
                output_text = repair_speech(
                    input_text,
                    model,
                    tokenizer,
                    repair_intensity=repair_intensity
                )

        # Display results
        st.subheader("Results")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Original:**")
            st.info(input_text)

        with col_b:
            st.markdown("**Repaired:**")
            st.success(output_text)

        # Show comparison metrics
        if input_text != output_text:
            st.markdown("**Changes Made:**")

            # Word count comparison
            original_words = len(input_text.split())
            repaired_words = len(output_text.split())

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Original Words", original_words)

            with metric_col2:
                st.metric("Repaired Words", repaired_words)

            with metric_col3:
                word_diff = repaired_words - original_words
                st.metric("Word Difference", word_diff)


def render_audio_input_tab(model, tokenizer, asr, repair_intensity, api_key=None):
    """Render audio input interface"""
    st.header("Audio Input Mode")

    if asr is None:
        st.warning("Whisper model not loaded. Audio transcription is unavailable.")
        st.info("Make sure Whisper is installed: `pip install openai-whisper`")
        return

    demo_mode = (model is None)
    if demo_mode and api_key:
        st.info("**DEMO MODE** - Using Whisper for transcription + GPT API for repairs")
    elif demo_mode:
        st.info("**DEMO MODE** - Using Whisper for transcription + rule-based repairs")
    else:
        st.success("Full mode - Using Whisper + trained repair model")

    st.markdown("Upload an audio file or record directly.")

    # Audio input method selection
    input_method = st.radio(
        "Input Method",
        options=["Upload Audio File", "Record Audio"],
        horizontal=True
    )

    audio_file = None
    audio_bytes = None

    if input_method == "Upload Audio File":
        st.markdown("**Upload an audio file:**")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
            help="Upload an audio file containing speech",
            key="audio_uploader"
        )
        if audio_file is not None:
            audio_bytes = audio_file.read()

    else:  # Record Audio
        st.markdown("**Record audio using your microphone:**")
        st.info("Click the record button below to start recording")

        # Use Streamlit's built-in audio recorder (available in Streamlit >= 1.28.0)
        try:
            recorded_audio = st.audio_input("Record your voice")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                audio_file = recorded_audio  # Treat recorded audio same as uploaded
                st.success("Audio recorded successfully")
        except AttributeError:
            # Fallback for older Streamlit versions
            st.warning("Audio recording requires Streamlit version 1.28.0 or higher")
            st.code("pip install streamlit --upgrade")
            st.markdown("**Alternative:** Switch to 'Upload Audio File' mode and use an external recorder.")

    if audio_bytes is not None and audio_file is not None:
        # Display audio player
        st.markdown("**Preview:**")
        st.audio(audio_bytes)

        col1, col2 = st.columns([1, 3])

        with col1:
            process_button = st.button("Process Audio", type="primary")

        if process_button:
            with st.spinner("Processing audio..."):
                # Save audio to temporary file
                # Determine file extension
                file_ext = '.wav'
                if hasattr(audio_file, 'name'):
                    file_ext = os.path.splitext(audio_file.name)[1] or '.wav'

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                try:
                    # Transcribe with Whisper
                    st.info("Transcribing audio...")
                    transcription_result = asr.transcribe_audio(
                        tmp_path,
                        return_confidence=True
                    )

                    transcribed_text = transcription_result['text']
                    confidence = transcription_result.get('confidence', 0.0)

                    # Repair text
                    st.info("Repairing text...")
                    if demo_mode:
                        repaired_text = demo_repair_with_gpt(transcribed_text, repair_intensity, api_key)
                    else:
                        repaired_text = repair_speech(
                            transcribed_text,
                            model,
                            tokenizer,
                            repair_intensity=repair_intensity
                        )

                    # Display results
                    st.success("Processing complete")

                    st.markdown("---")
                    st.subheader("Results")

                    # Section 1: Original Transcript
                    st.markdown("### 1. Original Transcript")
                    st.info(transcribed_text)
                    st.caption(f"ASR Confidence: {confidence:.2%}")

                    # Section 2: Repaired Transcript
                    st.markdown("### 2. Repaired Transcript")
                    st.success(repaired_text)

                    # Section 3: Original Audio
                    st.markdown("### 3. Original Audio")
                    st.audio(audio_bytes)
                    st.caption("Your uploaded/recorded audio")

                    # Section 4: Repaired Audio
                    st.markdown("### 4. Repaired Audio")
                    if GTTS_AVAILABLE:
                        with st.spinner("Generating speech..."):
                            tts_audio = text_to_speech(repaired_text)

                        if tts_audio:
                            st.audio(tts_audio, format='audio/mp3')
                            st.caption("Listen to the repaired speech")
                        else:
                            st.warning("Could not generate speech audio")
                    else:
                        st.warning("Text-to-Speech unavailable. Install gTTS: `pip install gTTS`")

                    # Section 5: Differences
                    st.markdown("---")
                    st.markdown("### Differences")
                    st.caption("Red = removed/errors | Green = added/corrections | White = unchanged")

                    diff_html = generate_diff_html(transcribed_text, repaired_text)
                    st.markdown(diff_html, unsafe_allow_html=True)

                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)


def render_batch_processing_tab(model, tokenizer, repair_intensity, api_key=None):
    """Render batch processing interface"""
    st.header("Batch Processing Mode")

    demo_mode = (model is None)
    if demo_mode and api_key:
        st.info("**DEMO MODE** - Using OpenAI GPT API for batch repairs")
    elif demo_mode:
        st.info("**DEMO MODE** - Using rule-based repair (add API key in sidebar for better results)")

    st.markdown("Process multiple text samples at once.")

    # Text area for batch input
    batch_input = st.text_area(
        "Batch Input (one per line)",
        placeholder="I-I-I want to play\nMy friend he goed home\nThe cat it runned away",
        height=200,
        help="Enter one sample per line"
    )

    if st.button("Process Batch", type="primary"):
        if batch_input:
            lines = [line.strip() for line in batch_input.split('\n') if line.strip()]

            with st.spinner(f"Processing {len(lines)} samples..."):
                results = []

                # Progress bar
                progress_bar = st.progress(0)

                for i, line in enumerate(lines):
                    if demo_mode:
                        # Use GPT API or rule-based repair
                        repaired = demo_repair_with_gpt(line, repair_intensity, api_key)
                    else:
                        repaired = repair_speech(
                            line,
                            model,
                            tokenizer,
                            repair_intensity=repair_intensity
                        )

                    results.append({
                        'original': line,
                        'repaired': repaired
                    })

                    # Update progress
                    progress_bar.progress((i + 1) / len(lines))

                # Display results
                st.subheader("Batch Results")

                for i, result in enumerate(results, 1):
                    with st.expander(f"Sample {i}", expanded=(i <= 3)):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original:**")
                            st.text(result['original'])

                        with col2:
                            st.markdown("**Repaired:**")
                            st.text(result['repaired'])


def render_evaluation_tab():
    """Render evaluation and metrics interface"""
    st.header("Evaluation & Metrics")

    st.markdown("Evaluate repair quality using multiple metrics.")

    st.info("Upload test data to compute evaluation metrics")

    # File upload for test data
    test_file = st.file_uploader(
        "Upload Test Data (JSON)",
        type=["json"],
        help="JSON file with format: [{\"original\": \"...\", \"repaired\": \"...\", \"reference\": \"...\"}]"
    )

    if test_file is not None:
        import json

        try:
            test_data = json.load(test_file)

            st.success(f"Loaded {len(test_data)} test samples")

            if st.button("Compute Metrics"):
                with st.spinner("Computing metrics..."):
                    evaluator = ComprehensiveEvaluator()

                    originals = [item['original'] for item in test_data]
                    predictions = [item['repaired'] for item in test_data]
                    references = [item['reference'] for item in test_data]

                    metrics = evaluator.evaluate_all(originals, predictions, references)

                    # Display metrics
                    st.subheader("Evaluation Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("BERTScore F1", f"{metrics['bertscore_f1']:.3f}")
                        st.metric("ROUGE-1", f"{metrics['rouge1']:.3f}")

                    with col2:
                        st.metric("Word Error Rate", f"{metrics['word_error_rate']:.3f}")
                        st.metric("Vocabulary Overlap", f"{metrics['vocabulary_overlap']:.3f}")

                    with col3:
                        st.metric("Readability (Flesch)", f"{metrics['pred_flesch_reading_ease']:.1f}")
                        st.metric("Grade Level", f"{metrics['pred_flesch_kincaid_grade']:.1f}")

        except Exception as e:
            st.error(f"Error loading test data: {e}")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application logic"""

    # Render header
    render_header()

    # Render sidebar and get settings
    model_path, whisper_size, repair_intensity, api_key = render_sidebar()

    # Load models separately
    with st.spinner("Loading models..."):
        # Load repair model (optional - for full mode)
        model, tokenizer = load_repair_model(model_path)

        # Load Whisper model (always try to load for audio transcription)
        asr = load_whisper_model(whisper_size)

    # Show appropriate status
    if model is None:
        if api_key:
            st.info("**DEMO MODE** - Using Whisper + OpenAI GPT API")
        else:
            st.warning("""
            **DEMO MODE** - Trained model not available

            - **Audio**: Whisper transcription is """ + ("available" if asr else "unavailable") + """
            - **Repair**: Using rule-based repair (add OpenAI API key in sidebar for better results)

            To enable full functionality, train the model:
            ```bash
            python train_limited_repair.py
            ```
            """)
    else:
        st.success("Full mode - All models loaded successfully")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Text Input",
        "Audio Input",
        "Batch Processing",
        "Evaluation"
    ])

    with tab1:
        render_text_input_tab(model, tokenizer, repair_intensity, api_key)

    with tab2:
        render_audio_input_tab(model, tokenizer, asr, repair_intensity, api_key)

    with tab3:
        render_batch_processing_tab(model, tokenizer, repair_intensity, api_key)

    with tab4:
        render_evaluation_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>SpeakWise - AI-Assisted Limited Repair Platform</p>
        <p style='font-size: 0.8em; color: gray;'>
            Helping children with language disorders communicate more clearly
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
