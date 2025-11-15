# How SpeakWise Works: Complete Technical Explanation

## Table of Contents
1. [Overall Architecture](#overall-architecture)
2. [Data Generation Process](#data-generation-process)
3. [Training Pipeline](#training-pipeline)
4. [Inference & Repair Mechanism](#inference--repair-mechanism)
5. [Example Walkthrough](#example-walkthrough)

---

## Overall Architecture

### The Big Picture

SpeakWise is a **controlled text generation system** that takes disfluent speech (with stuttering, grammar errors, repetitions) and repairs it while preserving the speaker's style. Here's the flow:

```
┌─────────────────┐
│  Audio Input    │ (Optional - via Whisper)
│  "I-I-I want... │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Whisper ASR     │ (Converts speech to text - independent)
│                 │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Disfluent Text  │
│ "I-I-I want to  │
│  go to p-park"  │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────┐
│              REPAIR ENGINE                  │
│  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Fine-tuned   │  │   Demo Mode         │ │
│  │ Phi-2 + LoRA │  │ • GPT-3.5 API       │ │
│  │ (Full Mode)  │  │ • Rule-based repair │ │
│  └──────────────┘  └─────────────────────┘ │
│         ↑ λ parameter controls intensity    │
└────────┬────────────────────────────────────┘
         │
         v
┌─────────────────┐
│ Repaired Text   │
│ "I want to go   │
│  to the park"   │
└─────────────────┘
```

### Key Components

1. **Data Generator** (`data_preparation.py`) - Creates synthetic training data
2. **Training Module** (`train_limited_repair.py`) - Fine-tunes Phi-2 with LoRA
3. **Whisper Integration** (`whisper_integration.py`) - Speech-to-text (independent)
4. **Evaluation** (`evaluation.py`) - Measures quality with multiple metrics
5. **Web UI** (`app.py`) - Interactive Streamlit interface with demo mode support
6. **Demo Mode** - Works without trained model using GPT API or rule-based repair

---

## Data Generation Process

### Step 1: Start with Clean Sentences

We begin with grammatically correct, child-appropriate sentences:

```python
clean_sentences = [
    "I want to go to the park today",
    "My favorite color is blue",
    "Can I have some water please",
    ...
]
```

### Step 2: Apply Synthetic Disfluencies

The `SyntheticDataGenerator` class applies different types of speech errors based on the **lambda (λ) level**. The lambda level represents how much repair will be needed:

#### Lambda Level Mapping

```python
λ = 0: No disfluencies (clean text)
λ = 1: Very minimal (10% intensity)  - Just slight stuttering
λ = 2: Light (25% intensity)         - Stuttering + basic grammar errors
λ = 3: Moderate (40% intensity)      - Above + word repetitions
λ = 4: Heavy (60% intensity)         - Above + filler words
```

### Step 3: Disfluency Transformations

Let's walk through how `"I want to go to the park"` becomes disfluent:

#### A. Stuttering (λ ≥ 1)

**Code:**
```python
def add_stuttering(self, text: str, intensity: float = 0.3) -> str:
    words = text.split()
    stuttered_words = []

    for word in words:
        # Add word repetition for common words
        if random.random() < intensity and word.lower() in ['I', 'the', 'a', 'to', ...]:
            repetitions = random.randint(2, 4)
            stuttered_words.append('-'.join([word] * repetitions))
        # Add syllable repetition
        elif random.random() < intensity * 0.5 and len(word) > 3:
            first_part = word[:2]
            rest = word[2:]
            repetitions = random.randint(2, 3)
            stuttered_words.append('-'.join([first_part] * repetitions) + rest)
```

**Example:**
- Input: `"I want to go to the park"`
- Output: `"I-I-I want to go to the pa-pa-park"`

**How it works:**
1. Split text into words: `["I", "want", "to", "go", "to", "the", "park"]`
2. For each word, randomly decide if it stutters based on `intensity` probability
3. Common words like "I", "the", "to" are more likely to stutter
4. Whole word repetition: `"I"` → `"I-I-I"`
5. Syllable repetition: `"park"` → `"pa-pa-park"` (first 2 chars repeated)

#### B. Grammar Errors (λ ≥ 2)

**Code:**
```python
def add_grammar_errors(self, text: str, intensity: float = 0.3) -> str:
    grammar_patterns = [
        ('went', 'goed'),      # Common child error
        ('ran', 'runned'),
        ('saw', 'seed'),
        ('I am', 'I'),
        ('I have', 'I'),
    ]

    for correct, error in grammar_patterns:
        if random.random() < intensity:
            text = text.replace(correct, error)
```

**Example:**
- Input: `"I went to the store"`
- Output: `"I goed to the store"`

#### C. Word Omissions (λ ≥ 2)

**Code:**
```python
def omit_words(self, text: str, intensity: float = 0.15) -> str:
    words = text.split()
    omit_candidates = ['is', 'are', 'am', 'the', 'a', 'an', 'to']

    result = []
    for word in words:
        if word.lower() in omit_candidates and random.random() < intensity:
            continue  # Skip this word
        result.append(word)
```

**Example:**
- Input: `"The dog is running"`
- Output: `"dog running"` (omitted "The" and "is")

#### D. Word Repetitions (λ ≥ 3)

**Code:**
```python
def add_word_repetitions(self, text: str, intensity: float = 0.2) -> str:
    words = text.split()
    result = []

    for word in words:
        result.append(word)
        if random.random() < intensity:
            result.append(word)  # Repeat the word
```

**Example:**
- Input: `"I like pizza"`
- Output: `"I I like like pizza pizza"`

#### E. Filler Words (λ ≥ 4)

**Code:**
```python
def add_filler_words(self, text: str, intensity: float = 0.2) -> str:
    fillers = ['um', 'uh', 'like', 'you know', 'I mean']
    words = text.split()
    result = []

    for i, word in enumerate(words):
        if random.random() < intensity and i > 0:
            result.append(random.choice(fillers))
        result.append(word)
```

**Example:**
- Input: `"I want pizza"`
- Output: `"I um want like pizza"`

### Step 4: Complete Example by Lambda Level

Starting sentence: **"I want to go to the park"**

```python
λ = 0: "I want to go to the park"
       (No changes - clean baseline)

λ = 1: "I-I want to go to the pa-park"
       (Only stuttering added, 10% intensity)

λ = 2: "I-I want go to pa-park"
       (Stuttering + omitted "to", 25% intensity)

λ = 3: "I-I want want go go to the the pa-pa-park"
       (Above + word repetitions, 40% intensity)

λ = 4: "I-I um want want like go go you know to the the um pa-pa-park"
       (Above + filler words, 60% intensity)
```

### Step 5: Create Training Pairs

**Code:**
```python
def generate_training_pairs(self, clean_sentences: List[str], num_variations: int = 3):
    training_data = []

    for sentence in clean_sentences:
        for _ in range(num_variations):
            # Random lambda level (1-4)
            lambda_level = random.randint(1, 4)

            # Generate disfluent version
            disfluent = self.generate_disfluent_version(sentence, lambda_level)

            training_data.append({
                'input': disfluent,
                'output': sentence,  # The clean version
                'lambda': lambda_level
            })

    return training_data
```

**What happens:**
1. Takes 20 clean sentences
2. Creates 50 variations per sentence (1000 total)
3. Each variation has random λ level (1-4)
4. Applies corresponding disfluencies
5. Saves as JSON:

```json
{
  "input": "I-I-I um want want to go like to the pa-park",
  "output": "I want to go to the park",
  "lambda": 4
}
```

---

## Training Pipeline

### Step 1: Prompt Engineering (Few-Shot Learning)

The model doesn't just see raw input/output pairs. It gets a **structured prompt** with examples:

```python
def create_prompt(input_text: str, repair_intensity: float,
                  include_output: bool = True, output_text: str = "") -> str:

    # System prompt defines the task
    system_prompt = """You are an expert AI assistant specialized in helping
    children with language disorders improve their speech clarity. Your role is
    to perform "limited repair" on their speech or text input.

    Guidelines:
    1. Preserve the child's original speaking style
    2. Correct only what is necessary based on repair intensity (λ)
    3. Do not over-correct or make speech too formal
    ...
    """

    # Few-shot examples teach the model
    few_shot_examples = """
    Example 1 (λ=1):
    Input: I-I-I want to go to the p-p-park today
    Output: I want to go to the park today

    Example 2 (λ=2):
    Input: Me and my friend we goed to store yesterday
    Output: Me and my friend went to the store yesterday

    Example 3 (λ=3):
    Input: The dog was so so big and and it runned really really fast
    Output: The dog was so big and it ran really fast
    """

    # Actual task
    prompt = f"""{system_prompt}

    {few_shot_examples}

    Now, perform limited repair on the following input:

    Repair Intensity (λ): {repair_intensity}
    Input: {input_text}
    Output: {output_text if include_output else ""}"""

    return prompt
```

**Generated prompt example:**

```
You are an expert AI assistant specialized in helping children...

Example 1 (λ=1):
Input: I-I-I want to go to the p-p-park today
Output: I want to go to the park today

Example 2 (λ=2):
Input: Me and my friend we goed to store yesterday
Output: Me and my friend went to the store yesterday

Example 3 (λ=3):
Input: The dog was so so big and and it runned really really fast
Output: The dog was so big and it ran really fast

Now, perform limited repair on the following input:

Repair Intensity (λ): 2
Input: I-I want go to the park
Output: I want to go to the park
```

### Step 2: Load and Prepare Data

**Code:**
```python
def prepare_dataset(data_path: str, tokenizer, config: ModelConfig):
    # Load raw JSON data
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    formatted_data = []
    for item in raw_data:
        # Create full prompt with few-shot examples
        full_prompt = create_prompt(
            input_text=item['input'],
            lambda_val=item['lambda'],
            include_output=True,
            output_text=item['output']
        )

        formatted_data.append({'text': full_prompt})

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    return tokenized
```

**What happens:**
1. Loads `data/train.json` (800 examples)
2. Each example gets wrapped in few-shot prompt
3. Tokenizes text into token IDs
4. Creates padding and attention masks

**Token Example:**
```
Text: "You are an expert... Output: I want to go to the park"
Tokens: [1, 887, 526, 385, 17924, ...]  (token IDs)
Length: 512 tokens (padded)
```

### Step 3: LoRA Configuration

Instead of training all 2.7 billion parameters of Phi-2, we use **LoRA (Low-Rank Adaptation)**:

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Rank - controls size of adapter
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.05,      # Regularization
    target_modules=[        # Which layers to adapt
        "q_proj",           # Query projection
        "v_proj",           # Value projection
        "k_proj",           # Key projection
        "o_proj"            # Output projection
    ],
    bias="none",
)
```

**How LoRA works:**

Instead of updating weight matrix W:
```
Normal fine-tuning: W → W + ΔW  (update all parameters)
```

LoRA decomposes the update:
```
LoRA: W → W + BA
where B is (d × r) and A is (r × d)
```

**Example (Phi-2):**
- Original weight matrix: 2560 × 2560 = 6.5M parameters (per layer)
- LoRA with r=8: (2560 × 8) + (8 × 2560) = 40K parameters
- **Reduction: 163x fewer parameters to train per layer!**

**Trainable parameters:**
```
Original Phi-2:     2,779,683,840 parameters
With LoRA:              1,703,936 parameters (0.06%!)
```

**Why Phi-2?**
- Smaller than Llama 3 (2.7B vs 8B) = faster training & inference
- Lower VRAM requirements (6GB vs 12GB+)
- Fully open-source with no restrictions
- Excellent performance for its size

### Step 4: Training Loop

```python
training_args = TrainingArguments(
    output_dir="./output/limited_repair_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 4×4 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    fp16=True,                      # Mixed precision
    gradient_checkpointing=True,    # Save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**What happens during training:**

1. **Forward pass:**
   - Input: Tokenized prompt (512 tokens)
   - Model predicts next token at each position
   - Compare prediction with actual next token

2. **Loss calculation:**
   ```python
   # Causal Language Modeling loss
   loss = CrossEntropyLoss(predictions, labels)
   ```

3. **Backward pass:**
   - Compute gradients only for LoRA parameters
   - Update using AdamW optimizer

4. **Repeat for 3 epochs:**
   - Epoch 1: Model learns basic patterns
   - Epoch 2: Refines understanding of λ levels
   - Epoch 3: Fine-tunes repair quality

**Memory optimization:**
- 8-bit quantization: Base model uses 8-bit integers (saves ~50% VRAM)
- Gradient checkpointing: Recomputes activations during backward pass
- Mixed precision: Uses FP16 for speed, FP32 for stability

---

## Inference & Repair Mechanism

### Repair Modes

SpeakWise supports **three repair modes**:

#### Mode 1: Full Mode (Trained Phi-2 Model)
```python
def repair_speech(input_text: str, model, tokenizer, repair_intensity: float = 2.0):
    # 1. Create prompt (WITHOUT output this time)
    prompt = create_prompt(input_text, repair_intensity, include_output=False)

    # Example prompt:
    # """You are an expert...
    # Example 1 (λ=1): ...
    #
    # Repair Intensity (λ): 2.0
    # Input: I-I-I want to go to park
    # Output:"""

    # 2. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Generate completion
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,      # Generate up to 128 tokens
        temperature=0.7,         # Randomness (lower = more deterministic)
        top_p=0.9,              # Nucleus sampling
        do_sample=True,         # Enable sampling
    )

    # 4. Decode tokens back to text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 5. Extract just the output part
    if "Output:" in full_output:
        repaired_text = full_output.split("Output:")[-1].strip()

    return repaired_text
```

#### Mode 2: Demo Mode with GPT API
```python
def demo_repair_with_gpt(input_text: str, repair_intensity: float, api_key: str):
    # Uses OpenAI GPT-3.5-turbo for repairs
    client = OpenAI(api_key=api_key)

    # Create intensity-specific instruction
    intensity_map = {
        0.0: "Return the text exactly as-is",
        1.0: "Fix only severe stuttering",
        2.0: "Fix stuttering and basic grammar",
        3.0: "Improve clarity and grammar",
        4.0: "Maximize clarity and correctness"
    }

    prompt = f"""You are helping a child with language disorders.
    Instructions: {intensity_map[repair_intensity]}
    Original text: "{input_text}"
    Repaired text:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
```

#### Mode 3: Rule-Based Fallback
```python
def rule_based_repair(input_text: str) -> str:
    # Simple pattern-based repair
    output = input_text

    # Remove stuttering (I-I-I → I)
    output = re.sub(r'(\w)-\1+-\1+', r'\1', output)

    # Remove word repetitions
    words = output.split()
    cleaned = [w for i, w in enumerate(words)
               if i == 0 or w.lower() != words[i-1].lower()]
    output = ' '.join(cleaned)

    return output
```

### How the App Selects Repair Mode

The Streamlit app automatically chooses the best available repair mode:

```python
# In app.py - Model loading
@st.cache_resource
def load_repair_model(model_path: str):
    if not os.path.exists(model_path):
        return None, None  # Triggers demo mode

    # Try to load trained model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, ...)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except:
        return None, None  # Falls back to demo mode

# Mode selection during repair
if model is None:
    # Demo mode
    if api_key:
        output = demo_repair_with_gpt(input_text, repair_intensity, api_key)
    else:
        output = rule_based_repair(input_text)
else:
    # Full mode with trained model
    output = repair_speech(input_text, model, tokenizer, repair_intensity)
```

**Decision flow:**
```
Is trained model available?
├─ YES → Use Full Mode (Phi-2 + LoRA)
└─ NO → Is OpenAI API key provided?
    ├─ YES → Use Demo Mode with GPT
    └─ NO → Use Rule-Based Fallback
```

### Generation Process Step-by-Step (Full Mode)

Let's say input is: `"I-I-I want to go to park"` with λ=2

**Step 1: Tokenization**
```
Prompt: "You are an expert... Repair Intensity (λ): 2 Input: I-I-I want to go to park Output:"
Tokens: [1, 887, 526, ..., 4451, 20986, 29901] (total: 384 tokens)
```

**Step 2: Autoregressive Generation**

The model generates one token at a time:

```
Token 1:  "I"       (logits → softmax → sample → "I")
Token 2:  "want"    (given "I", predict next)
Token 3:  "to"      (given "I want", predict next)
Token 4:  "go"
Token 5:  "to"
Token 6:  "the"     (← Notice model ADDS missing word!)
Token 7:  "park"
Token 8:  <EOS>     (End of sequence)
```

**How it "knows" to repair:**
1. Few-shot examples teach pattern: disfluent → fluent
2. λ value conditions the model on repair intensity
3. Training on 800 examples reinforces this behavior
4. Model learns:
   - Remove stuttering (I-I-I → I)
   - Add missing words (go to park → go to the park)
   - Fix grammar based on λ level

**Step 3: Decoding Parameters**

```python
temperature=0.7  # Controls randomness
# Lower (0.1): "I want to go to the park" (deterministic)
# Higher (1.5): "I wanna go to park" (more creative)

top_p=0.9  # Nucleus sampling
# Only sample from tokens that make up top 90% probability mass
# Prevents model from choosing unlikely/weird words
```

---

## Example Walkthrough: Complete Pipeline

Let's trace one complete example from data generation to inference.

### Step 1: Data Generation

**Input (clean sentence):**
```python
clean = "I want to eat pizza"
```

**Generate disfluent version (λ=3):**
```python
generator = SyntheticDataGenerator()

# Apply λ=3 transformations:
# 1. Stuttering (intensity=0.4 × 0.5 = 0.2)
disfluent = "I-I want to eat pizza"

# 2. Grammar errors (intensity=0.4)
# No applicable patterns, skip

# 3. Word omissions (intensity=0.4 × 0.5 = 0.2)
disfluent = "I-I want eat pizza"  # Omitted "to"

# 4. Word repetitions (intensity=0.4 × 0.7 = 0.28)
disfluent = "I-I want want eat pizza pizza"

# Final result:
{
    "input": "I-I want want eat pizza pizza",
    "output": "I want to eat pizza",
    "lambda": 3
}
```

### Step 2: Training Data Format

**Saved to train.json:**
```json
{
  "input": "I-I want want eat pizza pizza",
  "output": "I want to eat pizza",
  "lambda": 3
}
```

**Converted to training prompt:**
```
You are an expert AI assistant specialized in helping children...

Example 1 (λ=1):
Input: I-I-I want to go to the p-p-park today
Output: I want to go to the park today

Example 2 (λ=2):
Input: Me and my friend we goed to store yesterday
Output: Me and my friend went to the store yesterday

Example 3 (λ=3):
Input: The dog was so so big and and it runned really really fast
Output: The dog was so big and it ran really fast

Now, perform limited repair on the following input:

Repair Intensity (λ): 3
Input: I-I want want eat pizza pizza
Output: I want to eat pizza
```

### Step 3: Model Training

**Tokenization:**
```python
tokens = [1, 887, 526, 385, 17924, ..., 306, 864, 304, 17545, 21230, 21230, 13, 3744, 29901, 306, 864, 304, 17545, 21230]
          ↑                              ↑                                                           ↑
       <BOS>                          "I-I want want eat pizza pizza"                        "I want to eat pizza"
```

**Training objective:**
- Predict each token given previous tokens
- Loss measures how well model predicts "I want to eat pizza" after seeing the prompt

**After 3 epochs:**
- Model learns that λ=3 means moderate repair
- Learns to:
  - Remove stuttering (I-I → I)
  - Remove repetitions (want want → want, pizza pizza → pizza)
  - Add missing words (eat → to eat)

### Step 4: Inference (Using Trained Model)

**User input:**
```python
input_text = "I-I-I want want eat ice cream cream"
lambda_val = 3
```

**Create inference prompt (no output):**
```
You are an expert AI assistant...

[Few-shot examples...]

Repair Intensity (λ): 3
Input: I-I-I want want eat ice cream cream
Output:
```

**Model generates:**
```
Token by token:
1. "I"          (removes I-I-I)
2. "want"       (removes repetition)
3. "to"         (adds missing word!)
4. "eat"
5. "ice"
6. "cream"      (removes repetition)
7. <EOS>

Final: "I want to eat ice cream"
```

---

## Why This Works

### 1. Few-Shot Learning
The model sees examples before each task, teaching it the pattern:
```
Disfluent + λ level → Clean text
```

### 2. Controllable Generation
The λ parameter conditions the model:
- λ=1: Minimal changes
- λ=4: Aggressive repair

### 3. LoRA Efficiency
- Trains only 0.06% of parameters (Phi-2)
- Fast, memory-efficient (6GB VRAM vs 12GB+)
- Preserves base model knowledge
- Lower hardware requirements

### 4. Synthetic Data Quality
- Realistic disfluencies
- Diverse λ levels
- Enough examples (800) to learn patterns

### 5. Proper Prompting
- System message defines role
- Few-shot examples demonstrate task
- Structured format guides generation

### 6. Graceful Degradation (Demo Mode)
- Works without trained model
- GPT API mode: High-quality repairs using OpenAI
- Rule-based fallback: No external dependencies
- Whisper works independently for audio transcription

---

## Summary

**Data Generation:**
1. Start with clean sentences
2. Apply synthetic disfluencies based on λ level
3. Create {input, output, lambda} pairs
4. Save as JSON

**Training:**
1. Wrap data in few-shot prompts
2. Tokenize for model input
3. Fine-tune Phi-2 with LoRA (only ~1.7M parameters)
4. Train for 3 epochs

**Inference - Three Modes:**

**Full Mode (Best quality):**
1. Create prompt with input + λ
2. Phi-2 model generates repair autoregressively
3. Extract output text
4. Return repaired speech

**Demo Mode with GPT (Good quality):**
1. Create GPT prompt with intensity instructions
2. Call OpenAI API
3. Return GPT-3.5-turbo response

**Rule-Based Fallback (Basic quality):**
1. Apply regex patterns to remove stuttering
2. Remove word repetitions
3. Return cleaned text

The magic is in the combination of:
- Realistic synthetic data
- Few-shot prompting
- Efficient LoRA training with Phi-2
- Controllable generation via λ parameter
- Graceful degradation for usability without training
