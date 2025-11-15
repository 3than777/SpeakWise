@echo off
REM Setup conda environment for SpeakWise project

echo Creating conda environment: speakwise
conda create -n speakwise python=3.10 -y

echo Activating environment...
call conda activate speakwise

echo Installing PyTorch with CUDA support...
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo Installing core dependencies...
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install peft>=0.7.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0

echo Installing LLaMA-Factory...
pip install git+https://github.com/hiyouga/LLaMA-Factory.git

echo Installing Whisper and audio processing...
pip install openai-whisper
pip install librosa>=0.10.0
pip install soundfile>=0.12.0

echo Installing evaluation metrics...
pip install bert-score>=0.3.13
pip install rouge-score>=0.1.2
pip install textstat>=0.7.3
pip install jiwer>=3.0.0

echo Installing UI and utilities...
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install tqdm>=4.65.0

echo Installing additional tools...
pip install wandb
pip install tensorboard

echo Environment setup complete!
echo To activate: conda activate speakwise
pause
