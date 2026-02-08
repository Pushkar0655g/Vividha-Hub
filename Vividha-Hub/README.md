# Vividha Hub
AI-Powered Multilingual Video Dubbing & Subtitling Engine

## Overview
Vividha Hub is an AI-driven video localization system that automates:

• Speech transcription using OpenAI Whisper  
• Speaker diarization using Pyannote  
• Voice cloning using Coqui XTTS-v2  
• Background music preservation using Demucs  
• Subtitle rendering with language-specific fonts  
• CUDA-accelerated video rendering with FFmpeg  

The system reduces localization costs by up to 90% and processes videos 10x faster than traditional manual workflows.

---

## Key Features

- 🎙 Speaker-preserving voice cloning
- 🌍 Multilingual dubbing (17+ languages)
- 🎵 Background music retention
- ⚡ GPU optimized pipeline (70–80% utilization)
- 🎬 Automatic subtitle generation
- 🔁 Adaptive speech rate synchronization

---

## Architecture

Video Upload  
→ Audio Extraction  
→ Whisper Transcription  
→ Translation  
→ Speaker Diarization  
→ XTTS Voice Cloning  
→ Time-Stretch Synchronization  
→ Audio Mixing  
→ Final Video Rendering  

---

## Tech Stack

Backend:
- Python
- PyTorch
- OpenAI Whisper
- Pyannote.audio
- Coqui XTTS-v2
- Demucs
- Librosa
- MoviePy
- FFmpeg (CUDA)

Frontend:
- HTML
- CSS
- JavaScript
- Electron (Node.js)

---

## Setup Instructions

### 1. Clone Repository

git clone https://github.com/Pushkar0655/Vividha-Hub.git
cd Vividha-Hub
### 2. Install Backend Dependencies
pip install -r requirements.txt

### 3. Set HuggingFace Token

Windows:
set HF_TOKEN=your_token_here

Mac/Linux:
export HF_TOKEN=your_token_here

### 4. Run Backend
python backend/backend.py --video sample.mp4 --input_lang english --audio_lang hindi --subtitle_lang german

### 5. Run Frontend
npm install
npm start

---

## Demo Videos

See project demonstrations in docs/ or Google Drive links.

---

## Research & Innovation

- Speaker-aware multilingual dubbing
- Dynamic speech duration alignment
- Integrated background audio preservation
- End-to-end automated pipeline

---

## Future Scope

- Lip-sync integration (Wav2Lip)
- Standalone desktop application
- Additional language support
- API deployment

---

## License
Open-source for educational and research purposes.


