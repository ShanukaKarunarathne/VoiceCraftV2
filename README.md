# VoiceCraft

VoiceCraft is an all-in-one voice processing tool that allows you to:

- Clean and reduce noise in audio recordings
- Transcribe audio to text using OpenAI's Whisper
- Clone voices to generate new speech with the same voice characteristics

## Requirements

- macOS
- Python 3.8 or higher
- ffmpeg (installed automatically during setup)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/VoiceCraft.git
cd VoiceCraft
```

2. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment
- Install all required dependencies
- Install ffmpeg if not already present
- Create a run script

## Usage

1. Start the application:

```bash
./run.sh
```

2. Open your browser at http://localhost:8501

3. Follow the workflow:
   - Create a new project
   - Upload an audio file
   - Process the audio (with optional noise reduction)
   - Transcribe the audio
   - Clone the voice to generate new speech

## Features

### Audio Processing

- Upload audio files (WAV, MP3, M4A, OGG)
- Apply noise reduction
- Visualize audio waveforms
- Play processed audio

### Transcription

- Transcribe audio using OpenAI's Whisper model
- Choose from different model sizes (tiny, base, small, medium, large)
- Edit and save transcriptions

### Voice Cloning

- Clone voices using F5-TTS
- Generate new speech with the cloned voice
- Play generated audio

## Troubleshooting

If you encounter any issues:

1. Make sure you have the latest version of Python installed
2. Check that ffmpeg is properly installed
3. Try reinstalling the dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

````

### 5. Create a run script

```bash:run.sh
#!/bin/bash
source venv/bin/activate
streamlit run app.py
````

## How to Test VoiceCraft on macOS

1. First, make sure you have Python 3.8+ installed on your Mac. You can check by running:

```bash
python3 --version
```

2. Create a new directory for the project and set up the files:

```bash
mkdir -p VoiceCraft
cd VoiceCraft
```

3. Create all the files I provided above:

   - `requirements.txt`
   - `app.py`
   - `setup.sh`
   - `README.md`
   - `run.sh`

4. Make the scripts executable:

```bash
chmod +x setup.sh run.sh
```

5. Run the setup script:

```bash
./setup.sh
```

6. Start the application:

```bash
./run.sh
```
