#!/bin/bash

# VoiceCraft Setup Script for macOS
echo "====================================================="
echo "          Setting up VoiceCraft for macOS            "
echo "====================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    echo "   You can download it from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $python_version"

# Create virtual environment
echo "🔄 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install requirements with increased timeout and retry
echo "🔄 Installing requirements (this may take a while)..."
echo "   Using increased timeout and retry settings for better reliability..."

# Install packages one by one with increased timeout
packages=(
    "streamlit==1.28.0"
    "librosa==0.10.1"
    "soundfile==0.12.1"
    "noisereduce==3.0.0"
    "numpy==1.24.3"
    "matplotlib==3.7.3"
    "ipywidgets==8.1.1"
    "torch==2.1.0"
    "openai-whisper==20231117"
    "f5-tts"
)

for package in "${packages[@]}"; do
    echo "   Installing $package..."
    pip install --timeout=120 --retries=5 $package
    
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Failed to install $package. Trying again..."
        pip install --timeout=180 --retries=10 $package
        
        if [ $? -ne 0 ]; then
            echo "❌ Error installing $package. Please check your internet connection and try again."
            echo "   You can continue setup and manually install this package later with:"
            echo "   source venv/bin/activate && pip install $package"
            read -p "Continue with setup? (y/n): " continue_setup
            if [[ $continue_setup != "y" && $continue_setup != "Y" ]]; then
                exit 1
            fi
        fi
    fi
done

# Install ffmpeg if not already installed
if ! command -v ffmpeg &> /dev/null; then
    echo "🔄 Installing ffmpeg using Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "   Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install ffmpeg
    if [ $? -ne 0 ]; then
        echo "❌ Error installing ffmpeg. Please install it manually:"
        echo "   brew install ffmpeg"
        exit 1
    fi
else
    echo "✅ ffmpeg is already installed."
fi

# Create data directory
echo "🔄 Creating data directory..."
mkdir -p data

# Try to download Whisper base model to avoid first-run delay
echo "🔄 Pre-downloading Whisper base model..."
python3 -c "import whisper; whisper.load_model('base')" || echo "⚠️ Warning: Failed to pre-download Whisper model. It will be downloaded on first use."

echo "====================================================="
echo "✅ Setup complete! To run VoiceCraft, use the following command:"
echo "   source venv/bin/activate && streamlit run app.py"

# Create a run script
cat > run.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run app.py
EOF

chmod +x run.sh

echo "   Or simply run: ./run.sh"
echo "====================================================="
echo "🚀 Starting VoiceCraft..."
./run.sh
