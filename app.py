import warnings
# Move this to the very top of your file, before any other imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import streamlit as st
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import whisper
import torch
import tempfile
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import datetime
import torch
import warnings

# Suppress the specific torch.classes warning
warnings.filterwarnings("ignore", message=".*Tried to instantiate class '__path__._path'.*")

# Set page config
st.set_page_config(
    page_title="VoiceCraft",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize session state variables
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

# Sidebar for project management
with st.sidebar:
    st.title("🎙️ VoiceCraft")
    st.markdown("---")
    
    # Project management
    st.subheader("Project Management")
    
    # Create new project
    new_project_name = st.text_input("New Project Name")
    if st.button("Create Project"):
        if new_project_name:
            project_id = datetime.now().strftime("%Y%m%d%H%M%S")
            project_dir = os.path.join(DATA_DIR, project_id)
            os.makedirs(project_dir, exist_ok=True)
            
            st.session_state.projects[project_id] = {
                "name": new_project_name,
                "dir": project_dir,
                "original_audio": None,
                "cleaned_audio": None,
                "transcription": None,
                "cloned_audio": None
            }
            st.session_state.current_project_id = project_id
            st.success(f"Project '{new_project_name}' created!")
        else:
            st.error("Please enter a project name")
    
    # Select existing project
    if st.session_state.projects:
        st.markdown("---")
        st.subheader("Select Project")
        project_options = {f"{data['name']} ({pid})": pid for pid, data in st.session_state.projects.items()}
        selected_project = st.selectbox(
            "Choose a project",
            options=list(project_options.keys()),
            index=0 if st.session_state.current_project_id else None
        )
        
        if selected_project:
            selected_pid = project_options[selected_project]
            st.session_state.current_project_id = selected_pid
    
    st.markdown("---")
    st.info("Made with ❤️ by VoiceCraft")

# Main content
if st.session_state.current_project_id:
    project = st.session_state.projects[st.session_state.current_project_id]
    
    st.title(f"Project: {project['name']}")
    
    # Create tabs for different functionalities
    tabs = st.tabs(["Audio Processing", "Transcription", "Voice Cloning"])
    
    # Tab 1: Audio Processing
    with tabs[0]:
        st.header("Audio Processing")
        
        # Upload audio file
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
        
        if uploaded_file is not None:
            # Save the uploaded file
            original_path = os.path.join(project['dir'], "original_audio.wav")
            
            with open(original_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            project['original_audio'] = original_path
            st.success("Audio file uploaded successfully!")
            
            # Display audio waveform
            y, sr = librosa.load(original_path, sr=None)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue', alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Original Audio Waveform")
            st.pyplot(fig)
            
            # Audio player
            st.audio(original_path)
            
            # Noise reduction options
            st.subheader("Noise Reduction")
            apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=True)
            
            if st.button("Process Audio"):
                with st.spinner("Processing audio..."):
                    # Load the audio file
                    audio_data, sample_rate = librosa.load(original_path, sr=None)
                    
                    cleaned_path = os.path.join(project['dir'], "cleaned_audio.wav")
                    
                    if apply_noise_reduction:
                        # Perform noise reduction
                        reduced_noise = nr.reduce_noise(
                            y=audio_data,
                            sr=sample_rate,
                            stationary=True,
                            prop_decrease=1.0
                        )
                        # Save the cleaned audio
                        sf.write(cleaned_path, reduced_noise, sample_rate)
                        st.success("Noise reduction completed!")
                    else:
                        # Just copy the file without noise reduction
                        sf.write(cleaned_path, audio_data, sample_rate)
                        st.success("File processed without noise reduction.")
                    
                    project['cleaned_audio'] = cleaned_path
                    
                    # Display cleaned audio waveform
                    y_cleaned, sr_cleaned = librosa.load(cleaned_path, sr=None)
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(np.linspace(0, len(y_cleaned)/sr_cleaned, len(y_cleaned)), y_cleaned, color='green', alpha=0.7)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Processed Audio Waveform")
                    st.pyplot(fig)
                    
                    # Audio player for cleaned audio
                    st.subheader("Processed Audio")
                    st.audio(cleaned_path)
        
        elif project.get('original_audio') and os.path.exists(project['original_audio']):
            st.success("Audio file already uploaded.")
            
            # Display audio waveform
            y, sr = librosa.load(project['original_audio'], sr=None)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue', alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Original Audio Waveform")
            st.pyplot(fig)
            
            # Audio player
            st.audio(project['original_audio'])
            
            if project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']):
                # Display cleaned audio waveform
                y_cleaned, sr_cleaned = librosa.load(project['cleaned_audio'], sr=None)
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, len(y_cleaned)/sr_cleaned, len(y_cleaned)), y_cleaned, color='green', alpha=0.7)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Processed Audio Waveform")
                st.pyplot(fig)
                
                # Audio player for cleaned audio
                st.subheader("Processed Audio")
                st.audio(project['cleaned_audio'])
    
    # Add this to the Audio Processing tab after displaying the processed audio
        if project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']):
            st.markdown("---")
            st.subheader("Optimize for Voice Cloning")
            st.write("Trimming the audio to a shorter duration may improve voice cloning performance.")
            
            trim_duration = st.slider("Trim Duration (seconds)", min_value=5, max_value=30, value=10, key="trim_duration")
            
            if st.button("Trim Audio", key="trim_audio_button"):
                with st.spinner(f"Trimming audio to {trim_duration} seconds..."):
                    try:
                        # Load the audio
                        y, sr = librosa.load(project['cleaned_audio'], sr=None)
                        
                        # Trim to specified duration
                        samples = min(len(y), int(trim_duration * sr))
                        trimmed_y = y[:samples]
                        
                        # Save trimmed audio
                        trimmed_path = os.path.join(project['dir'], "trimmed_audio.wav")
                        sf.write(trimmed_path, trimmed_y, sr)
                        
                        # Update project
                        project['trimmed_audio'] = trimmed_path
                        
                        st.success(f"Audio trimmed to {trim_duration} seconds!")
                        st.audio(trimmed_path)
                    except Exception as e:
                        st.error(f"Error trimming audio: {str(e)}")

    
    # Tab 2: Transcription
    with tabs[1]:
        st.header("Audio Transcription")
        
        if project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']):
            # Model selection
            model_size = st.selectbox(
                "Select Whisper Model Size",
                options=["tiny", "base", "small", "medium", "large"],
                index=1  # Default to "base"
            )
            
            if st.button("Transcribe Audio"):
                with st.spinner(f"Loading Whisper {model_size} model and transcribing audio..."):
                    try:
                        # Load model
                        model = whisper.load_model(model_size)
                        st.session_state.whisper_model = model
                        
                        # Transcribe
                        result = model.transcribe(
                            project['cleaned_audio'],
                            fp16=False,
                            language='en',
                            verbose=False,
                            temperature=0,
                        )
                        
                        # Get the transcribed text
                        transcribed_text = result["text"]
                        
                        # Save the transcription
                        transcription_path = os.path.join(project['dir'], "transcription.txt")
                        with open(transcription_path, "w", encoding="utf-8") as f:
                            f.write(transcribed_text)
                        
                        project['transcription'] = transcription_path
                        st.success("Transcription completed!")
                        
                        # Display transcription
                        st.subheader("Transcription Result")
                        st.text_area("Transcribed Text", transcribed_text, height=200)
                        
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
            
            # Display existing transcription if available
            if project.get('transcription') and os.path.exists(project['transcription']):
                with open(project['transcription'], 'r', encoding='utf-8') as f:
                    transcribed_text = f.read()
                
                st.subheader("Existing Transcription")
                st.text_area("Transcribed Text", transcribed_text, height=200, key="transcribed_text_display")
                
                # Allow editing transcription
                edited_text = st.text_area("Edit Transcription", transcribed_text, height=200, key="edit_transcription")
                
                if st.button("Save Edited Transcription"):
                    with open(project['transcription'], 'w', encoding='utf-8') as f:
                        f.write(edited_text)
                    st.success("Transcription updated!")
        else:
            st.warning("Please process an audio file first in the 'Audio Processing' tab.")
    
    # Tab 3: Voice Cloning
    with tabs[2]:
        st.header("Voice Cloning")
        
        if project.get('cleaned_audio') and project.get('transcription') and os.path.exists(project['cleaned_audio']) and os.path.exists(project['transcription']):
            # Load transcription as reference text
            with open(project['transcription'], 'r', encoding='utf-8') as f:
                ref_text = f.read()
            
            # Display reference text
            st.subheader("Reference Text (from transcription)")
            st.text_area("Reference Text", ref_text, height=100, key="ref_text_display", disabled=True)
            
            # Text to generate with cloned voice
            gen_text = st.text_area(
                "Text to Generate with Cloned Voice", 
                value="This is a sample text generated with my cloned voice.",
                height=100,
                key="gen_text_input"
            )
            
            # Add debug mode option
            debug_mode = st.checkbox("Debug Mode (Show command details)", key="debug_mode")
            
            # Voice cloning button
            if st.button("Clone Voice", key="clone_voice_button"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing voice cloning process...")
                progress_bar.progress(10)
                
                try:
                    # Set output path
                    output_path = os.path.join(project['dir'], "cloned_voice.wav")
                    
                    # Prepare the command
                    cmd = [
                        "f5-tts_infer-cli",
                        "--model", "F5TTS_v1_Base",
                        "--ref_audio", project['cleaned_audio'],
                        "--ref_text", ref_text,
                        "--gen_text", gen_text,
                        "--output_file", output_path,
                        "--device", "cpu"  # Force CPU for macOS compatibility
                    ]
                    
                    # Show command in debug mode
                    if debug_mode:
                        st.code(" ".join(cmd), language="bash")
                        st.write("Running this command directly in terminal might provide more information.")
                    
                    status_text.text("Running voice cloning process (this may take several minutes)...")
                    progress_bar.progress(20)
                    
                    # Use subprocess with timeout
                    import subprocess
                    
                    # Run the process with a timeout
                    try:
                        process = subprocess.run(
                            cmd, 
                            capture_output=True, 
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                        
                        # Check if process was successful
                        if process.returncode == 0:
                            status_text.text("Voice cloning completed!")
                            progress_bar.progress(100)
                            project['cloned_audio'] = output_path
                            st.success("Voice cloning completed successfully!")
                            
                            # Display cloned audio
                            st.subheader("Cloned Voice")
                            st.audio(output_path)
                        else:
                            status_text.text("Process failed")
                            progress_bar.progress(100)
                            st.error(f"Voice cloning failed: {process.stderr}")
                            
                            if debug_mode:
                                st.subheader("Error Details")
                                st.code(process.stderr)
                                st.subheader("Output")
                                st.code(process.stdout)
                    
                    except subprocess.TimeoutExpired:
                        status_text.text("Process timed out after 5 minutes")
                        progress_bar.progress(100)
                        st.error("Voice cloning process timed out. This might be due to insufficient resources or a problem with the model.")
                
                except Exception as e:
                    status_text.text("An error occurred")
                    progress_bar.progress(100)
                    st.error(f"An error occurred during voice cloning: {str(e)}")
                    import traceback
                    if debug_mode:
                        st.code(traceback.format_exc())
            
            # Add alternative voice synthesis option
            st.markdown("---")
            st.subheader("Alternative: Simple Voice Synthesis")
            st.write("If F5-TTS is too resource-intensive, try this simpler voice synthesis:")
            
            simple_text = st.text_area(
                "Text to synthesize", 
                value="This is a simple voice synthesis alternative that works faster on most computers.",
                height=100,
                key="simple_synthesis_text"
            )
            
            if st.button("Generate Simple Voice", key="generate_simple_voice"):
                with st.spinner("Generating voice..."):
                    try:
                        # Install pyttsx3 if not already installed
                        try:
                            import pyttsx3
                        except ImportError:
                            st.info("Installing pyttsx3...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
                            import pyttsx3
                        
                        # Initialize the TTS engine
                        engine = pyttsx3.init()
                        
                        # Set properties
                        engine.setProperty('rate', 150)  # Speed of speech
                        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
                        
                        # Output file path
                        simple_output_path = os.path.join(project['dir'], "simple_voice.wav")
                        
                        # Save to file
                        engine.save_to_file(simple_text, simple_output_path)
                        engine.runAndWait()
                        
                        st.success("Voice generated successfully!")
                        st.audio(simple_output_path)
                        
                    except Exception as e:
                        st.error(f"Error generating voice: {str(e)}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.warning("Please process an audio file and transcribe it first.")

else:
    st.title("Welcome to VoiceCraft")
    st.write("Create a new project or select an existing one from the sidebar to get started.")
    
    # Display sample workflow
    st.header("How to use VoiceCraft")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Audio Processing")
        st.write("Upload an audio file and apply noise reduction to clean it.")
        st.image("https://via.placeholder.com/300x200?text=Audio+Processing", use_column_width=True)
    
    with col2:
        st.subheader("2. Transcription")
        st.write("Transcribe your audio to text using OpenAI's Whisper model.")
        st.image("https://via.placeholder.com/300x200?text=Transcription", use_column_width=True)
    
    with col3:
        st.subheader("3. Voice Cloning")
        st.write("Clone the voice from your audio to generate new speech.")
        st.image("https://via.placeholder.com/300x200?text=Voice+Cloning", use_column_width=True)

# Add this to the sidebar
with st.sidebar:
    if st.checkbox("Show System Info", key="show_system_info"):
        st.subheader("System Information")
        
        try:
            # Try to import psutil
            try:
                import psutil
            except ImportError:
                st.info("Installing psutil...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
                import psutil
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            st.write(f"CPU Usage: {cpu_percent}%")
            
            # Memory info
            memory = psutil.virtual_memory()
            st.write(f"Memory Usage: {memory.percent}%")
            st.write(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
            
            # Disk info
            disk = psutil.disk_usage('/')
            st.write(f"Disk Usage: {disk.percent}%")
            st.write(f"Free Disk Space: {disk.free / (1024 ** 3):.2f} GB")
            
            # Check if CUDA is available
            st.write(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.write("No GPU available - voice cloning will be slower")
                
            # Python version
            import platform
            st.write(f"Python Version: {platform.python_version()}")
            
            # PyTorch version
            st.write(f"PyTorch Version: {torch.__version__}")
        
        except Exception as e:
            st.error(f"Error getting system info: {str(e)}")
