import streamlit as st
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import torch
import tempfile
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
import ssl
import certifi

# Fix SSL certificate verification issues
ssl_context = ssl.create_default_context(cafile=certifi.where())
import urllib.request
urllib.request._opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ssl_context)
)

# Custom function to safely load Whisper model
def load_whisper_model_safely(model_name="base"):
    """Load whisper model with SSL verification handling"""
    import whisper
    
    try:
        # First try normal loading
        return whisper.load_model(model_name)
    except Exception as e:
        if "certificate verify failed" in str(e):
            # Create an unverified context
            ssl_context = ssl._create_unverified_context()
            
            # Patch urllib to use the unverified context
            original_opener = urllib.request._opener
            urllib.request._opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            
            try:
                # Try loading again with the patched context
                return whisper.load_model(model_name)
            finally:
                # Restore the original opener
                urllib.request._opener = original_opener
        else:
            # If it's not a certificate error, re-raise
            raise

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

# Load existing projects from disk
def load_existing_projects():
    if not os.path.exists(DATA_DIR):
        return
    
    for project_id in os.listdir(DATA_DIR):
        project_dir = os.path.join(DATA_DIR, project_id)
        if os.path.isdir(project_dir) and project_id not in st.session_state.projects:
            # Try to find project name
            name_file = os.path.join(project_dir, "project_name.txt")
            if os.path.exists(name_file):
                with open(name_file, 'r') as f:
                    project_name = f.read().strip()
            else:
                project_name = f"Project {project_id}"
            
            # Check for files
            original_audio = None
            cleaned_audio = None
            transcription = None
            cloned_audio = None
            
            for file in os.listdir(project_dir):
                if file == "original_audio.wav":
                    original_audio = os.path.join(project_dir, file)
                elif file == "cleaned_audio.wav":
                    cleaned_audio = os.path.join(project_dir, file)
                elif file == "transcription.txt":
                    transcription = os.path.join(project_dir, file)
                elif file == "cloned_voice.wav":
                    cloned_audio = os.path.join(project_dir, file)
            
            st.session_state.projects[project_id] = {
                "name": project_name,
                "dir": project_dir,
                "original_audio": original_audio,
                "cleaned_audio": cleaned_audio,
                "transcription": transcription,
                "cloned_audio": cloned_audio
            }

# Load existing projects at startup
load_existing_projects()

# Sidebar for project management
with st.sidebar:
    st.title("🎙️ VoiceCraft")
    st.markdown("---")
    
    # Project management
    st.subheader("Project Management")
    
    # Create new project
    new_project_name = st.text_input("New Project Name", key="new_project_name_input")
    if st.button("Create Project", key="create_project_button"):
        if new_project_name:
            project_id = datetime.now().strftime("%Y%m%d%H%M%S")
            project_dir = os.path.join(DATA_DIR, project_id)
            os.makedirs(project_dir, exist_ok=True)
            
            # Save project name to file
            with open(os.path.join(project_dir, "project_name.txt"), 'w') as f:
                f.write(new_project_name)
            
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
            index=0 if st.session_state.current_project_id else None,
            key="project_selector"
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
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"], key="audio_file_uploader")
        
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
            
            # Audio player - REMOVED KEY PARAMETER
            st.audio(original_path, format="audio/wav")
            
            # Noise reduction options
            st.subheader("Noise Reduction")
            apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=True, key="noise_reduction_checkbox")
            
            if st.button("Process Audio", key="process_audio_button"):
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
                    
                    # Audio player for cleaned audio - REMOVED KEY PARAMETER
                    st.subheader("Processed Audio")
                    st.audio(cleaned_path, format="audio/wav")
        
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
            
            # Audio player - REMOVED KEY PARAMETER
            st.audio(project['original_audio'], format="audio/wav")
            
            if project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']):
                # Display cleaned audio waveform
                y_cleaned, sr_cleaned = librosa.load(project['cleaned_audio'], sr=None)
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, len(y_cleaned)/sr_cleaned, len(y_cleaned)), y_cleaned, color='green', alpha=0.7)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Processed Audio Waveform")
                st.pyplot(fig)
                
                # Audio player for cleaned audio - REMOVED KEY PARAMETER
                st.subheader("Processed Audio")
                st.audio(project['cleaned_audio'], format="audio/wav")
    
    # Tab 2: Transcription
    with tabs[1]:
        st.header("Audio Transcription")
        
        if project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']):
            # Model selection
            model_size = st.selectbox(
                "Select Whisper Model Size",
                options=["tiny", "base", "small", "medium", "large"],
                index=1,  # Default to "base"
                key="whisper_model_selector"
            )
            
            if st.button("Transcribe Audio", key="transcribe_button"):
                with st.spinner(f"Loading Whisper {model_size} model and transcribing audio..."):
                    try:
                        # Load model using our safe function
                        model = load_whisper_model_safely(model_size)
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
                        st.text_area("Transcribed Text", transcribed_text, height=200, key="transcribed_text_result")
                        
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
            
            # Display existing transcription if available
            if project.get('transcription') and os.path.exists(project['transcription']):
                with open(project['transcription'], 'r', encoding='utf-8') as f:
                    transcribed_text = f.read()
                
                st.subheader("Existing Transcription")
                st.text_area("Transcribed Text", transcribed_text, height=200, key="existing_transcription_display")
                
                # Allow editing transcription
                edited_text = st.text_area("Edit Transcription", transcribed_text, height=200, key="edit_transcription")
                
                if st.button("Save Edited Transcription", key="save_edited_transcription_button"):
                    with open(project['transcription'], 'w', encoding='utf-8') as f:
                        f.write(edited_text)
                    st.success("Transcription updated!")
        else:
            st.warning("Please process an audio file first in the 'Audio Processing' tab.")
    
    # Tab 3: Voice Cloning
    with tabs[2]:
        st.header("Voice Cloning")
        
        if (project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']) and 
            project.get('transcription') and os.path.exists(project['transcription'])):
            
            # Read the reference text
            with open(project['transcription'], 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            
            # Text to generate
            gen_text = st.text_area(
                "
                "Enter text to generate with the cloned voice",
                value="I am excited to explore new opportunities in the field of machine learning and natural language processing.",
                height=150,
                key="voice_cloning_input"
            )
            
            if st.button("Clone Voice", key="clone_voice_button"):
                with st.spinner("Cloning voice and generating speech... This may take a while."):
                    try:
                        # Import F5 TTS
                        from f5 import F5TTS
                        
                        # Initialize F5 TTS
                        f5_tts = F5TTS()
                        
                        # Clone voice
                        cloned_audio_path = os.path.join(project['dir'], "cloned_voice.wav")
                        
                        # Generate speech with cloned voice
                        f5_tts.tts(
                            text=gen_text,
                            ref_audio_path=project['cleaned_audio'],
                            ref_text=ref_text,
                            output_path=cloned_audio_path
                        )
                        
                        project['cloned_audio'] = cloned_audio_path
                        st.success("Voice cloning completed!")
                        
                        # Display generated audio - REMOVED KEY PARAMETER
                        st.subheader("Generated Speech with Cloned Voice")
                        st.audio(cloned_audio_path, format="audio/wav")
                        
                        # Display audio waveform
                        y, sr = librosa.load(cloned_audio_path, sr=None)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='purple', alpha=0.7)
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        ax.set_title("Generated Audio Waveform")
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"An error occurred during voice cloning: {str(e)}")
            
            # Display existing cloned voice if available
            if project.get('cloned_audio') and os.path.exists(project['cloned_audio']):
                st.subheader("Previously Generated Speech")
                # REMOVED KEY PARAMETER
                st.audio(project['cloned_audio'], format="audio/wav")
                
                # Display audio waveform
                y, sr = librosa.load(project['cloned_audio'], sr=None)
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='purple', alpha=0.7)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Generated Audio Waveform")
                st.pyplot(fig, key="existing_cloned_voice_waveform")
        else:
            st.warning("Please process an audio file and transcribe it first.")
else:
    st.title("Welcome to VoiceCraft")
    st.markdown("""
    ## All-in-one voice processing tool
    
    VoiceCraft allows you to:
    - Clean and reduce noise in audio recordings
    - Transcribe audio to text using OpenAI's Whisper
    - Clone voices to generate new speech with the same voice characteristics
    
    ### Getting Started
    
    1. Create a new project using the sidebar
    2. Upload an audio file
    3. Process the audio (with optional noise reduction)
    4. Transcribe the audio
    5. Clone the voice to generate new speech
    
    ### Features
    
    - **Audio Processing**: Upload audio files (WAV, MP3, M4A, OGG), apply noise reduction, visualize audio waveforms
    - **Transcription**: Transcribe audio using OpenAI's Whisper model, choose from different model sizes
    - **Voice Cloning**: Clone voices using F5-TTS, generate new speech with the cloned voice
    """)
    
    st.info("👈 Create a new project using the sidebar to get started")
