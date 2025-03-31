import streamlit as st
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import whisper
import torch
import tempfile
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime

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
                st.text_area("Transcribed Text", transcribed_text, height=200)
                
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
        
        if (project.get('cleaned_audio') and os.path.exists(project['cleaned_audio']) and 
            project.get('transcription') and os.path.exists(project['transcription'])):
            
            # Read the reference text
            with open(project['transcription'], 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            
            # Text to generate
            gen_text = st.text_area(
                "Enter text to generate with the cloned voice",
                value="I am excited to explore new opportunities in the field of machine learning and natural language processing.",
                height=150
            )
            
            if st.button("Clone Voice"):
                with st.spinner("Running voice cloning process..."):
                    try:
                        # Set output path
                        output_path = os.path.join(project['dir'], "cloned_voice.wav")
                        
                        # Run F5-TTS CLI
                        cmd = [
                            "f5-tts_infer-cli",
                            "--model", "F5TTS_v1_Base",
                            "--ref_audio", project['cleaned_audio'],
                            "--ref_text", ref_text,
                            "--gen_text", gen_text,
                            "--output_file", output_path,
                            "--device", "cuda" if torch.cuda.is_available() else "cpu"
                        ]
                        
                        process = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if process.returncode != 0:
                            st.error(f"Voice cloning failed: {process.stderr}")
                        else:
                            project['cloned_audio'] = output_path
                            st.success("Voice cloning completed!")
                            
                            # Display cloned audio
                            st.subheader("Cloned Voice")
                            st.audio(output_path)
                    
                    except Exception as e:
                        st.error(f"An error occurred during voice cloning: {str(e)}")
            
            # Display existing cloned audio if available
            if project.get('cloned_audio') and os.path.exists(project['cloned_audio']):
                st.subheader("Previously Cloned Voice")
                st.audio(project['cloned_audio'])
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
