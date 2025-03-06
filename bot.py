import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import pyaudio
import wave
import numpy as np
import whisper
import base64
import PIL.Image
import io
from groq import Groq

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY in the .env file.")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY in the .env file.")

# Configure the Gemini API
genai.configure(api_key=API_KEY)

# Available models
MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash Thinking": "gemini-2.0-flash-thinking-exp-1219"
}

# Global variables for audio recording and media
audio_frames = []
is_recording = False
last_uploaded_file = None
last_image_file = None

# Image encoding function
def encode_image(image_path):   
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image analysis function with Groq
def analyze_image_with_query(query, model, encoded_image):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

# Media processing and analysis function
def process_media(file, webcam_image=None, analysis_query=None):
    global last_uploaded_file, last_image_file
    
    # Check webcam image first
    if webcam_image is not None:
        try:
            # Save the image
            filename = f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            webcam_image.save(filename)
            last_uploaded_file = filename
            last_image_file = filename
            
            # If analysis query is provided, perform image analysis
            if analysis_query:
                try:
                    encoded_image = encode_image(filename)
                    analysis_result = analyze_image_with_query(
                        analysis_query, 
                        "llama3-70b-8192",  
                        encoded_image
                    )
                    return f"Webcam image captured and analyzed: {analysis_result}"
                except Exception as e:
                    return f"Image analysis error: {str(e)}"
            
            return f"Webcam image captured: {filename}"
        except Exception as e:
            return f"Webcam image processing error: {str(e)}"
    
    # Process uploaded file
    if file is None:
        return "No file uploaded."
    
    try:
        filename = file.name
        last_uploaded_file = filename
        last_image_file = filename
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Image handling
        if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            # If analysis query is provided, perform image analysis
            if analysis_query:
                try:
                    encoded_image = encode_image(filename)
                    analysis_result = analyze_image_with_query(
                        analysis_query, 
                        "llama3-70b-8192",  
                        encoded_image
                    )
                    return f"Image uploaded and analyzed: {analysis_result}"
                except Exception as e:
                    return f"Image analysis error: {str(e)}"
            
            return f"Image uploaded: {os.path.basename(filename)}"
        
        # Other file type handling
        elif file_extension in ['.pdf', '.docx', '.txt']:
            return f"Document uploaded: {os.path.basename(filename)}"
        
        elif file_extension in ['.mp3', '.wav', '.ogg']:
            return f"Audio file uploaded: {os.path.basename(filename)}"
        
        elif file_extension in ['.mp4', '.avi', '.mov']:
            return f"Video file uploaded: {os.path.basename(filename)}"
        
        else:
            return f"Unsupported file type: {file_extension}"
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Enhanced chatbot function to handle media
def enhanced_chatbot(prompt, chat_history, model_name, temperature, conversation_name):
    global last_uploaded_file, last_image_file
    
    if not prompt.strip() and not last_uploaded_file:
        return [], chat_history, conversation_name
    
    try:
        # Update conversation name if it's the default and this is the first message
        if conversation_name == "New Conversation" and not chat_history:
            conversation_name = prompt[:30] + "..." if len(prompt) > 30 else prompt
        
        model = genai.GenerativeModel(MODELS[model_name], generation_config={
            "temperature": temperature
        })
        
        # Create chat session with history
        chat = model.start_chat(history=[
            {"role": "user" if i % 2 == 0 else "model", "parts": [msg]}
            for i, msg in enumerate([item for sublist in chat_history for item in sublist]) 
            if msg  # Skip empty messages
        ])
        
        # Handle media file if present
        if last_image_file:
            # Add media context to the prompt
            media_context = f"I've uploaded an image file: {last_image_file}. "
            prompt = media_context + prompt
            last_image_file = None  # Reset after use
        
        if last_uploaded_file:
            # Add media context to the prompt
            media_context = f"I've uploaded a file: {last_uploaded_file}. "
            prompt = media_context + prompt
            last_uploaded_file = None  # Reset after use
        
        # Get response
        response = chat.send_message(prompt)
        
        # Create messages in the new format
        new_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.text}
        ]
        
        chat_history.append(new_messages)
        
        return new_messages, chat_history, conversation_name
    except Exception as e:
        error_message = f"Error: {str(e)}"
        new_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": error_message}
        ]
        chat_history.append(new_messages)
        return new_messages, chat_history, conversation_name

# Audio Recording Functions
def start_recording():
    global audio_frames, is_recording
    audio_frames = []
    is_recording = True
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    def record_audio():
        global audio_frames, is_recording
        while is_recording:
            data = stream.read(CHUNK)
            audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    import threading
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    return "Recording started..."

def stop_recording():
    global is_recording, audio_frames, last_uploaded_file
    is_recording = False
    
    WAVE_OUTPUT_FILENAME = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    
    # Set as last uploaded file for context
    last_uploaded_file = WAVE_OUTPUT_FILENAME
    
    # Transcribe using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(WAVE_OUTPUT_FILENAME)
    
    return result['text']

# Create Gradio Interface
def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css="""
        .container {
            max-width: 1200px;
            margin: auto;
        }
        .input-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .media-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }
    """) as gui:
        
        # Header
        with gr.Row(elem_classes=["header"]):
            gr.Markdown("""# 🤖 SERENA AI Assistant
            Multimodal Conversational AI
            """)
        
        # Main layout
        with gr.Row(elem_classes=["container"]):
            # Left sidebar for controls
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["control-panel"]):
                    gr.Markdown("### Model Settings")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Gemini 2.0 Flash Thinking",
                        label="Select AI Model"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (Creativity)",
                        info="Higher values make output more random"
                    )
            
            # Right side for chat
            with gr.Column(scale=3):
                chat_state = gr.State([])
                conversation_name_state = gr.State("New Conversation")
                chat_display = gr.Chatbot(
                    height=550,
                    bubble_full_width=False,
                    type="messages"  # Updated to use messages type
                )
                
                # Media status and transcription
                media_status = gr.Textbox(label="Media/Recording Status", interactive=False)
                audio_transcript = gr.Textbox(label="Speech Transcript", interactive=False)
                
                # Integrated input row
                with gr.Row(elem_classes=["input-row"]):
                    # Media upload with webcam
                    with gr.Column(scale=1, elem_classes=["media-input"]):
                        webcam_input = gr.Image(
                            type="pil",  # Changed to PIL 
                            label="Capture Photo",
                            width=100,
                            height=100
                        )
                        file_input = gr.File(
                            file_types=['image', 'video', 'audio', 'text'],
                            label="Upload File",
                            type="file"
                        )
                        analysis_query = gr.Textbox(
                            placeholder="Optional image analysis query...",
                            label="Image Analysis Query"
                        )
                    
                    # Text and speech input
                    with gr.Column(scale=3):
                        user_input = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            elem_classes=["smaller-textbox"]
                        )
                    
                    # Action buttons
                    with gr.Column(scale=1):
                        start_recording_btn = gr.Button("🎤 Record")
                        send_button = gr.Button("Send", variant="primary")
        
        # Event Handlers
        # Media Upload Handlers
        webcam_input.change(
            process_media, 
            inputs=[file_input, webcam_input, analysis_query], 
            outputs=[media_status]
        )
        
        file_input.upload(
            process_media, 
            inputs=[file_input, webcam_input, analysis_query], 
            outputs=[media_status]
        )
        
        # Recording Handlers
        start_recording_btn.click(
            start_recording, 
            outputs=[media_status]
        ).then(
            stop_recording, 
            outputs=[audio_transcript, media_status]
        )
        
        # Message Sending Handlers
        send_button.click(
            enhanced_chatbot,
            inputs=[user_input, chat_state, model_dropdown, temperature_slider, conversation_name_state],
            outputs=[chat_display, chat_state, conversation_name_state]
        ).then(
            lambda: "",
            None,
            user_input
        )
        
        user_input.submit(
            enhanced_chatbot,
            inputs=[user_input, chat_state, model_dropdown, temperature_slider, conversation_name_state],
            outputs=[chat_display, chat_state, conversation_name_state]
        ).then(
            lambda: "",
            None,
            user_input
        )
        
        return gui

# Launch the application
if __name__ == "__main__":
    gui = create_gradio_interface()
    gui.launch(share=True)