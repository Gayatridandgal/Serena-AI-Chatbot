import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from datetime import datetime
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY in the .env file.")

# Configure the Gemini API
genai.configure(api_key=API_KEY)

# Available models
MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash Thinking": "gemini-2.0-flash-thinking-exp-1219"
}

# Chat history management
def load_chat_history():
    try:
        with open("chat_history.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"conversations": {}}

def save_chat_history(history, conversation_name):
    all_history = load_chat_history()
    all_history["conversations"][conversation_name] = history
    with open("chat_history.json", "w") as file:
        json.dump(all_history, file, indent=2)
    return f"Chat '{conversation_name}' saved!"

def get_saved_conversations():
    history = load_chat_history()
    return list(history["conversations"].keys())

# Chatbot function
def chatbot(prompt, chat_history, model_name, temperature, conversation_name):
    if not prompt.strip():
        return chat_history, chat_history, conversation_name
    
    try:
        # Update conversation name if it's the default and this is the first message
        if conversation_name == "New Conversation" and not chat_history:
            # Create a name based on the first message (limited to 30 chars)
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
        
        # Get response
        response = chat.send_message(prompt)
        chat_history.append([prompt, response.text])
        
        return chat_history, chat_history, conversation_name
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history.append([prompt, error_message])
        return chat_history, chat_history, conversation_name

# Speech-to-Text function
def record_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            yield "Listening... (speak now)"
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        yield "Processing speech..."
        text = recognizer.recognize_google(audio)
        yield text
    except sr.UnknownValueError:
        yield "Could not understand audio. Please try again."
    except sr.RequestError:
        yield "Speech recognition service unavailable. Check your internet connection."
    except Exception as e:
        yield f"Error: {str(e)}"

# Load a saved conversation
def load_conversation(conversation_name):
    if not conversation_name:
        return [], []
    
    all_history = load_chat_history()
    if conversation_name in all_history["conversations"]:
        return all_history["conversations"][conversation_name], all_history["conversations"][conversation_name]
    return [], []

# Clear chat function
def clear_chat():
    return [], [], "New Conversation"

# Export chat as markdown
def export_chat_markdown(chat_history, conversation_name):
    if not chat_history:
        return "Chat is empty."
    
    filename = f"{conversation_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# {conversation_name}\n\n")
        file.write(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        for user_msg, bot_msg in chat_history:
            file.write(f"## User\n{user_msg}\n\n")
            file.write(f"## Gemini\n{bot_msg}\n\n")
            file.write("---\n\n")
    
    return f"Chat exported to {filename}"

# UI Design with modern theme
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .container {
<<<<<<< HEAD
        max-width: 1200px; /* Reduced max-width for better responsiveness */
=======
        max-width: 1400px;
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
        margin: auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
<<<<<<< HEAD
        padding: 10px; /* Added padding */
        background-color: #f8f9fa; /* Light background */
        border-radius: 10px;
    }
    .chat-container {
        min-height: 400px; /* Increased chat height */
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
=======
    }
    .chat-container {
        min-height: 300px;
        border-radius: 20px;
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
    }
    .footer {
        text-align: center;
        margin-top: 20px;
<<<<<<< HEAD
        padding: 10px; /* Added padding */
        font-size: 0.9em; /* Slightly larger font */
        color: #888;
    }
    .control-panel {
        padding: 15px; /* Increased padding */
        border-radius: 15px; /* More rounded corners */
        background-color: rgba(0,0,0,0.03); /* Lighter background */
        margin-bottom: 15px; /* Spacing between control panels */
    }
    .gr-button {
        border-radius: 8px !important; /* Rounded buttons */
        padding: 8px 16px !important; /* Button padding */
    }
    .gr-chat-bubble {
        border-radius: 12px !important; /* Rounded chat bubbles */
        padding: 10px !important; /* Chat bubble padding */
    }
    #status {
        margin-top: 10px; /* Space above status textbox */
    }
    .dark .gr-chat-bubble {
        background-color: #343a40 !important; /* Dark mode chat bubble color */
        color: #fff !important;
    }

    .smaller-textbox textarea {
        height: 50px !important; /* Adjust the height value as needed */
    }

=======
        font-size: 0.8em;
        color: #888;
    }
    .control-panel {
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(0,0,0,0.05);
    }
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
""") as gui:
    
    # Header
    with gr.Row(elem_classes=["header"]):
<<<<<<< HEAD
        gr.Markdown("""# 🤖 SERENA AI Assistant
            Multimodal Conversational AI
=======
        gr.Markdown("""# 🤖 AI Assistant
        A powerful chatbot interface AI models
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
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
            
            with gr.Group(elem_classes=["control-panel"]):
                gr.Markdown("### Conversation Management")
                conversation_name = gr.Textbox(
                    label="Conversation Name",
                    placeholder="Name your conversation",
                    value="New Conversation"
                )
                save_button = gr.Button("💾 Save Conversation", variant="primary")
                
                saved_conversations = gr.Dropdown(
                    choices=get_saved_conversations(),
                    label="Load Saved Conversation",
                    interactive=True
                )
                load_button = gr.Button("📂 Load")
                refresh_button = gr.Button("🔄 Refresh List")
                
                export_button = gr.Button("📤 Export as Markdown")
                clear_button = gr.Button("🗑️ Clear Chat", variant="stop")
        
        # Right side for chat
        with gr.Column(scale=3, elem_classes=["chat-container"]):
            chat_state = gr.State([])
            chat_display = gr.Chatbot(
<<<<<<< HEAD
                height=550,  # slightly increase height
=======
                height=500,
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
                bubble_full_width=False,
                show_label=False
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    show_label=False,
<<<<<<< HEAD
                    scale=5,
                    elem_classes=["smaller-textbox"] #apply the smaller-textbox CSS class

=======
                    scale=5
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
                )
                speech_button = gr.Button("🎤", scale=1)
            
            with gr.Row():
                send_button = gr.Button("Send Message", variant="primary", scale=5)
<<<<<<< HEAD
                status = gr.Textbox(label="Status", visible=False, elem_id="status")
=======
                status = gr.Textbox(label="Status", visible=False)
>>>>>>> d4a2909c (Initial commit: Added AI chatbot with Google Gemini API)
    
    # Footer
    with gr.Row(elem_classes=["footer"]):
        gr.Markdown("Created with Gradio and Google Generative AI")
    
    # Event handlers
    send_button.click(
        chatbot,
        inputs=[user_input, chat_state, model_dropdown, temperature_slider, conversation_name],
        outputs=[chat_display, chat_state, conversation_name]
    ).then(
        lambda: "",
        None,
        user_input
    )
    
    # Also send on Enter key
    user_input.submit(
        chatbot,
        inputs=[user_input, chat_state, model_dropdown, temperature_slider, conversation_name],
        outputs=[chat_display, chat_state, conversation_name]
    ).then(
        lambda: "",
        None,
        user_input
    )
    
    # Speech recognition
    speech_button.click(
        record_speech,
        outputs=[user_input],
        api_name="speech_to_text"
    )
    
    # Save, load, and clear functionality
    save_button.click(
        save_chat_history,
        inputs=[chat_state, conversation_name],
        outputs=[status]
    )
    
    refresh_button.click(
        get_saved_conversations,
        outputs=[saved_conversations]
    )
    
    load_button.click(
        load_conversation,
        inputs=[saved_conversations],
        outputs=[chat_display, chat_state]
    )
    
    clear_button.click(
        clear_chat,
        outputs=[chat_display, chat_state, conversation_name]
    )
    
    export_button.click(
        export_chat_markdown,
        inputs=[chat_state, conversation_name],
        outputs=[status]
    )

# Launch the application
if __name__ == "__main__":
    gui.launch(share=True)