# 🧠 Macan Chat AI — Cross-API Intelligent Chat System

Macan Chat AI is a professional AI chat application based on PySide6 with direct integration with two major global APIs:
- Google Gemini (Generative AI)
- OpenAI (GPT-4o & DALL·E 3)

Designed and developed by Danx Exodus — under the Macan Angkasa Independent Technology Ecosystem — this application delivers a cutting-edge AI conversational and content generation experience with real-time performance, a modern UI, and modular expansion capabilities.

---

## 🚀 Key Features

### 💬 Dual AI Engine
- Google Gemini — supports both the `gemini-1.5-flash-latest` and `gemini-1.5-pro-latest` models for text and images.
- **OpenAI GPT** – compatible with the `gpt-4o-mini` and *DALL·E 3* models for image generation.

### 🧩 Advanced Features
- Chat streaming with real-time display.
- Live image generation with the `/image <description>` command.
- Voice recognition (speech-to-text) support based on `speech_recognition`.
- Text-to-Speech (TTS)** using `pyttsx3`.
- Support for sending files and images directly to the AI.
- Automatic conversation history (JSONL log) storage system.
- Built-in chat log search functionality.
- Modern UI based on **PySide6/Qt6** with SVG icon integration.
- Dual mode (Gemini/OpenAI)** can be switched without restarting.

---

## 📸 Screenshot
<img width="1080" height="1207" alt="macan-orbit-ai-v450" src="https://github.com/user-attachments/assets/fedf92bf-dbb8-40d9-a668-a59f1e3a970b" />


---
## 📜 Changelog:
- Update Framework
---

## ⚙️ Preparation and Installation

### 1. Requirements
Ensure Python 3.10+ is installed, then run:

```bash
pip install PySide6 google-generativeai openai Pillow pyttsx3 speechrecognition

2. Running the Application
python macan_chat_ai.py

The application will automatically create a configuration file:
macan_ai_config.json
macan_ai_chatlog.jsonl
generated_images/

🔑 API Configuration
Access the API menu → Set Gemini/OpenAI API Key to save your API key. Example configuration format:
{
"active_api": "gemini",
"gemini": {
"api_key": "GEMINI_API_KEY",
"model": "gemini-1.5-flash-latest",
"image_model": "gemini-1.5-pro-latest"
},
"openai": {
"api_key": "OPENAI_API_KEY",
"model": "gpt-4o-mini"
}
}

🧠 Custom Commands

Command
Description
/image <description>
Creates an image based on a text description
🎤
Enables speech recognition
🗑️
Deletes all conversation history
🔄
Starts a new conversation
🔍 Search Log
Searches for keywords across all chat logs

🪄 Technology and Architecture
Framework: PySide6 (Qt for Python)
AI Engine: Google Gemini & OpenAI API
Speech Engine: pyttsx3 & Speech Recognition
Image Engine: Pillow (PIL)
Logging System: JSONL incremental logging
Every AI interaction runs through a QThread-based worker to keep the UI responsive and hang-free.

🧰 Important Directories & Files
📁 macan_chat_ai/
├── macan_chat_ai.py # Main application
├── macan_ai_config.json # API configuration
├── macan_ai_chatlog.jsonl # Conversation log
├── generated_images/ # AI image output

🦁 About Macan Angkasa
Macan Angkasa Independent Technology Ecosystem
is a technology ecosystem based in Bandung, Indonesia,
focusing on cross-platform software development, AI, and automation systems.
“Understanding Before Transforming — Trace Before Replace.”

📜 License
MIT License
© 2025 Macan Angkasa Independent Technology Ecosystem
