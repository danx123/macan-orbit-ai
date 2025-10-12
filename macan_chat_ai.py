# nama file: macan_chat_ai.py (Upgrade ke Google Gemini & OpenAI API)
import sys
import os
import json
import csv
import requests
from datetime import datetime
import threading
import base64
from PySide6.QtCore import QBuffer

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QMessageBox,
    QInputDialog, QListWidget, QListWidgetItem, QDialog, QTextEdit,
    QMenu, QFileDialog, QMenuBar
)
from PySide6.QtCore import Qt, QObject, QThread, Signal, QUrl, QByteArray, QTimer
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap, QPainter, QAction, QFont
from PySide6.QtSvg import QSvgRenderer

# --- Integrasi API ---
try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
# -----------------------------

import pyttsx3

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Shortcut enum
Expanding = QSizePolicy.Policy.Expanding

# === Path Konfigurasi ===
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "macan_ai_config.json")
LOG_PATH = os.path.join(BASE_PATH, "macan_ai_chatlog.jsonl")

if not os.path.exists(LOG_PATH):
    open(LOG_PATH, 'w').close()

def load_config(path):
    if not os.path.exists(path):
        # --- OPTIMISASI: Mengganti model default OpenAI ke gpt-4o-mini ---
        default_config = {
            "active_api": "gemini",
            "gemini": {
                "api_key": "",
                "model": "gemini-1.5-flash-latest",
                "generation_config": {
                    "temperature": 0.9,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048
                }
            },
            "openai": {
                "api_key": "",
                "model": "gpt-4o-mini"
                }
            }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(path, config_data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

# === SVG Icons (Tidak Berubah) ===
SVG_USER_ICON = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60d060" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
  <circle cx="12" cy="7" r="4"></circle>
</svg>
"""
SVG_BOT_ICON = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60d060" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 12V3"></path>
  <path d="M18 10h-2"></path>
  <path d="M8 10H6"></path>
  <path d="M2 16c0 1.1.9 2 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v8z"></path>
  <path d="M7 21h10"></path>
</svg>
"""

def get_svg_icon(svg_string, size=18):
    renderer = QSvgRenderer(QByteArray(svg_string.encode('utf-8')))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)

# === Workers ===

# --- OPTIMISASI: Worker sekarang mendukung streaming ---
class GeminiWorker(QObject):
    finished = Signal(str) # Sinyal saat seluruh respons selesai
    chunk_received = Signal(str) # Sinyal untuk setiap potongan data
    error = Signal(str)

    def __init__(self, api_key, model_name, generation_config, history, user_prompt_parts):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.generation_config = generation_config
        self.history = history
        self.user_prompt_parts = user_prompt_parts
        self.full_response = ""

    def run(self):
        if not GEMINI_AVAILABLE:
            self.error.emit("Pustaka Google Gemini (google-generativeai) tidak terinstal.\nSilakan jalankan: pip install google-generativeai Pillow")
            return
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            gemini_history = []
            for msg in self.history:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                content = msg.get("content", "")
                gemini_history.append({'role': role, 'parts': [content]})

            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(self.user_prompt_parts, stream=True, generation_config=self.generation_config)
            
            for chunk in response:
                if chunk.text:
                    self.full_response += chunk.text
                    self.chunk_received.emit(chunk.text)
            
            self.finished.emit(self.full_response)

        except Exception as e:
            error_message = f"Error dari Gemini API: {str(e)}"
            if "API key not valid" in str(e):
                error_message = "API Key Google Gemini tidak valid. Mohon periksa kembali."
            elif "location" in str(e) and "is not supported" in str(e):
                 error_message = "Lokasi Anda mungkin tidak didukung oleh API. Coba gunakan VPN."
            self.error.emit(error_message)

class OpenAIWorker(QObject):
    finished = Signal(str)
    chunk_received = Signal(str)
    error = Signal(str)

    def __init__(self, api_key, model_name, history, user_prompt_parts):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.history = history
        self.user_prompt_parts = user_prompt_parts
        self.full_response = ""

    def run(self):
        if not OPENAI_AVAILABLE:
            self.error.emit("Pustaka OpenAI (openai) tidak terinstal.\nSilakan jalankan: pip install openai")
            return
        try:
            client = openai.OpenAI(api_key=self.api_key)
            messages = list(self.history)
            
            openai_prompt_content = []
            for part in self.user_prompt_parts:
                if isinstance(part, str):
                    openai_prompt_content.append({"type": "text", "text": part})
                elif isinstance(part, Image.Image):
                    buffered = QByteArray()
                    buffer = QBuffer(buffered)
                    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
                    part.save(buffer, "PNG")
                    base64_image = buffered.toBase64().data().decode('utf-8')
                    openai_prompt_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })

            messages.append({"role": "user", "content": openai_prompt_content})

            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    self.full_response += content
                    self.chunk_received.emit(content)

            self.finished.emit(self.full_response)

        except Exception as e:
            error_message = f"Error dari OpenAI API: {str(e)}"
            if "Incorrect API key" in str(e):
                error_message = "API Key OpenAI tidak valid. Mohon periksa kembali."
            self.error.emit(error_message)


class SpeechRecognitionWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    def run(self):
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.error.emit("Library 'speech_recognition' tidak terinstal."); return
        r = sr.Recognizer()
        self.status_update.emit("Mendengarkan...")
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            self.status_update.emit("Memproses suara...")
            text = r.recognize_google(audio, language="id-ID")
            self.finished.emit(text)
        except sr.UnknownValueError: self.error.emit("Tidak dapat mengenali ucapan.")
        except sr.RequestError as e: self.error.emit(f"Error layanan Google Speech: {e}")
        except sr.WaitTimeoutError: self.error.emit("Tidak ada suara terdeteksi.")
        except Exception as e: self.error.emit(f"Error pengenalan suara: {e}")
        finally: self.status_update.emit("")

class SearchResultsDialog(QDialog):
    def __init__(self, results_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hasil Pencarian Log")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout(self)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setHtml(results_text)
        layout.addWidget(self.text_area)
        
        close_button = QPushButton("Tutup")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


# === Main AI Chat Application ===
class MacanAIChat(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowIcon(QIcon("robot.ico"))
        self.setGeometry(100, 100, 800, 600)

        self.config = load_config(CONFIG_PATH)
        self.messages = []
        self.last_reply = ""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.pending_media_path = None
        self.pending_media_type = None
        self.current_bot_bubble_label = None # --- OPTIMISASI: Untuk streaming

        self.setup_ui()
        self.update_ui_for_active_api()
        self.load_initial_chat_history()

    def setup_ui(self):
        menu_bar = self.menuBar()
        api_menu = menu_bar.addMenu("API")
        self.switch_to_gemini_action = QAction("Gunakan Google Gemini", self); self.switch_to_gemini_action.setCheckable(True)
        self.switch_to_gemini_action.triggered.connect(lambda: self.switch_api("gemini"))
        api_menu.addAction(self.switch_to_gemini_action)
        self.switch_to_openai_action = QAction("Gunakan OpenAI", self); self.switch_to_openai_action.setCheckable(True)
        self.switch_to_openai_action.triggered.connect(lambda: self.switch_api("openai"))
        api_menu.addAction(self.switch_to_openai_action)
        api_menu.addSeparator()
        set_gemini_key_action = QAction("Set Kunci API Gemini", self); set_gemini_key_action.triggered.connect(lambda: self.set_api_key("gemini"))
        api_menu.addAction(set_gemini_key_action)
        set_openai_key_action = QAction("Set Kunci API OpenAI", self); set_openai_key_action.triggered.connect(lambda: self.set_api_key("openai"))
        api_menu.addAction(set_openai_key_action)

        central_widget = QWidget(); self.setCentralWidget(central_widget)
        top_h_layout = QHBoxLayout(central_widget); top_h_layout.setContentsMargins(0, 0, 0, 0); top_h_layout.setSpacing(0)

        self.history_list_widget = QListWidget(); self.history_list_widget.setFixedWidth(200)
        self.history_list_widget.setStyleSheet("""
            QListWidget { background-color: #2e2e2e; color: #e0e0e0; border-right: 1px solid #444444; font-size: 10pt; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #3a3a3a; }
            QListWidget::item:selected { background-color: #007bff; color: white; }
            QListWidget::item:hover { background-color: #3a3a3a; }
        """)
        self.history_list_widget.itemClicked.connect(self.load_conversation_from_history)
        top_h_layout.addWidget(self.history_list_widget)

        chat_area_widget = QWidget(); chat_area_layout = QVBoxLayout(chat_area_widget)
        chat_area_layout.setContentsMargins(10, 10, 10, 10); chat_area_layout.setSpacing(10)        

        self.scrollArea = QScrollArea(); self.scrollArea.setWidgetResizable(True)
        self.chatContent = QWidget(); self.chatLayout = QVBoxLayout(self.chatContent)
        self.chatLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollArea.setWidget(self.chatContent)
        chat_area_layout.addWidget(self.scrollArea, stretch=1)

        self.loader = QLabel(""); self.loader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loader.setStyleSheet("color: #00FF00; font-style: italic;")
        chat_area_layout.addWidget(self.loader)

        input_row_layout = QHBoxLayout()
        self.inputPrompt = QLineEdit(); self.inputPrompt.returnPressed.connect(self.sendPrompt)
        self.inputPrompt.setStyleSheet("padding: 8px; border-radius: 5px; border: 1px solid #cccccc;")
        input_row_layout.addWidget(self.inputPrompt, stretch=1)        

        input_section = QVBoxLayout(); input_section.addLayout(input_row_layout)
        self.branding_label = QLabel("Built by Danx — WhatsApp: 089626479736")        
        self.branding_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.branding_label.setStyleSheet("color: gray; font-size: 10px; font-style: regular; margin-top: 2px;")
        input_section.addWidget(self.branding_label)     
        chat_area_layout.addLayout(input_section)

        self.sendButton = QPushButton("Kirim"); self.sendButton.clicked.connect(self.sendPrompt)
        self.sendButton.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 8px 15px;")
        input_row_layout.addWidget(self.sendButton)

        self.readButton = QPushButton("Baca"); self.readButton.clicked.connect(self.readReply)
        self.readButton.setStyleSheet("background-color: #007bff; color: white; border-radius: 5px; padding: 8px 15px;")
        input_row_layout.addWidget(self.readButton)
        
        # --- OPTIMISASI: Tombol baru dan reset dengan ikon ---
        button_font = QFont(); button_font.setPointSize(12)

        self.newChatButton = QPushButton("🔄"); self.newChatButton.setFixedSize(36, 36); self.newChatButton.setFont(button_font)
        self.newChatButton.clicked.connect(self.start_new_chat)
        self.newChatButton.setToolTip("Mulai Chat Baru")
        self.newChatButton.setStyleSheet("background-color: #17a2b8; color: white; border-radius: 5px;")
        input_row_layout.addWidget(self.newChatButton)

        self.resetButton = QPushButton("🗑️"); self.resetButton.setFixedSize(36, 36); self.resetButton.setFont(button_font)
        self.resetButton.clicked.connect(self.resetChat)
        self.resetButton.setToolTip("Hapus Semua Riwayat")
        self.resetButton.setStyleSheet("background-color: #dc3545; color: white; border-radius: 5px;")
        input_row_layout.addWidget(self.resetButton)

        self.mic_button = QPushButton("🎤"); self.mic_button.setFixedSize(36, 36)
        self.mic_button.clicked.connect(self.start_speech_recognition)
        self.mic_button.setToolTip("Gunakan suara untuk mengetik prompt.")
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.mic_button.setEnabled(False); self.mic_button.setToolTip("Fitur ini membutuhkan library 'speech_recognition' dan 'PyAudio'.")
        input_row_layout.insertWidget(0, self.mic_button)

        self.addMediaButton = QPushButton("➕"); self.addMediaButton.setFixedSize(36, 36)
        self.addMediaButton.setToolTip("Kirim gambar atau file")
        input_row_layout.insertWidget(0, self.addMediaButton)

        media_menu = QMenu(self)
        send_image_action = QAction("🖼️ Kirim Gambar", self); send_file_action = QAction("📄 Kirim File Teks", self)
        media_menu.addAction(send_image_action); media_menu.addAction(send_file_action)
        self.addMediaButton.setMenu(media_menu)
        send_image_action.triggered.connect(self.handle_send_image)
        send_file_action.triggered.connect(self.handle_send_file)
       
        self.search_log_button = QPushButton("🔍 Cari Log"); self.search_log_button.clicked.connect(self.open_log_search_dialog)
        self.search_log_button.setStyleSheet("background-color: #6c757d; color: white; border-radius: 5px; padding: 5px 10px;")
        chat_area_layout.addWidget(self.search_log_button, alignment=Qt.AlignmentFlag.AlignRight)
       
        top_h_layout.addWidget(chat_area_widget, stretch=1)

    def switch_api(self, api_name):
        self.config['active_api'] = api_name; save_config(CONFIG_PATH, self.config)
        self.update_ui_for_active_api()
        QMessageBox.information(self, "Mode Diganti", f"Sekarang menggunakan API {api_name.capitalize()}.")

    def update_ui_for_active_api(self):
        api_name = self.config.get('active_api', 'gemini')
        if api_name == 'gemini':
            self.setWindowTitle("Macan Orbit AI v4.1 - Powered by Google Gemini")
            self.inputPrompt.setPlaceholderText("Ketik pesan Anda atau kirim gambar...")
            self.switch_to_gemini_action.setChecked(True); self.switch_to_openai_action.setChecked(False)
        else: # openai
            self.setWindowTitle("Macan Orbit AI v4.1 - Powered by OpenAI")
            self.inputPrompt.setPlaceholderText("Ketik pesan Anda atau kirim gambar (model Vision)...")
            self.switch_to_gemini_action.setChecked(False); self.switch_to_openai_action.setChecked(True)

    def set_api_key(self, api_name):
        prompt_text = f"Masukkan {'Google Gemini' if api_name == 'gemini' else 'OpenAI'} API Key Anda:"
        current_key = self.config.get(api_name, {}).get("api_key", "")
        text, ok = QInputDialog.getText(self, f"Input {api_name.capitalize()} API Key", prompt_text, QLineEdit.EchoMode.Password, current_key)
        if ok and text:
            self.config[api_name]['api_key'] = text.strip(); save_config(CONFIG_PATH, self.config)
            QMessageBox.information(self, "Sukses", f"{api_name.capitalize()} API Key berhasil disimpan.")
        else:
            QMessageBox.warning(self, "Batal", "API Key tidak diubah.")
        
    def handle_send_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Image Files (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.pending_media_path = file_path; self.pending_media_type = 'image'
            file_name = os.path.basename(file_path)
            self.inputPrompt.setText(f"[Gambar terlampir: {file_name}]")
            self.inputPrompt.setStyleSheet("padding: 8px; border-radius: 5px; border: 1px solid #cccccc; color: green; font-style: italic;")

    def handle_send_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih File Teks", "", "Text Files (*.txt *.py *.json *.md);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                file_name = os.path.basename(file_path)
                current_prompt = self.inputPrompt.text() if not self.inputPrompt.text().startswith("[Gambar") else ""
                new_prompt = f"{current_prompt}\n\n--- MULAI KONTEN DARI FILE '{file_name}' ---\n{content}\n--- SELESAI KONTEN DARI FILE ---"
                self.inputPrompt.setText(new_prompt)
                self.inputPrompt.setStyleSheet("padding: 8px; border-radius: 5px; border: 1px solid #cccccc;")
                QMessageBox.information(self, "Sukses", f"Konten dari '{file_name}' telah ditambahkan ke prompt.")
            except Exception as e:
                QMessageBox.critical(self, "Error Membaca File", f"Gagal membaca konten file: {e}")

    def addBubble(self, message_obj, is_streaming=False):
        sender = message_obj.get("role")
        content = message_obj.get("content")
        
        text_content = ""; image_pixmap = None

        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text": text_content += part.get("text", "")
                elif part.get("type") == "image_url": 
                    url_data = part.get("image_url", {}).get("url", "")
                    if "base64," in url_data:
                        img_data = base64.b64decode(url_data.split("base64,")[1])
                        image_pixmap = QPixmap(); image_pixmap.loadFromData(img_data)
                elif part.get("type") == "image_path": image_pixmap = QPixmap(part.get("image_path"))
        else:
            text_content = content or ""
        
        bubble_widget = QWidget(); bubble_layout = QHBoxLayout(bubble_widget)
        bubble_layout.setContentsMargins(5, 5, 5, 5); bubble_layout.setSpacing(10)
        max_bubble_width = int(self.scrollArea.viewport().width() * 0.80) if self.scrollArea.viewport().width() > 0 else 550
        bubble_widget.setMaximumWidth(max_bubble_width)
        
        content_widget = QWidget(); content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0,0,0,0)

        if image_pixmap:
            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(image_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            content_layout.addWidget(thumbnail_label)

        message_label = QLabel(text_content); message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if text_content: content_layout.addWidget(message_label)
        
        if sender == "user":
            bubble_widget.setStyleSheet("background-color: #E0E0E0; border-radius: 15px; padding: 10px; margin-bottom: 5px;")
            user_icon_label = QLabel(); user_icon_label.setPixmap(get_svg_icon(SVG_USER_ICON, size=24).pixmap(24,24))
            bubble_layout.addStretch(1); bubble_layout.addWidget(content_widget); bubble_layout.addWidget(user_icon_label)
        elif sender == "assistant":
            bubble_widget.setStyleSheet("background-color: #A0D4E4; border-radius: 15px; padding: 10px; margin-bottom: 5px;")
            bot_icon_label = QLabel(); bot_icon_label.setPixmap(get_svg_icon(SVG_BOT_ICON, size=24).pixmap(24,24))
            bubble_layout.addWidget(bot_icon_label); bubble_layout.addWidget(content_widget); bubble_layout.addStretch(1)
            if is_streaming: self.current_bot_bubble_label = message_label # Simpan referensi untuk streaming
        
        self.chatLayout.addWidget(bubble_widget)
        self.scroll_to_bottom()
        return message_label

    def log_chat(self, message_obj):
        message_obj['conversation_id'] = self.current_conversation_id
        message_obj['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                json.dump(message_obj, f); f.write('\n')
        except Exception as e:
            QMessageBox.warning(self, "Logging Error", f"Gagal menulis ke log: {e}")

    def sendPrompt(self):
        prompt = self.inputPrompt.text().strip()
        if not prompt and not self.pending_media_path: return
        if not self.check_active_api_key(): return

        prompt_text_for_api = prompt if not prompt.startswith("[Gambar terlampir:") else "Jelaskan atau proses gambar yang terlampir."
        user_prompt_parts = [prompt_text_for_api] 
        display_content_parts = [{"type": "text", "text": prompt_text_for_api}]

        if self.pending_media_type == 'image' and self.pending_media_path:
            try:
                img = Image.open(self.pending_media_path)
                user_prompt_parts.append(img)
                display_content_parts.append({"type": "image_path", "image_path": self.pending_media_path})
            except Exception as e:
                QMessageBox.critical(self, "Error Gambar", f"Gagal memproses gambar: {e}"); return

        message_for_ui_and_log = {"role": "user", "content": display_content_parts}
        self.addBubble(message_for_ui_and_log)
        self.log_chat(message_for_ui_and_log)
        
        api_history_content = prompt_text_for_api
        if self.pending_media_path: api_history_content += " [user sent an image]"
        self.messages.append({"role": "user", "content": api_history_content})

        self.inputPrompt.clear(); self.inputPrompt.setStyleSheet("padding: 8px; border-radius: 5px; border: 1px solid #cccccc;")
        self.pending_media_path = None; self.pending_media_type = None
        self.set_ui_enabled(False)
        
        # --- OPTIMISASI: Tambahkan bubble kosong untuk diisi oleh streaming ---
        self.addBubble({"role": "assistant", "content": ""}, is_streaming=True)

        active_api = self.config.get('active_api', 'gemini')
        self.loader.setText(f"{active_api.capitalize()} sedang berpikir...");

        self.thread = QThread()
        if active_api == 'gemini':
            worker = GeminiWorker(self.config["gemini"]["api_key"], self.config["gemini"]["model"], self.config["gemini"].get("generation_config", {}), self.messages[:-1], user_prompt_parts)
        else: # openai
            worker = OpenAIWorker(self.config["openai"]["api_key"], self.config["openai"]["model"], self.messages[:-1], user_prompt_parts)
        
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.chunk_received.connect(self.handle_ai_chunk) # Terhubung ke sinyal streaming
        self.worker.finished.connect(self.handle_ai_reply)
        self.worker.error.connect(self.handle_ai_error)
        self.thread.start()

    # --- OPTIMISASI: Slot baru untuk menangani streaming chunk ---
    def handle_ai_chunk(self, chunk):
        if self.current_bot_bubble_label:
            current_text = self.current_bot_bubble_label.text()
            self.current_bot_bubble_label.setText(current_text + chunk)
            self.scroll_to_bottom()

    def handle_ai_reply(self, full_reply):
        message_obj = {"role": "assistant", "content": full_reply}
        self.log_chat(message_obj)
        
        self.messages.append(message_obj)
        self.last_reply = full_reply
        self.loader.setText(""); self.set_ui_enabled(True)
        self.current_bot_bubble_label = None # Reset referensi bubble
        if hasattr(self, 'thread') and self.thread.isRunning(): self.thread.quit(); self.thread.wait()

    def handle_ai_error(self, error_msg):
        if self.current_bot_bubble_label:
            self.current_bot_bubble_label.setText(f"Error: {error_msg}")
            self.current_bot_bubble_label.setStyleSheet("color: red;")
        self.loader.setText("Error!"); self.set_ui_enabled(True)
        self.current_bot_bubble_label = None
        if hasattr(self, 'thread') and self.thread.isRunning(): self.thread.quit(); self.thread.wait()
        QMessageBox.critical(self, "API Error", error_msg)

    def set_ui_enabled(self, enabled):
        self.inputPrompt.setEnabled(enabled); self.sendButton.setEnabled(enabled)
        self.addMediaButton.setEnabled(enabled); self.resetButton.setEnabled(enabled)
        self.newChatButton.setEnabled(enabled);
        self.mic_button.setEnabled(enabled and SPEECH_RECOGNITION_AVAILABLE)

    def check_active_api_key(self):
        active_api = self.config.get('active_api', 'gemini')
        api_name_display = "Google Gemini" if active_api == "gemini" else "OpenAI"
        
        if (active_api == 'gemini' and not GEMINI_AVAILABLE) or (active_api == 'openai' and not OPENAI_AVAILABLE):
            lib_name = 'google-generativeai Pillow' if active_api == 'gemini' else 'openai'
            QMessageBox.critical(self, "Dependensi Hilang", f"Pustaka untuk {api_name_display} tidak terinstal.\n\n`pip install {lib_name}`")
            return False

        key = self.config.get(active_api, {}).get("api_key", "")
        if key: return True
        self.set_api_key(active_api)
        return bool(self.config.get(active_api, {}).get("api_key", ""))
    
    def readReply(self):
        if self.engine.isBusy(): self.engine.stop()
        elif self.last_reply and not self.last_reply.lower().startswith("error:"):
            threading.Thread(target=self._speak_reply, args=(self.last_reply,)).start()

    def _speak_reply(self, text):
        try:
            self.engine.say(text); self.engine.runAndWait()
        except Exception as e: print(f"Error during text-to-speech: {e}")

    # --- OPTIMISASI: Tombol chat baru ---
    def start_new_chat(self):
        for i in reversed(range(self.chatLayout.count())):
            widget = self.chatLayout.itemAt(i).widget()
            if widget: widget.deleteLater()
        
        self.messages.clear()
        self.last_reply = ""
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.inputPrompt.clear()
        self.loader.setText("Chat baru dimulai.")
        QTimer.singleShot(2000, lambda: self.loader.setText(""))
        
        # Tambahkan item baru ke histori UI
        list_item = QListWidgetItem(f"{datetime.now().strftime('%d/%m %H:%M')} - Chat Baru...")
        list_item.setIcon(get_svg_icon(SVG_USER_ICON))
        list_item.setData(Qt.ItemDataRole.UserRole, self.current_conversation_id)
        self.history_list_widget.insertItem(0, list_item) # Tambah di paling atas
        self.history_list_widget.setCurrentRow(0)
            
    def resetChat(self):
        confirm = QMessageBox.question(self, "Konfirmasi Hapus Total", "Yakin ingin menghapus SEMUA riwayat percakapan secara permanen? Tindakan ini tidak bisa dibatalkan.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.start_new_chat() # Memulai dengan membersihkan UI
            try:
                if os.path.exists(LOG_PATH): os.remove(LOG_PATH)
                open(LOG_PATH, 'w').close()
                self.history_list_widget.clear()
                QMessageBox.information(self, "Sukses", "Semua riwayat percakapan telah dihapus.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menghapus file log: {e}")
            self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f") # Pastikan ID baru

    def load_initial_chat_history(self):
        self.history_list_widget.clear()
        conversations = {}
        if not os.path.exists(LOG_PATH): return

        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        conv_id = log_entry.get("conversation_id")
                        if conv_id not in conversations: conversations[conv_id] = []
                        conversations[conv_id].append(log_entry)
                    except json.JSONDecodeError: continue
        except IOError as e:
            QMessageBox.warning(self, "Load History Error", f"Gagal memuat riwayat: {e}"); return
        
        sorted_conv_ids = sorted(conversations.keys(), key=lambda k: conversations[k][0].get('timestamp', '0'), reverse=True) # Urutkan dari terbaru

        for conv_id in sorted_conv_ids:
            first_msg = conversations[conv_id][0]
            first_content = first_msg.get("content")
            display_text = ""
            if isinstance(first_content, list):
                display_text = next((p.get("text", "") for p in first_content if p.get("type") == "text"), "[Gambar]")
            else: display_text = first_content or ""
            
            ts_str = first_msg.get('timestamp', '1970-01-01 00:00:00')
            try: ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').strftime('%d/%m %H:%M')
            except ValueError: ts = "N/A"

            list_item = QListWidgetItem(f"{ts} - {display_text[:30]}...")
            list_item.setIcon(get_svg_icon(SVG_USER_ICON))
            list_item.setData(Qt.ItemDataRole.UserRole, conv_id)
            self.history_list_widget.addItem(list_item)
        
    def load_conversation_from_history(self, item):
        conv_id_to_load = item.data(Qt.ItemDataRole.UserRole)
        # --- OPTIMISASI: Jangan load ulang jika chat yang sama sudah aktif ---
        if conv_id_to_load == self.current_conversation_id: return
        
        self.inputPrompt.clear() # Kosongkan input saat ganti chat
        for i in reversed(range(self.chatLayout.count())):
            widget = self.chatLayout.itemAt(i).widget()
            if widget: widget.deleteLater()
        
        self.messages.clear()
        self.current_conversation_id = conv_id_to_load
        
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if log_entry.get("conversation_id") == conv_id_to_load:
                            self.addBubble(log_entry)
                            
                            # --- OPTIMISASI: Logika direvisi agar lebih jelas ---
                            # Siapkan histori untuk API (selalu format teks sederhana)
                            content = log_entry.get("content", "")
                            role = log_entry.get("role")
                            api_history_content = ""
                            if isinstance(content, list):
                                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                                api_history_content = " ".join(text_parts)
                                if any(p.get("type") in ["image_path", "image_url"] for p in content):
                                    api_history_content += f" [{role} sent an image]"
                            else:
                                api_history_content = content or ""
                            
                            self.messages.append({"role": role, "content": api_history_content.strip()})
                    except json.JSONDecodeError: continue
        except Exception as e:
            QMessageBox.warning(self, "Load Conversation Error", f"Gagal memuat percakapan: {e}")
            
    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum()))

    def start_speech_recognition(self):
        self.set_ui_enabled(False); self.inputPrompt.setText("Mendengarkan...")
        self.speech_thread = QThread(); self.speech_worker = SpeechRecognitionWorker()
        self.speech_worker.moveToThread(self.speech_thread)
        self.speech_thread.started.connect(self.speech_worker.run)
        self.speech_worker.finished.connect(self.handle_speech_result)
        self.speech_worker.error.connect(self.handle_speech_error)
        self.speech_worker.status_update.connect(self.update_speech_status)
        self.speech_thread.start()

    def handle_speech_result(self, text):
        self.inputPrompt.setText(text); self.loader.setText(""); self.set_ui_enabled(True)
        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            self.speech_thread.quit(); self.speech_thread.wait()

    def handle_speech_error(self, error_msg):
        self.inputPrompt.setText(""); self.loader.setText(""); self.set_ui_enabled(True)
        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            self.speech_thread.quit(); self.speech_thread.wait()
        QMessageBox.warning(self, "Pengenalan Suara", error_msg)
    
    def update_speech_status(self, status_text): self.loader.setText(status_text)
        
    def open_log_search_dialog(self):
        keyword, ok = QInputDialog.getText(self, "Cari Log Chat", "Masukkan kata kunci:")
        if ok and keyword: self.perform_log_search(keyword)

    def perform_log_search(self, keyword):
        if not os.path.exists(LOG_PATH):
            QMessageBox.information(self, "Cari Log", "File log chat belum ada."); return
        results = []
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line); text_content = ""
                        content = log_entry.get("content", "")
                        if isinstance(content, list):
                            text_content = next((p.get("text", "") for p in content if p.get("type") == "text"), "")
                        else: text_content = content or ""
                        
                        if keyword.lower() in text_content.lower():
                            timestamp = log_entry.get("timestamp", "N/A"); sender = log_entry.get("role", "N/A").capitalize()
                            conv_id_full = log_entry.get("conversation_id", "N/A")
                            conv_id = conv_id_full[-6:] if len(conv_id_full) > 6 else conv_id_full
                            results.append(f"<b>[{timestamp}] ({conv_id}) {sender}:</b> {text_content}<br>")
                    except json.JSONDecodeError: continue
        except Exception as e:
            QMessageBox.critical(self, "Error Membaca Log", f"Gagal membaca file log JSONL: {e}"); return
        if results:
            results_text = "<h3>Hasil Pencarian:</h3>" + "".join(results)
        else:
            results_text = f"<p>Tidak ditemukan hasil untuk kata kunci '<b>{keyword}</b>'.</p>"
        dialog = SearchResultsDialog(results_text, self)
        dialog.exec()

# === Entrypoint ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_app = MacanAIChat()
    chat_app.show()
    sys.exit(app.exec())