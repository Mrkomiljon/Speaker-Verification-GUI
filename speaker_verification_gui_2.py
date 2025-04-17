import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import torchaudio
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tempfile
from pydub import AudioSegment
import sounddevice as sd
import scipy.io.wavfile as wav

REF_DIR = "./ref_voices"
TEST_DIR = "./test_voices"
EMBED_FILE = "speaker_vectors.pkl"
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

class SpeakerVerifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("🎤 Speaker Verification")
        master.configure(bg="#f0f2f5")

        self.speakers = self.load_embeddings()
        self.last_scores = []

        self.model_var = tk.StringVar(value="speechbrain")
        self.threshold = tk.DoubleVar(value=0.75)

        style = ttk.Style()
        style.theme_use('clam')

        button_frame = tk.Frame(master, bg="#f0f2f5")
        button_frame.grid(row=0, column=0, padx=15, pady=10)

        ttk.Label(button_frame, text="Model:").grid(row=0, column=0, sticky="e")
        self.model_selector = ttk.Combobox(button_frame, textvariable=self.model_var, values=["speechbrain"])
        self.model_selector.grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="📁 Register Voice", width=20, command=self.register_voice).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="🧠 Identify Voice", width=20, command=self.identify_voice).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="📂 Auto-Register ref_voices", width=25, command=self.auto_register_ref_voices).grid(row=0, column=4, padx=5, pady=5)

        ttk.Button(button_frame, text="🎙 Record & Register", width=20, command=self.record_and_register).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="🎙 Record & Identify", width=20, command=self.record_and_identify).grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="🧹 Clear Log", width=20, command=self.clear_log).grid(row=1, column=4, padx=5, pady=5)
        ttk.Button(button_frame, text="❌ Delete Speaker", width=20, command=self.delete_speaker).grid(row=1, column=0, padx=5, pady=5)

        tk.Scale(button_frame, from_=0.3, to=0.95, resolution=0.01,
                 label="Threshold", orient="horizontal", variable=self.threshold).grid(row=1, column=1)

        self.log = tk.Text(master, height=18, width=85, bg="white", fg="#222", font=("Consolas", 10), wrap=tk.WORD)
        self.log.grid(row=1, column=0, padx=15, pady=5)
        scrollbar = tk.Scrollbar(master, command=self.log.yview)
        scrollbar.grid(row=1, column=1, sticky='ns', pady=5)
        self.log['yscrollcommand'] = scrollbar.set

        self.log_msg("🔷 Welcome to Speaker Verification!")
        self.init_model()
        self.auto_register_ref_voices()

    def init_model(self):
        model_type = self.model_var.get()
        if model_type == "speechbrain":
            self.verifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb")

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        with open("speaker_gui.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def clear_log(self):
        self.log.delete(1.0, tk.END)
        self.log_msg("🧹 Log cleared.")

    def delete_speaker(self):
        if not self.speakers:
            messagebox.showinfo("Delete Speaker", "No registered speakers.")
            return

        user_id = simpledialog.askstring("Delete Speaker", "Enter User ID to delete:")
        if not user_id:
            return

        if user_id in self.speakers:
            del self.speakers[user_id]
            self.save_embeddings()
            self.log_msg(f"🗑️ Embedding deleted for: {user_id}")
        else:
            self.log_msg(f"⚠️ No embedding found for: {user_id}")

        audio_file = os.path.join(REF_DIR, f"{user_id}.wav")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                self.log_msg(f"🗑️ Audio file deleted: {audio_file}")
            except Exception as e:
                self.log_msg(f"❌ Error deleting file: {e}")
        else:
            self.log_msg("⚠️ No audio file found.")

    def load_embeddings(self):
        if os.path.exists(EMBED_FILE):
            with open(EMBED_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def save_embeddings(self):
        with open(EMBED_FILE, "wb") as f:
            pickle.dump(self.speakers, f)

    def extract_embedding(self, audio_path):
        ext = os.path.splitext(audio_path)[1].lower()
        temp_wav = None

        try:
            if ext == ".mp3":
                sound = AudioSegment.from_file(audio_path, format="mp3")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_wav = tmp.name
                    sound.export(temp_wav, format="wav")
                    audio_path = temp_wav

            signal, fs = torchaudio.load(audio_path)
            if signal.shape[1] < 32000:
                self.log_msg("⚠️ Ovoz juda qisqa, kamida 2 soniya kerak.")
                return None
            if fs != 16000:
                signal = torchaudio.functional.resample(signal, fs, 16000)

            emb = self.verifier.encode_batch(signal).squeeze(0).detach().cpu().numpy()
        except Exception as e:
            self.log_msg(f"❌ Error extracting embedding: {e}")
            emb = None
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

        return emb

    def auto_register_ref_voices(self):
        count = 0
        for fname in os.listdir(REF_DIR):
            if fname.endswith((".wav", ".mp3")):
                user_id = os.path.splitext(fname)[0]
                filepath = os.path.join(REF_DIR, fname)
                emb = self.extract_embedding(filepath)
                if emb is not None:
                    self.speakers[user_id] = emb
                    self.log_msg(f"✅ Auto-registered: {user_id}")
                    count += 1
                else:
                    self.log_msg(f"❌ Failed to register: {user_id}")
        self.save_embeddings()
        self.log_msg(f"📦 Total auto-registered speakers: {count}")

    def register_voice(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if not file:
            return
        user_id = simpledialog.askstring("Register", "Enter User ID:")
        if not user_id:
            return
        emb = self.extract_embedding(file)
        if emb is not None:
            self.speakers[user_id] = emb
            self.save_embeddings()
            dest = os.path.join(REF_DIR, f"{user_id}.wav")
            try:
                AudioSegment.from_file(file).export(dest, format="wav")
            except Exception as e:
                self.log_msg(f"⚠️ Save error: {e}")
            self.log_msg(f"✅ Registered speaker: {user_id}")
        else:
            messagebox.showerror("Embedding Error", "Failed to extract speaker embedding.")

    def identify_voice(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if not file:
            return
        emb = self.extract_embedding(file)
        if emb is None or not self.speakers:
            self.log_msg("❌ No embedding or no speakers.")
            return
        sims = {user: cosine_similarity(emb.reshape(1, -1), ref.reshape(1, -1))[0][0] for user, ref in self.speakers.items()}
        best = max(sims.items(), key=lambda x: x[1])
        score = best[1]
        self.last_scores.append(f"{best[0]}: {score:.2f}")
        if score >= self.threshold.get():
            self.log_msg(f"✅ Speaker matched: {best[0]} (Score: {score:.2f})")
        else:
            self.log_msg(f"❌ Unknown speaker (Best score: {score:.2f})")

    def record_audio(self, filename, duration=5):
        fs = 16000
        self.log_msg(f"🎙 Recording {duration}s...")

        if os.path.exists(filename):
            try:
                os.remove(filename)
                self.log_msg("🧹 Old file removed.")
            except Exception as e:
                self.log_msg(f"⚠️ Could not remove old file: {e}")
                return

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        try:
            wav.write(filename, fs, recording)
            self.log_msg(f"📁 Saved to {filename}")
        except Exception as e:
            self.log_msg(f"❌ Failed to save file: {e}")

    def record_and_register(self):
        user_id = simpledialog.askstring("Record & Register", "Enter User ID:")
        if not user_id:
            return
        filepath = os.path.join(REF_DIR, f"{user_id}.wav")
        self.record_audio(filepath)
        emb = self.extract_embedding(filepath)
        if emb is not None:
            self.speakers[user_id] = emb
            self.save_embeddings()
            self.log_msg(f"✅ Recorded & registered: {user_id}")

    def record_and_identify(self):
        filepath = os.path.join(TEST_DIR, "recorded_test.wav")
        self.record_audio(filepath)
        emb = self.extract_embedding(filepath)
        if emb is None or not self.speakers:
            self.log_msg("❌ No embedding or no speakers.")
            return
        sims = {user: cosine_similarity(emb.reshape(1, -1), ref.reshape(1, -1))[0][0] for user, ref in self.speakers.items()}
        best = max(sims.items(), key=lambda x: x[1])
        self.log_msg(f"🔍 Real-time match: {best[0]} (Score: {best[1]:.2f})")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeakerVerifierGUI(root)
    root.mainloop()
