# Speaker-Verification-GUI
# 🎤 Speaker Verification GUI (SpeechBrain ECAPA-TDNN)

An intuitive and efficient GUI-based system for real-time speaker registration and verification using the [SpeechBrain](https://speechbrain.readthedocs.io/) library and ECAPA-TDNN model.

---

## 🚀 Features

✅ Register speaker from audio file  
✅ Record voice and register directly  
✅ Identify speaker from file or real-time recording  
✅ Auto-register all files in `ref_voices/`  
✅ View logs, delete speakers, and adjust threshold  
✅ Modern Tkinter interface (VSCode-inspired style)


## GUI_3
![image](https://github.com/user-attachments/assets/a6d361e1-213d-4213-88af-44f9a57ffbd5)
## GUI_2
![image](https://github.com/user-attachments/assets/d4768d61-974a-4798-95f6-14f17b2e940a)
## GUI_1
![image](https://github.com/user-attachments/assets/e58a6a97-8ff9-4081-bdc8-e3cb32d56420)

---

## 🧠 Model Information

- **Model**: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Downloaded automatically** from Hugging Face on first run
- **Stored locally** in `pretrained_models/` folder for reuse (no need to redownload)

---

## 🛠️ Installation

```bash
git clone https://github.com/Mrkomiljon/Speaker-Verification-GUI.git
cd Speaker-Verification-GUI
```
# Install dependencies
```
pip install -r requirements.txt
```
## ▶️ Usage
```
python speaker_verification_gui_1/2/3.py
```
What You Can Do:
🎙 Record your voice → auto-save to ref_voices/ → register

📁 Select any .wav or .mp3 → identify speaker

🧹 Clear logs, ❌ Delete speakers, 🔧 Adjust threshold

## 📌 Notes
Ensure your microphone is enabled and default device is selected.

The minimum required audio duration is 2 seconds for accurate embedding.

Supported formats: .wav, .mp3

## 💡 Future Plans
 Multi-model selection (e.g., ECAPA vs. Titanet)

 Speaker statistics and timeline

 CRM integration (e.g., saving call logs or speaker names)

 ## 📄 License
This project is open-source and available under the MIT License.

## ✨ Acknowledgements
[SpeechBrain](https://speechbrain.readthedocs.io/en/latest/)

[ECAPA-TDNN Paper](https://arxiv.org/abs/2005.07143)

Tkinter + ttk for UI
