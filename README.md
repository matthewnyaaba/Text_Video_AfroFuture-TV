# Afrofuture AI TV Studio (Colab + GitHub)

A lightweight, Colab-ready “studio” that lets you:

- Choose providers (voice, presenter video, scenes)
- Paste or generate a script
- Insert scenes during narration by splitting the script at marker phrases
- Stitch everything into one final MP4 (moviepy)
- Add a ticker overlay (ffmpeg drawtext)
- Generate captions (SRT) via OpenAI Whisper (optional)


### Studio style and panel mode

In the **Setup** tab you can now choose a studio ambience and describe what should be visible on screen:

- **Modern Studio**: clean studio, LED wall, news ambience  
- **Sci-Fi Command**: futuristic command room, holograms  
- **Presentation**: slides, charts, lecture visuals  
- **Standing Mode**: presenter in front of screen, talk show style  
- **Panel Discussion Mode (optional)**: enable multiple speakers/avatars (stored in config and can shape script into a dialogue)

### Languages

The interface includes these narration and caption language options (best-effort; captions can also be auto-detected):

- English, French, Spanish, Arabic
- Swahili, Yoruba, **Twi (Akan)**, Hausa, **Zulu**
- **Dagbani (Dagoma)**, **Gurune (Gurene/Frafra)**

> **Important**: Do not commit API keys to GitHub. Enter keys only inside the interface when running in Colab.

---

## Repository structure (recommended)

```
/
  index.html
  colab_backend.py
  README.md
```

- `index.html` is the interface (runs inside Colab output).
- `colab_backend.py` is the Colab backend (installs packages, registers callbacks, generates videos).

---

## Quick start (run from a GitHub URL)

### 1) Upload `index.html` and `colab_backend.py` to GitHub

Create a repo (for example, `afrofuture-studio`) and add these files.

### 2) Open Google Colab and run this boot cell

Replace the two `RAW_URL_*` values with your GitHub **raw** URLs.

```python
import requests
from IPython.display import display, HTML

RAW_URL_INDEX = "https://raw.githubusercontent.com/<YOUR_USER>/<YOUR_REPO>/main/index.html"
RAW_URL_BACKEND = "https://raw.githubusercontent.com/<YOUR_USER>/<YOUR_REPO>/main/colab_backend.py"

# Load backend
backend_code = requests.get(RAW_URL_BACKEND, timeout=60).text
exec(backend_code, globals())

# Load interface
html = requests.get(RAW_URL_INDEX, timeout=60).text
display(HTML(html))
```

When it loads, you will see the Afrofuture Studio interface in the notebook output.

---

## How to use

### Setup tab
1. Choose **Voice provider**:
   - ElevenLabs TTS/clone
   - OpenAI TTS
   - Upload audio
2. Choose **Presenter provider**:
   - D-ID (works with any presenter image)
   - HeyGen (typically uses `avatar_id`)
   - Synthesia (typically uses `avatar_id`)
3. Choose **Scene provider**:
   - OpenAI Sora (prompt → video)
   - Google Veo (prompt → video)
   - Upload a scene clip (mp4)

Then click **Save setup**.

### Script tab
- Choose **Use my script** or **Generate script with OpenAI** (topics → script).
- Paste or generate the script.
- Click **Save script**.

### Timeline tab
- Add scenes by:
  - Writing a scene prompt
  - Providing an **insert-after phrase** that appears in your script
  - Setting scene seconds
  - Optionally overriding the scene provider per scene
- Scenes will be inserted by splitting the script.

### Produce tab
- Click **Start production**
- Watch progress logs
- When done, you will see a final video link (and SRT link if captions were enabled)

---

## Notes and limitations

### Midjourney
Midjourney does not offer an official public API and automated access is commonly restricted. For safety and reliability, this studio does not include a Midjourney key field. If you want Midjourney visuals, generate them manually and then upload them as assets.

### HeyGen and Synthesia
HeyGen and Synthesia often require you to use their own avatars (or enterprise video-avatar pipelines). For personal presenter photos, D-ID is typically the simplest option.

### Scene insertion logic
This version inserts scenes by splitting the script at the **first match** of each marker phrase, in the order you add scenes.

### Links expiry
The generated links may expire depending on the provider or the hosting service used for uploads. Download your final MP4 soon after generation.

---

## Provider documentation (for reference)

- HeyGen uses `X-Api-Key` and provides video generation + status endpoints.  
- Synthesia provides a REST API for creating and polling videos.  
- D-ID “talks” API accepts a source image URL and an audio URL.  
- OpenAI provides TTS, transcription (Whisper), and video generation endpoints depending on account access.  
- Gemini provides a video generation model (Veo) depending on account access.

---

## Troubleshooting

- If buttons do nothing: ensure you are running inside **Google Colab** (not a plain browser page).  
- If a provider fails: confirm your plan supports the endpoint and that your key is valid.  
- If captions fail: make sure your OpenAI key is set and the audio is reachable.

---

## License
MIT (recommended).
