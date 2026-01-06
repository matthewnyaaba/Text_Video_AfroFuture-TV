# Afrofuture AI TV Studio - Colab Backend
# Paste this whole file into a Colab cell (or run it from a .py file).
# It loads index.html from a raw GitHub URL (optional), displays it, and registers Colab callbacks.

import os, re, time, json, base64, uuid, threading, subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import requests

from IPython.display import display, HTML
from google.colab import files, output

# ---------------------------
# Install packages (quiet)
# ---------------------------
def _pip(pkg: str):
    subprocess.check_call(["python", "-m", "pip", "install", "-q", pkg])

for pkg in ["moviepy", "pydub", "openai>=1.40.0", "google-genai>=0.4.0"]:
    try:
        __import__(pkg.split(">=")[0].split("==")[0].replace("-", "_"))
    except Exception:
        _pip(pkg)

from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment
from openai import OpenAI

# ---------------------------
# Utilities
# ---------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or f"file_{int(time.time())}"

def upload_to_tmpfiles(path: str, timeout: int = 180) -> str:
    """Upload a file to tmpfiles.org and return a direct download URL."""
    with open(path, "rb") as f:
        r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=timeout)
    r.raise_for_status()
    url = r.json()["data"]["url"]
    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/").replace("http://", "https://")

def download_to(url: str, out_path: str, timeout: int = 300) -> str:
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path

def ffmpeg_draw_ticker(in_path: str, out_path: str, ticker_text: str) -> str:
    """Overlay bottom ticker using ffmpeg drawtext."""
    if not ticker_text.strip():
        return in_path
    # Escape for ffmpeg
    txt = ticker_text.replace(":", r"\:").replace("'", r"\'").replace("\n", " ")
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf",
        f"drawtext=text='{txt}':x=20:y=h-60:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12",
        "-c:a", "copy",
        out_path
    ]
    subprocess.check_call(cmd)
    return out_path

def ffmpeg_burn_subtitles(in_path: str, srt_path: str, out_path: str) -> str:
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        out_path
    ]
    subprocess.check_call(cmd)
    return out_path

def seconds_to_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h = ms // 3600000; ms -= h * 3600000
    m = ms // 60000; ms -= m * 60000
    s = ms // 1000; ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments: List[Tuple[float, float, str]], out_path: str) -> str:
    lines = []
    for i, (st, en, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(st)} --> {seconds_to_srt_time(en)}")
        lines.append(text.strip())
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

# ---------------------------
# Job store
# ---------------------------
JOBS: Dict[str, Dict[str, Any]] = {}
UPLOADS: Dict[str, str] = {}  # presenter_image, narration_audio, scene_clip

def job_log(job_id: str, msg: str, level: str = "info"):
    JOBS[job_id]["logs"].append({"msg": msg, "level": level})
    # Trim to last 300 lines to keep payload small
    JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-300:]

def set_progress(job_id: str, pct: int):
    JOBS[job_id]["progress"] = max(0, min(100, int(pct)))

# ---------------------------
# Provider implementations
# ---------------------------
def openai_generate_script(openai_key: str, topics: str, minutes: int, language: str,
                         studio_style: str = "modern", studio_demo: str = "",
                         panel_enabled: bool = False, panel_speakers: str = "") -> str:
    client = OpenAI(api_key=openai_key)
    target_words = int(minutes * 150)
    prompt = (
        f"Write a TV news style script for Afrofuture AI TV in {language}. "
        f"Topics: {topics}. "
        f"Target length: about {minutes} minutes (about {target_words} words). "
        f"Style: clear, confident, warm, Africa-centered, with short paragraphs. "
        f"Studio style: {studio_style}. "
        + (f"Studio visuals/demonstration: {studio_demo}. " if studio_demo else "")
        + ("Panel mode: enabled. Write as a dialogue with speaker labels and balanced turns. "
           f"Speakers: {panel_speakers}. " if panel_enabled else "")
        + "Include a short greeting and a closing. No stage directions."
    )
    # Use responses API for broader compatibility

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return resp.output_text.strip()

def elevenlabs_tts(eleven_key: str, text: str, voice_id: str = "", timeout: int = 180) -> str:
    vid = voice_id.strip() or "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {"xi-api-key": eleven_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.65, "similarity_boost": 0.75}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    out = f"audio_{int(time.time())}.mp3"
    with open(out, "wb") as f:
        f.write(r.content)
    return out

def openai_tts(openai_key: str, text: str, voice: str = "alloy") -> str:
    client = OpenAI(api_key=openai_key)
    out = f"audio_{int(time.time())}.mp3"
    # OpenAI TTS uses audio.speech.create in current docs
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    audio.stream_to_file(out)
    return out

def openai_transcribe(openai_key: str, audio_path: str) -> List[Tuple[float, float, str]]:
    """Return (start, end, text) segments."""
    client = OpenAI(api_key=openai_key)
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    segs = []
    for s in tr.segments:
        segs.append((float(s["start"]), float(s["end"]), s["text"]))
    return segs

def did_create_video(did_key_raw: str, image_url: str, audio_url: str, timeout: int = 180) -> str:
    # D-ID uses Basic auth. We encode key as username with empty password.
    token = base64.b64encode((did_key_raw.strip() + ":").encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
    payload = {
        "source_url": image_url,
        "script": {"type": "audio", "audio_url": audio_url},
        "config": {"fluent": True, "pad_audio": 0.0, "stitch": True}
    }
    r = requests.post("https://api.d-id.com/talks", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    talk_id = r.json()["id"]

    # Poll
    for _ in range(60):
        time.sleep(5)
        s = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers, timeout=timeout)
        s.raise_for_status()
        status = s.json().get("status")
        if status == "done":
            return s.json()["result_url"]
        if status == "error":
            raise RuntimeError(f"D-ID error: {s.text}")
    raise TimeoutError("D-ID video generation timed out.")

def heygen_create_video(heygen_key: str, avatar_id: str, script_text: str, voice_id: str = "", timeout: int = 180) -> str:
    headers = {"X-Api-Key": heygen_key, "Content-Type": "application/json", "accept": "application/json"}
    payload = {
        "test": False,
        "title": f"Afrofuture_{int(time.time())}",
        "scriptText": script_text,
        "avatar": {"avatar_id": avatar_id},
        "background": {"type": "color", "value": "#111827"}
    }
    if voice_id.strip():
        payload["voice"] = {"voice_id": voice_id.strip()}
    r = requests.post("https://api.heygen.com/v2/video/generate", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    video_id = r.json()["data"]["video_id"]

    # Poll status (v1 endpoint)
    status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    for _ in range(120):
        time.sleep(5)
        s = requests.get(status_url, headers=headers, timeout=timeout)
        s.raise_for_status()
        data = s.json().get("data", {})
        if data.get("status") == "completed" and data.get("video_url"):
            return data["video_url"]
        if data.get("status") == "failed":
            raise RuntimeError(f"HeyGen failed: {s.text}")
    raise TimeoutError("HeyGen timed out.")

def synthesia_create_video(synth_key: str, avatar_id: str, script_text: str, timeout: int = 180) -> str:
    # Synthesia uses X-API-KEY header according to their quickstart
    headers = {"X-API-KEY": synth_key, "Content-Type": "application/json"}
    payload = {
        "test": False,
        "title": f"Afrofuture_{int(time.time())}",
        "scriptText": script_text,
        "avatar": {"avatar_id": avatar_id},
        "background": {"type": "color", "value": "#111827"}
    }
    r = requests.post("https://api.synthesia.io/v2/videos", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    video_id = r.json()["id"]

    # Poll
    for _ in range(180):
        time.sleep(5)
        s = requests.get(f"https://api.synthesia.io/v2/videos/{video_id}", headers=headers, timeout=timeout)
        s.raise_for_status()
        data = s.json()
        if data.get("status") == "complete" and data.get("download_url"):
            return data["download_url"]
        if data.get("status") in ("failed", "error"):
            raise RuntimeError(f"Synthesia failed: {s.text}")
    raise TimeoutError("Synthesia timed out.")

def openai_sora_scene(openai_key: str, prompt: str, seconds: int = 5) -> str:
    client = OpenAI(api_key=openai_key)
    seconds = int(seconds)
    # Current OpenAI docs expose /v1/videos generations via SDK: client.videos.generate(...)
    # We keep to a request call through responses if needed, but SDK supports videos in recent versions.
    # Fallback to raw HTTP if SDK lacks attribute.
    try:
        vid = client.videos.generate(
            model="sora-2",
            prompt=prompt,
            seconds=seconds
        )
        # The SDK returns a file-like or url; we download by requesting returned url if present
        # Many responses include "data[0].url"
        url = vid.data[0].url
        out = f"scene_sora_{int(time.time())}.mp4"
        return download_to(url, out)
    except Exception as e:
        raise RuntimeError(f"Sora generation failed (API access may be restricted): {e}")

def gemini_veo_scene(gemini_key: str, prompt: str, seconds: int = 5) -> str:
    # Uses google-genai SDK (Gemini API)
    from google import genai
    client = genai.Client(api_key=gemini_key)
    seconds = int(seconds)
    # Model names can change; this is a common naming pattern in Gemini docs for Veo.
    model = "veo-2.0-generate-001"
    try:
        op = client.models.generate_video(
            model=model,
            prompt=prompt,
            config={"duration_seconds": seconds}
        )
        # Poll operation until done
        while not op.done:
            time.sleep(5)
            op = client.operations.get(op.name)
        # Result contains a uri (GCS or signed url) depending on API;
        # We attempt to fetch a direct URL field if present.
        result = op.response
        # Heuristic: find first URL-like field
        url = None
        if isinstance(result, dict):
            for k in ["video_uri", "uri", "url", "videoUrl", "video_url"]:
                if k in result and isinstance(result[k], str) and result[k].startswith("http"):
                    url = result[k]
                    break
        if not url:
            raise RuntimeError(f"Veo response did not include a direct URL. Raw response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        out = f"scene_veo_{int(time.time())}.mp4"
        return download_to(url, out)
    except Exception as e:
        raise RuntimeError(f"Veo generation failed: {e}")

# ---------------------------
# Script splitting by scenes
# ---------------------------
def split_script(script: str, scenes: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (script_segments, ordered_scenes_to_insert_between_segments)
    We split sequentially by each scene.insert_after phrase (first match).
    """
    text = script
    segs: List[str] = []
    inserts: List[Dict[str, Any]] = []

    for sc in scenes:
        marker = sc["insert_after"]
        idx = text.find(marker)
        if idx == -1:
            # If marker is missing, we do not split; scene will be appended at end
            continue
        cut = idx + len(marker)
        segs.append(text[:cut].strip())
        inserts.append(sc)
        text = text[cut:].strip()

    if text.strip():
        segs.append(text.strip())

    # If some scenes could not be placed, append them at end
    placed = len(inserts)
    if placed < len(scenes):
        for sc in scenes[placed:]:
            inserts.append(sc)
            segs.append("")  # empty narration segment after appended scene

    return segs, inserts

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(job_id: str, state: Dict[str, Any]):
    try:
        JOBS[job_id]["status"] = "running"
        set_progress(job_id, 3)
        job_log(job_id, "Pipeline started.", "info")

        # Extract settings
        providers = state["providers"]
        keys = state["keys"]
        presenter = state["presenter"]
        voice_cfg = state.get("voice", {})
        captions_cfg = state.get("captions", {})
        script_cfg = state.get("script", {})
        scenes = state.get("scenes", [])

        language = state.get("language", "en")
        ticker = state.get("ticker", "")
        studio_cfg = state.get("studio", {})
        studio_style = studio_cfg.get("style", "modern")
        studio_demo = studio_cfg.get("demo", "")
        panel_cfg = studio_cfg.get("panel", {}) or {}
        panel_enabled = bool(panel_cfg.get("enabled", False))
        panel_speakers = str(panel_cfg.get("speakers", ""))


        # 1) Get script
        set_progress(job_id, 6)
        if script_cfg.get("mode") == "openai_generate":
            if not keys.get("openai"):
                raise ValueError("OpenAI key missing for script generation.")
            job_log(job_id, "Generating script with OpenAI...", "info")
            script_text = openai_generate_script(keys["openai"], script_cfg.get("topics",""), int(script_cfg.get("minutes",5)), language,
                                             studio_style=studio_style, studio_demo=studio_demo,
                                             panel_enabled=panel_enabled, panel_speakers=panel_speakers)
        else:
            script_text = script_cfg.get("text","").strip()
        if not script_text:
            raise ValueError("Script is empty.")
        job_log(job_id, "Script ready.", "ok")

        # 2) Split script by scenes
        set_progress(job_id, 10)
        script_segments, insert_scenes = split_script(script_text, scenes)
        job_log(job_id, f"Script split into {len(script_segments)} narration segment(s). Scenes to insert: {len(insert_scenes)}.", "info")

        # 3) Prepare presenter image (D-ID)
        presenter_image_url = presenter.get("imageUrl") or ""
        if not presenter_image_url and "presenter_image_url" in UPLOADS:
            presenter_image_url = UPLOADS["presenter_image_url"]

        # 4) Build timeline video files
        timeline_files: List[str] = []

        # helper: create narration segment video
        def create_narration_video(text: str, seg_index: int) -> Optional[str]:
            txt = (text or "").strip()
            if not txt:
                return None

            # 4a) audio
            if providers["voice"] == "elevenlabs":
                job_log(job_id, f"Audio (ElevenLabs) for segment {seg_index+1}...", "info")
                audio_path = elevenlabs_tts(keys["eleven"], txt, voice_cfg.get("elevenVoiceId",""))
            elif providers["voice"] == "openai_tts":
                job_log(job_id, f"Audio (OpenAI TTS) for segment {seg_index+1}...", "info")
                audio_path = openai_tts(keys["openai"], txt)
            else:
                # upload_audio
                audio_path = UPLOADS.get("narration_audio")
                if not audio_path:
                    raise ValueError("Narration audio not uploaded.")
                job_log(job_id, f"Using uploaded audio for segment {seg_index+1}.", "info")

            audio_url = upload_to_tmpfiles(audio_path)
            job_log(job_id, f"Uploaded audio for segment {seg_index+1}.", "ok")

            # 4b) video
            if providers["talk"] == "did":
                if not presenter_image_url:
                    raise ValueError("Presenter image is required for D-ID.")
                job_log(job_id, f"Creating D-ID presenter segment {seg_index+1}...", "info")
                vurl = did_create_video(keys["did"], presenter_image_url, audio_url)
                out = f"presenter_{seg_index+1}_{int(time.time())}.mp4"
                download_to(vurl, out)
                return out

            if providers["talk"] == "heygen":
                if not state.get("avatar", {}).get("id"):
                    raise ValueError("avatar_id is required for HeyGen.")
                job_log(job_id, f"Creating HeyGen segment {seg_index+1} (avatar_id)...", "info")
                vurl = heygen_create_video(keys["heygen"], state["avatar"]["id"], txt, voice_cfg.get("heygenVoiceId",""))
                out = f"presenter_{seg_index+1}_{int(time.time())}.mp4"
                download_to(vurl, out)
                return out

            if providers["talk"] == "synthesia":
                if not state.get("avatar", {}).get("id"):
                    raise ValueError("avatar_id is required for Synthesia.")
                job_log(job_id, f"Creating Synthesia segment {seg_index+1} (avatar_id)...", "info")
                vurl = synthesia_create_video(keys["synthesia"], state["avatar"]["id"], txt)
                out = f"presenter_{seg_index+1}_{int(time.time())}.mp4"
                download_to(vurl, out)
                return out

            raise ValueError(f"Unknown presenter provider: {providers['talk']}")

        # helper: create scene file
        def create_scene_video(scene: Dict[str, Any], scene_index: int) -> str:
            provider = scene.get("provider_override", "default")
            if provider == "default":
                provider = providers["scene"]
            seconds = int(scene.get("seconds", state.get("defaultSceneSeconds", 5)))
            prompt = scene.get("prompt", "")

            if provider == "upload_scene":
                clip = UPLOADS.get("scene_clip")
                if not clip:
                    raise ValueError("Scene clip not uploaded.")
                job_log(job_id, f"Using uploaded scene clip for scene {scene_index+1}.", "info")
                return clip

            if provider == "sora":
                if not keys.get("openai"):
                    raise ValueError("OpenAI key missing for Sora scenes.")
                job_log(job_id, f"Generating Sora scene {scene_index+1}...", "info")
                return openai_sora_scene(keys["openai"], prompt, seconds)

            if provider == "veo":
                if not keys.get("gemini"):
                    raise ValueError("Gemini key missing for Veo scenes.")
                job_log(job_id, f"Generating Veo scene {scene_index+1}...", "info")
                return gemini_veo_scene(keys["gemini"], prompt, seconds)

            raise ValueError(f"Unknown scene provider: {provider}")

        # 5) Generate segments with interleaved scenes
        set_progress(job_id, 20)
        narration_files: List[str] = []
        narration_audio_paths: List[str] = []

        # We also build one combined audio for transcription later if needed
        for i, seg_text in enumerate(script_segments):
            vid = create_narration_video(seg_text, i)
            if vid:
                timeline_files.append(vid)

            # Insert scene after segment if available
            if i < len(insert_scenes):
                sc = insert_scenes[i]
                scene_file = create_scene_video(sc, i)
                timeline_files.append(scene_file)

            set_progress(job_id, 20 + int((i+1) * 45 / max(1, len(script_segments))))

        job_log(job_id, f"Timeline ready with {len(timeline_files)} clip(s).", "ok")

        # 6) Stitch clips
        set_progress(job_id, 70)
        job_log(job_id, "Stitching final video (moviepy)...", "info")
        clips = []
        for p in timeline_files:
            clips.append(VideoFileClip(p))
        final = concatenate_videoclips(clips, method="compose")
        out_mp4 = f"afrofuture_final_{int(time.time())}.mp4"
        final.write_videofile(out_mp4, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
        for c in clips:
            c.close()
        final.close()
        job_log(job_id, "Stitch complete.", "ok")

        # 7) Ticker overlay
        set_progress(job_id, 78)
        if ticker.strip():
            job_log(job_id, "Adding ticker overlay...", "info")
            ticked = f"afrofuture_ticker_{int(time.time())}.mp4"
            out_mp4 = ffmpeg_draw_ticker(out_mp4, ticked, ticker)
            job_log(job_id, "Ticker added.", "ok")

        # 8) Captions (optional)
        srt_url = ""
        if captions_cfg.get("mode") == "srt_openai":
            if not keys.get("openai"):
                raise ValueError("OpenAI key missing for captions transcription.")
            job_log(job_id, "Generating captions (OpenAI Whisper)...", "info")

            # Extract audio from final video
            audio_tmp = f"final_audio_{int(time.time())}.mp3"
            subprocess.check_call(["ffmpeg", "-y", "-i", out_mp4, "-vn", "-acodec", "mp3", audio_tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            segs = openai_transcribe(keys["openai"], audio_tmp)
            srt_path = f"afrofuture_{int(time.time())}.srt"
            write_srt(segs, srt_path)
            job_log(job_id, "SRT captions created.", "ok")

            if captions_cfg.get("burn","yes") == "yes":
                job_log(job_id, "Burning captions into video...", "info")
                burnt = f"afrofuture_captions_{int(time.time())}.mp4"
                out_mp4 = ffmpeg_burn_subtitles(out_mp4, srt_path, burnt)
                job_log(job_id, "Captions burned into video.", "ok")

            srt_url = upload_to_tmpfiles(srt_path)
            job_log(job_id, "Uploaded SRT.", "ok")

        # 9) Upload final video to tmpfiles for interface link
        set_progress(job_id, 92)
        job_log(job_id, "Uploading final video...", "info")
        final_url = upload_to_tmpfiles(out_mp4, timeout=300)
        job_log(job_id, "Upload complete.", "ok")

        JOBS[job_id]["final_url"] = final_url
        JOBS[job_id]["srt_url"] = srt_url
        JOBS[job_id]["status"] = "done"
        set_progress(job_id, 100)
        job_log(job_id, "All done.", "ok")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        set_progress(job_id, 100)
        job_log(job_id, f"ERROR: {e}", "bad")

# ---------------------------
# Colab callbacks
# ---------------------------
def upload_presenter_image():
    print("Upload presenter image (png/jpg)...")
    up = files.upload()
    if not up:
        return
    name = list(up.keys())[0]
    UPLOADS["presenter_image"] = name
    UPLOADS["presenter_image_url"] = upload_to_tmpfiles(name)
    print("Presenter image uploaded:", UPLOADS["presenter_image_url"])

def upload_narration_audio():
    print("Upload narration audio (mp3/wav)...")
    up = files.upload()
    if not up:
        return
    name = list(up.keys())[0]
    UPLOADS["narration_audio"] = name
    print("Narration audio uploaded:", name)

def upload_scene_clip():
    print("Upload a scene clip (mp4)...")
    up = files.upload()
    if not up:
        return
    name = list(up.keys())[0]
    UPLOADS["scene_clip"] = name
    print("Scene clip uploaded:", name)

def start_job(state_json: str) -> str:
    state = json.loads(state_json)
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status":"queued","progress":0,"logs":[], "final_url":"", "srt_url":"", "error":""}
    job_log(job_id, "Job queued.", "info")

    t = threading.Thread(target=run_pipeline, args=(job_id, state), daemon=True)
    t.start()
    return job_id

def get_job_status(job_id: str) -> str:
    job = JOBS.get(job_id)
    if not job:
        return json.dumps({"status":"error","progress":100,"logs":[{"msg":"Unknown job id","level":"bad"}]})
    return json.dumps(job)

# Register callbacks
output.register_callback("afrofuture.upload_presenter_image", upload_presenter_image)
output.register_callback("afrofuture.upload_narration_audio", upload_narration_audio)
output.register_callback("afrofuture.upload_scene_clip", upload_scene_clip)
output.register_callback("afrofuture.start_job", start_job)
output.register_callback("afrofuture.get_job_status", get_job_status)

print("✅ Afrofuture backend is ready. Now display the interface (see README).")
