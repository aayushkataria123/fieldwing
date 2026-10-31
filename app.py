"""FieldWing - maize disease detection from drone video (Hugging Face Space).

Files expected next to this script:  maize_model_dynamic.tflite  (or maize_model_fp32.tflite)  +  class_names.json
Optional Space secret:               GEMINI_API_KEY  (only needed for advice in languages other than Hindi / English)
"""
import os, json, shutil, subprocess, tempfile, threading, warnings
from collections import Counter
import cv2
import numpy as np
import gradio as gr
import requests

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:                                    # fallback if the Space uses full TensorFlow instead
    from tensorflow.lite import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = next(p for p in (os.path.join(HERE, n) for n in ("maize_model_dynamic.tflite", "maize_model_fp32.tflite"))
             if os.path.exists(p))
CLASS_NAMES = json.load(open(os.path.join(HERE, "class_names.json")))
K = len(CLASS_NAMES)
HEALTHY = next(i for i, c in enumerate(CLASS_NAMES) if "health" in c.lower())
PRETTY = {i: c.replace("_", " ") for i, c in enumerate(CLASS_NAMES)}

MAX_SECONDS = 30        # Allow up to 30 seconds of video
SAMPLE_FPS = 1          # Check 1 frame per second (cuts analysis time in half)
OUT_W_MAX = 640         # Scale down video width (cuts frame processing time by 50%+)
MIN_GREEN = 0.25
SMOOTH = 3

# ---------------------------------------------------------------- model
_interp = Interpreter(model_path=MODEL, num_threads=2)
_interp.allocate_tensors()
_in, _out = _interp.get_input_details()[0], _interp.get_output_details()[0]
IMG = int(_in["shape"][1])
_lock = threading.Lock()                              # one prediction at a time (interpreter is not thread-safe)


def predict(rgb):
    x = cv2.resize(rgb, (IMG, IMG)).astype(np.float32)[None]
    with _lock:
        _interp.set_tensor(_in["index"], x)
        _interp.invoke()
        return _interp.get_tensor(_out["index"])[0].copy()


def green_frac(rgb):
    return cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (25, 40, 40), (90, 255, 255)).mean() / 255


def tile_probs(rgb, xs, ys):
    rows, cols = len(ys) - 1, len(xs) - 1
    A = np.full((rows, cols, K), np.nan, np.float32)
    for r in range(rows):
        for c in range(cols):
            t = rgb[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            if green_frac(t) >= MIN_GREEN:
                A[r, c] = predict(t)
    return A


def decide(A, thresh):
    valid = ~np.isnan(A[..., 0])
    V = np.nan_to_num(A)
    k, conf = V.argmax(-1), V.max(-1)
    act = np.where(~valid, 0, np.where(conf < thresh, 2, np.where(k == HEALTHY, 1, 3)))   # 0 none, 1 ok, 2 unsure, 3 disease
    return act, k, conf


# ---------------------------------------------------------------- drawing
RED, GRN, ORG = (40, 40, 225), (70, 185, 70), (0, 150, 240)           # BGR


def _label(fr, text, x, y, fs):
    cv2.putText(fr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(fr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)


def render(fr, act, k, conf, xs, ys, style):
    h, w = fr.shape[:2]
    rows, cols = act.shape
    if style in ("Glow", "Both"):
        score = np.where(act == 3, conf, 0).astype(np.float32)
        sc = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(sc, (0, 0), sigmaX=max(8, w / cols / 2.2))
        plants = cv2.resize((act > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        sc = (np.clip(sc * 1.7, 0, 1) * plants)[..., None]
        fr = (fr * (1 - 0.6 * sc) + np.array(RED, np.float32) * 0.6 * sc).astype(np.uint8)
    if style in ("Boxes", "Both"):
        if style == "Boxes":
            ov = fr.copy()
            for r in range(rows):
                for c in range(cols):
                    if act[r, c] == 3:
                        cv2.rectangle(ov, (xs[c], ys[r]), (xs[c + 1], ys[r + 1]), RED, -1)
            fr = cv2.addWeighted(ov, 0.30, fr, 0.70, 0)
        fs = float(np.clip((xs[1] - xs[0]) / 330, 0.4, 0.7))
        for r in range(rows):
            for c in range(cols):
                a = act[r, c]
                x0, y0, x1, y1 = xs[c] + 2, ys[r] + 2, xs[c + 1] - 2, ys[r + 1] - 2
                if a == 3:
                    cv2.rectangle(fr, (x0, y0), (x1, y1), RED, 3)
                    _label(fr, f"{PRETTY[int(k[r, c])]} {conf[r, c]:.0%}", x0 + 6, y0 + int(24 * fs / 0.5), fs)
                elif a == 2:
                    cv2.rectangle(fr, (x0, y0), (x1, y1), ORG, 2)
                elif a == 1:
                    cv2.rectangle(fr, (x0, y0), (x1, y1), GRN, 1)
    n_valid, n_dis, n_uns = int((act > 0).sum()), int((act == 3).sum()), int((act == 2).sum())
    ov = fr.copy()
    cv2.rectangle(ov, (0, 0), (w, 46), (25, 25, 25), -1)
    fr = cv2.addWeighted(ov, 0.72, fr, 0.28, 0)
    cv2.putText(fr, "FIELDWING", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(fr, f"Plant zones: {n_valid}   Diseased: {n_dis} ({n_dis / max(1, n_valid):.0%})   Unsure: {n_uns}",
                (230, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    x = 16
    for col, txt in [(RED, "Disease"), (GRN, "Healthy"), (ORG, "Unsure")]:
        cv2.rectangle(fr, (x, h - 34), (x + 16, h - 18), col, -1)
        cv2.putText(fr, txt, (x + 24, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 120
    return fr


# ---------------------------------------------------------------- advice (template, or Gemini for other languages)
DISEASE_HI = {"Common Rust": "कॉमन रस्ट (गेरुआ रोग)", "Gray Leaf Spot": "ग्रे लीफ स्पॉट (पत्तों पर धूसर धब्बे)",
              "Blight": "ब्लाइट (झुलसा रोग)"}
LANG_CODE = {"Hindi": "hi", "English": "en", "Marathi": "mr", "Gujarati": "gu", "Tamil": "ta", "Telugu": "te",
             "Bengali": "bn", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa"}
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]


def template_advice(lang, disease, pct, unsure_pct):
    if lang == "English":
        if disease is None or pct < 2:
            if unsure_pct >= 40:
                return ("The drone could not judge most of this clip clearly. Please check the field by eye, "
                        "or upload a steadier video filmed closer to the plants.")
            return "No significant disease was found in this clip, so spraying is not needed now. Please keep watching the field."
        t = (f"The drone check found {disease} on about {pct:.0f}% of the plants it could see. Spray only the patches marked in red; "
             "the green areas are healthy and do not need spraying. ")
        t += "Please look at the orange marks by eye, the drone was not sure there. " if unsure_pct >= 5 else ""
        return t + ("This is an AI suggestion. Ask your local agriculture officer (KVK) for the right medicine and amount, "
                    "and cover your face and hands while spraying.")
    if disease is None or pct < 2:
        if unsure_pct >= 40:
            return ("ड्रोन इस वीडियो के ज़्यादातर हिस्से को साफ़ नहीं समझ पाया। कृपया खेत खुद जाकर देखें, "
                    "या पौधों के पास से बना साफ़ और स्थिर वीडियो डालें।")
        return "इस वीडियो में कोई बड़ी बीमारी नहीं मिली, अभी छिड़काव की ज़रूरत नहीं है। फिर भी खेत पर नज़र रखें।"
    dn = DISEASE_HI.get(disease, disease)
    t = (f"ड्रोन की जाँच में लगभग {pct:.0f}% पौधों पर {dn} मिला है। सिर्फ़ लाल निशान वाले हिस्सों में ही दवा का छिड़काव करें; "
         "हरे हिस्से ठीक हैं, वहाँ छिड़काव की ज़रूरत नहीं है। ")
    t += "नारंगी निशान वाले हिस्सों को खुद जाकर देख लें। " if unsure_pct >= 5 else ""
    return t + ("यह AI की सलाह है। सही दवा और मात्रा के लिए अपने नज़दीकी कृषि अधिकारी (KVK) से पूछें, "
                "और छिड़काव करते समय मुँह और हाथ ढक कर रखें।")


def ask_gemini(lang, disease, pct, unsure_pct):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    if disease is None and unsure_pct >= 40:
        facts_extra = "The drone could NOT judge most of the video clearly, so do not say the field is healthy; ask him to check by eye or film closer. "
    else:
        facts_extra = ""
    facts = (facts_extra + f"Disease found: {disease or 'none'}. About {pct:.0f}% of the plants in view are affected. "
             f"About {unsure_pct:.0f}% of the area was unclear and should be checked by eye. "
             "Red marks on the video show where to spray; green means healthy, no spray.")
    prompt = (f"You are a friendly agriculture helper speaking to a small farmer in India, in {lang}. Write a short spoken message "
              f"(maximum 80 words) based ONLY on these facts from a drone check of his maize field: {facts} "
              "Rules: simple everyday words; plain text only (no markdown, bullets or emojis); tell him to spray only the red-marked "
              "patches if disease was found; say this is an AI suggestion; do NOT name any specific pesticide or dose - tell him to ask "
              "the local agriculture officer (KVK); add one short safety line about covering face and hands and avoiding wind or rain. "
              "Reply with the message only.")
    for model in GEMINI_MODELS:
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            if r.ok:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
    return None


def make_advice(lang, disease, pct, unsure_pct):
    if lang in ("Hindi", "English"):
        text = template_advice(lang, disease, pct, unsure_pct)
        g = ask_gemini(lang, disease, pct, unsure_pct) if os.environ.get("GEMINI_API_KEY") else None
        return g or text, lang
    g = ask_gemini(lang, disease, pct, unsure_pct)
    if g:
        return g, lang
    return template_advice("Hindi", disease, pct, unsure_pct), "Hindi"      # no key / failed -> fall back to Hindi


def speak(text, lang):
    try:
        from gtts import gTTS
        path = os.path.join(tempfile.mkdtemp(), "advice.mp3")
        gTTS(text=text, lang=LANG_CODE.get(lang, "hi")).save(path)
        return path
    except Exception:
        return None


# ---------------------------------------------------------------- tab 1: single leaf photo
def run_image(img, thresh):
    if img is None:
        raise gr.Error("Please upload a leaf photo.")
    p = predict(img)
    k = int(p.argmax())
    if p[k] < thresh:
        verdict = f"Unsure ({p[k]:.0%}). Please check this plant by eye - no spray decision."
    elif k == HEALTHY:
        verdict = f"Healthy ({p[k]:.0%}). No spray needed."
    else:
        verdict = f"{PRETTY[k]} detected ({p[k]:.0%}). Needs treatment - ask your agriculture officer for the right medicine."
    return {PRETTY[i]: float(v) for i, v in enumerate(p)}, verdict


# ---------------------------------------------------------------- tab 2: drone video
def run_video(video, language, thresh, tile_px, style, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video (a short clip, up to ~20 MB, works best).")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise gr.Error("Could not read this video. Try an .mp4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W0 == 0 or H0 == 0:
        raise gr.Error("Could not read this video. Try an .mp4 file.")
    duration = n_frames / fps if n_frames > 0 else MAX_SECONDS
    end_f = int(min(duration, MAX_SECONDS) * fps)

    out_w = int(min(OUT_W_MAX, W0)) // 2 * 2
    out_h = int(H0 * out_w / W0) // 2 * 2
    cols, rows = max(1, round(out_w / tile_px)), max(1, round(out_h / tile_px))
    xs, ys = np.linspace(0, out_w, cols + 1).astype(int), np.linspace(0, out_h, rows + 1).astype(int)
    step = max(1, int(round(fps / SAMPLE_FPS)))

    # pass 1: classify zones on sampled frames
    samples = []
    for f in range(end_f):
        if f % step == 0:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            samples.append(tile_probs(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), xs, ys))
            progress(0.05 + 0.5 * f / max(1, end_f), desc="Looking for disease...")
        elif not cap.grab():
            break
    cap.release()
    if not samples:
        raise gr.Error("No frames could be read from this video.")

    A = np.stack(samples)
    S = len(A)
    sm = np.empty_like(A)
    half = SMOOTH // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(S):
            sm[s] = np.nanmean(A[max(0, s - half):s + half + 1], axis=0)
    DEC = [decide(sm[s], thresh) for s in range(S)]

    # pass 2: draw marks on every frame
    raw = os.path.join(tempfile.mkdtemp(), "raw.mp4")
    final = raw.replace("raw.mp4", "fieldwing_marked.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    cap = cv2.VideoCapture(video)
    for f in range(end_f):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        s = int(np.clip(round(f / step), 0, S - 1))
        vw.write(render(fr, *DEC[s], xs, ys, style))
        if f % 10 == 0:
            progress(0.55 + 0.4 * f / max(1, end_f), desc="Marking the video...")
    cap.release()
    vw.release()
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "27", final], check=True)
        except Exception:
            final = raw
    else:
        final = raw

    # summary + advice
    shares = [(a == 3).sum() / max(1, (a > 0).sum()) for a, _, _ in DEC]
    uns = [(a == 2).sum() / max(1, (a > 0).sum()) for a, _, _ in DEC]
    cnt = Counter(PRETTY[int(k[r, c])] for a, k, _ in DEC for r in range(a.shape[0]) for c in range(a.shape[1]) if a[r, c] == 3)
    disease = cnt.most_common(1)[0][0] if cnt else None
    pct, peak, unsure_pct = 100 * float(np.mean(shares)), 100 * float(np.max(shares)), 100 * float(np.mean(uns))
    clipped = " (only the first %d s are analysed)" % MAX_SECONDS if duration > MAX_SECONDS else ""
    summary = (f"**Analysed:** {min(duration, MAX_SECONDS):.0f} s of video{clipped}  \n"
               f"**Main disease:** {disease or 'none found'}  \n"
               f"**Average share of visible plants flagged:** {pct:.0f}%  (peak {peak:.0f}%)  \n"
               f"**Unsure areas:** {unsure_pct:.0f}%")
    progress(0.97, desc="Writing advice...")
    text, used_lang = make_advice(language, disease, pct, unsure_pct)
    if used_lang != language:
        text += f"\n\n({language} needs a Gemini key on this Space - showing {used_lang} instead.)"
    return final, summary, text, speak(text, used_lang)


# ---------------------------------------------------------------- UI
with gr.Blocks(title="FieldWing - maize disease detection") as demo:
    gr.Markdown("# 🌽 FieldWing\n**Maize leaf-disease detection from drone video - mark where to spray, explain it in the farmer's language.**")
    gr.Markdown("*Prototype. The model was trained on maize leaf photos (4 classes: " + ", ".join(PRETTY.values()) +
                "). Results on field video are indicative, not a diagnosis.*")
    with gr.Tab("Drone video"):
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="Upload a short field video")
                lang = gr.Dropdown(list(LANG_CODE), value="Hindi", label="Advice language")
                with gr.Accordion("Settings", open=False):
                    th = gr.Slider(0.5, 0.95, value=0.7, step=0.05, label="Confidence needed to mark a zone as disease/healthy")
                    tile = gr.Slider(120, 300, value=180, step=20, label="Zone size (pixels)")
                    sty = gr.Radio(["Both", "Boxes", "Glow"], value="Both", label="Marking style")
                go = gr.Button("Analyse video", variant="primary")
            with gr.Column():
                out_vid = gr.Video(label="Marked video")
                out_sum = gr.Markdown()
                out_adv = gr.Textbox(label="Advice for the farmer", lines=6)
                out_aud = gr.Audio(label="Listen", type="filepath")
        go.click(run_video, [vid, lang, th, tile, sty], [out_vid, out_sum, out_adv, out_aud])
    with gr.Tab("Single leaf photo"):
        with gr.Row():
            img = gr.Image(type="numpy", label="Upload a leaf photo")
            with gr.Column():
                lab = gr.Label(num_top_classes=K, label="Prediction")
                ver = gr.Textbox(label="What to do", lines=3)
        th2 = gr.Slider(0.5, 0.95, value=0.7, step=0.05, label="Confidence needed")
        gr.Button("Check leaf", variant="primary").click(run_image, [img, th2], [lab, ver])

demo.queue(max_size=10)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
