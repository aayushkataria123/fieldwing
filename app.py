"""FieldWing - maize disease detection from drone video (Hugging Face Space).

Files expected next to this script:  maize_model_dynamic.tflite  (or maize_model_fp32.tflite)  +  class_names.json
Optional Space secret:               GEMINI_API_KEY  (only for languages without a built-in template)
"""
import os, json, shutil, subprocess, tempfile, threading, warnings
from collections import Counter
import cv2
import numpy as np
import gradio as gr
import requests

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = next(p for p in (os.path.join(HERE, n) for n in ("maize_model_dynamic.tflite", "maize_model_fp32.tflite"))
             if os.path.exists(p))
CLASS_NAMES = json.load(open(os.path.join(HERE, "class_names.json")))
K = len(CLASS_NAMES)
HEALTHY = next(i for i, c in enumerate(CLASS_NAMES) if "health" in c.lower())
PRETTY = {i: c.replace("_", " ") for i, c in enumerate(CLASS_NAMES)}

MAX_SECONDS = 10
SAMPLE_FPS = 2
OUT_W_MAX = 960
MIN_GREEN = 0.25
SMOOTH = 3

# ---------------------------------------------------------------- model
_interp = Interpreter(model_path=MODEL, num_threads=os.cpu_count() or 2)
_interp.allocate_tensors()
_in, _out = _interp.get_input_details()[0], _interp.get_output_details()[0]
IMG = int(_in["shape"][1])
_lock = threading.Lock()


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
    act = np.where(~valid, 0, np.where(conf < thresh, 2, np.where(k == HEALTHY, 1, 3)))
    return act, k, conf


# ---------------------------------------------------------------- drawing (overlay built once per sample)
RED, GRN, ORG = (40, 40, 225), (70, 185, 70), (0, 150, 240)           # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(L, text, x, y, fs):
    # outline is (1,1,1) not (0,0,0) so "any non-zero pixel" works as the opaque mask
    cv2.putText(L, text, (x, y), FONT, fs, (1, 1, 1), 3, cv2.LINE_AA)
    cv2.putText(L, text, (x, y), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)


def build_overlay(act, k, conf, xs, ys, style, w, h):
    """Returns (inv, add) so that  frame_out = frame * inv + add  (cheap per-frame)."""
    rows, cols = act.shape
    A = np.zeros((h, w), np.float32)                    # blend alpha
    C = np.empty((h, w, 3), np.float32)
    C[:] = RED
    L = np.zeros((h, w, 3), np.uint8)                   # opaque drawings

    if style in ("Glow", "Both"):
        score = np.where(act == 3, conf, 0).astype(np.float32)
        sc = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(sc, (0, 0), sigmaX=max(8, w / cols / 2.2))
        plants = cv2.resize((act > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        A = np.clip(sc * 1.7, 0, 1) * plants * 0.6

    if style in ("Boxes", "Both"):
        if style == "Boxes":
            for r in range(rows):
                for c in range(cols):
                    if act[r, c] == 3:
                        A[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = 0.30
        fs = float(np.clip((xs[1] - xs[0]) / 330, 0.4, 0.7))
        for r in range(rows):
            for c in range(cols):
                a = act[r, c]
                x0, y0, x1, y1 = xs[c] + 2, ys[r] + 2, xs[c + 1] - 2, ys[r + 1] - 2
                if a == 3:
                    cv2.rectangle(L, (x0, y0), (x1, y1), RED, 3)
                    _label(L, f"{PRETTY[int(k[r, c])]} {conf[r, c]:.0%}", x0 + 6, y0 + int(24 * fs / 0.5), fs)
                elif a == 2:
                    cv2.rectangle(L, (x0, y0), (x1, y1), ORG, 2)
                elif a == 1:
                    cv2.rectangle(L, (x0, y0), (x1, y1), GRN, 1)

    # header bar + legend
    n_valid, n_dis, n_uns = int((act > 0).sum()), int((act == 3).sum()), int((act == 2).sum())
    A[:46] = 0.72
    C[:46] = (25, 25, 25)
    cv2.putText(L, "FIELDWING", (16, 31), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(L, f"Plant zones: {n_valid}   Diseased: {n_dis} ({n_dis / max(1, n_valid):.0%})   Unsure: {n_uns}",
                (230, 31), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    x = 16
    for col, txt in [(RED, "Disease"), (GRN, "Healthy"), (ORG, "Unsure")]:
        cv2.rectangle(L, (x, h - 34), (x + 16, h - 18), col, -1)
        cv2.putText(L, txt, (x + 24, h - 19), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 120

    M = L.any(-1)[..., None]
    A3 = np.where(M, 1.0, A[..., None]).astype(np.float32)
    C = np.where(M, L.astype(np.float32), C)
    return (1 - A3), (C * A3)


# ---------------------------------------------------------------- advice (templates; Gemini only for other languages)
DISEASE_HI = {"Common Rust": "कॉमन रस्ट (गेरुआ रोग)", "Gray Leaf Spot": "ग्रे लीफ स्पॉट (पत्तों पर धूसर धब्बे)",
              "Blight": "ब्लाइट (झुलसा रोग)"}
DISEASE_MR = {"Common Rust": "कॉमन रस्ट (तांबेरा)", "Gray Leaf Spot": "ग्रे लीफ स्पॉट (पानांवर राखाडी ठिपके)",
              "Blight": "ब्लाइट (करपा)"}
LANG_CODE = {"Hindi": "hi", "English": "en", "Marathi": "mr", "Gujarati": "gu", "Tamil": "ta", "Telugu": "te",
             "Bengali": "bn", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa"}
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
TEMPLATE_LANGS = ("Hindi", "English", "Marathi")


def template_advice(lang, disease, pct, unsure_pct):
    none = disease is None or pct < 2
    if lang == "English":
        if none:
            if unsure_pct >= 40:
                return ("The drone could not judge most of this clip clearly. Please check the field by eye, "
                        "or upload a steadier video filmed closer to the plants.")
            return "No significant disease was found in this clip, so spraying is not needed now. Please keep watching the field."
        t = (f"The drone check found {disease} on about {pct:.0f}% of the plants it could see. Spray only the patches marked in red; "
             "the green areas are healthy and do not need spraying. ")
        t += "Please look at the orange marks by eye, the drone was not sure there. " if unsure_pct >= 5 else ""
        return t + "This is an AI suggestion. Ask your local agriculture officer (KVK) for the right medicine and amount."

    if lang == "Marathi":
        if none:
            if unsure_pct >= 40:
                return ("ड्रोनला या व्हिडिओचा बहुतेक भाग नीट समजला नाही. कृपया शेतात जाऊन स्वतः पाहा, "
                        "किंवा झाडांच्या जवळून काढलेला स्थिर व्हिडिओ टाका.")
            return "या व्हिडिओमध्ये मोठा रोग आढळला नाही, सध्या फवारणीची गरज नाही. तरीही शेतावर लक्ष ठेवा."
        dn = DISEASE_MR.get(disease, disease)
        t = (f"ड्रोन तपासणीत सुमारे {pct:.0f}% झाडांवर {dn} आढळला आहे. फक्त लाल खुणा केलेल्या भागातच फवारणी करा; "
             "हिरवे भाग निरोगी आहेत, तिथे फवारणीची गरज नाही. ")
        t += "केशरी खुणा असलेले भाग स्वतः जाऊन पाहा. " if unsure_pct >= 5 else ""
        return t + "ही AI ची सूचना आहे. योग्य औषध आणि प्रमाणासाठी जवळच्या कृषी अधिकाऱ्याला (KVK) विचारा."

    # Hindi
    if none:
        if unsure_pct >= 40:
            return ("ड्रोन इस वीडियो के ज़्यादातर हिस्से को साफ़ नहीं समझ पाया। कृपया खेत खुद जाकर देखें, "
                    "या पौधों के पास से बना साफ़ और स्थिर वीडियो डालें।")
        return "इस वीडियो में कोई बड़ी बीमारी नहीं मिली, अभी छिड़काव की ज़रूरत नहीं है। फिर भी खेत पर नज़र रखें।"
    dn = DISEASE_HI.get(disease, disease)
    t = (f"ड्रोन की जाँच में लगभग {pct:.0f}% पौधों पर {dn} मिला है। सिर्फ़ लाल निशान वाले हिस्सों में ही दवा का छिड़काव करें; "
         "हरे हिस्से ठीक हैं, वहाँ छिड़काव की ज़रूरत नहीं है। ")
    t += "नारंगी निशान वाले हिस्सों को खुद जाकर देख लें। " if unsure_pct >= 5 else ""
    return t + "यह AI की सलाह है। सही दवा और मात्रा के लिए अपने नज़दीकी कृषि अधिकारी (KVK) से पूछें।"


def ask_gemini(lang, disease, pct, unsure_pct):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    extra = ("The drone could NOT judge most of the video clearly, so do not say the field is healthy; "
             "ask him to check by eye or film closer. ") if (disease is None and unsure_pct >= 40) else ""
    facts = (extra + f"Disease found: {disease or 'none'}. About {pct:.0f}% of the plants in view are affected. "
             f"About {unsure_pct:.0f}% of the area was unclear and should be checked by eye. "
             "Red marks on the video show where to spray; green means healthy, no spray.")
    prompt = (f"You are a friendly agriculture helper speaking to a small farmer in India, in {lang}. Write a short spoken message "
              f"(maximum 60 words) based ONLY on these facts from a drone check of his maize field: {facts} "
              "Rules: simple everyday words; plain text only (no markdown, bullets or emojis); tell him to spray only the red-marked "
              "patches if disease was found; say this is an AI suggestion; do NOT name any specific pesticide or dose - tell him to ask "
              "the local agriculture officer (KVK); do not give any protective-equipment or safety instructions. "
              "Reply with the message only.")
    for model in GEMINI_MODELS:
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
            if r.ok:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
    return None


_advice_cache = {}


def make_advice(lang, disease, pct, unsure_pct):
    ck = (lang, disease, round(pct), round(unsure_pct))
    if ck in _advice_cache:
        return _advice_cache[ck]
    if lang in TEMPLATE_LANGS:
        res = (template_advice(lang, disease, pct, unsure_pct), lang)
    else:
        g = ask_gemini(lang, disease, pct, unsure_pct)
        res = (g, lang) if g else (template_advice("Hindi", disease, pct, unsure_pct), "Hindi")
    _advice_cache[ck] = res
    return res


def speak(text, lang):
    try:
        from gtts import gTTS
        path = os.path.join(tempfile.mkdtemp(), "advice.mp3")
        gTTS(text=text, lang=LANG_CODE.get(lang, "hi")).save(path)
        return path
    except Exception:
        return None


def update_advice(language, state):
    """Cheap step: re-runs only text + voice. The video is NOT reprocessed."""
    if not state:
        return "", None
    text, used = make_advice(language, state["disease"], state["pct"], state["unsure_pct"])
    if used != language:
        text += f"\n\n({language} needs a Gemini key on this Space - showing {used} instead.)"
    return text, speak(text, used)


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
def run_video(video, thresh, tile_px, style, progress=gr.Progress()):
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
            progress(0.05 + 0.6 * f / max(1, end_f), desc="Looking for disease...")
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

    # build overlays ONCE per sample (the slow part used to be redoing this for every frame)
    OV = [build_overlay(*DEC[s], xs, ys, style, out_w, out_h) for s in range(S)]
    progress(0.7, desc="Marking the video...")

    # pass 2: apply cached overlay to every frame
    tmpd = tempfile.mkdtemp()
    raw = os.path.join(tmpd, "raw.mp4")
    final = os.path.join(tmpd, "fieldwing_marked.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    cap = cv2.VideoCapture(video)
    for f in range(end_f):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        s = int(np.clip(round(f / step), 0, S - 1))
        inv, add = OV[s]
        vw.write((fr * inv + add).astype(np.uint8))
    cap.release()
    vw.release()
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "ultrafast", "-crf", "28", final], check=True)
        except Exception:
            final = raw
    else:
        final = raw

    # summary + state for advice
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
    state = {"disease": disease, "pct": pct, "unsure_pct": unsure_pct}
    return final, summary, state


# ---------------------------------------------------------------- UI
with gr.Blocks(title="FieldWing - maize disease detection") as demo:
    gr.Markdown("# 🌽 FieldWing\n**Maize leaf-disease detection from drone video - mark where to spray, explain it in the farmer's language.**")
    gr.Markdown("*Prototype. The model was trained on maize leaf photos (4 classes: " + ", ".join(PRETTY.values()) +
                "). Results on field video are indicative, not a diagnosis.*")
    with gr.Tab("Drone video"):
        result = gr.State(None)
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="Upload a short field video")
                with gr.Accordion("Settings", open=False):
                    th = gr.Slider(0.5, 0.95, value=0.7, step=0.05, label="Confidence needed to mark a zone as disease/healthy")
                    tile = gr.Slider(120, 300, value=180, step=20, label="Zone size (pixels)")
                    sty = gr.Radio(["Both", "Boxes", "Glow"], value="Both", label="Marking style")
                go = gr.Button("Analyse video", variant="primary")
            with gr.Column():
                out_vid = gr.Video(label="Marked video")
                out_sum = gr.Markdown()
                lang = gr.Dropdown(list(LANG_CODE), value="Hindi", label="Advice language (change anytime - video is not re-analysed)")
                out_adv = gr.Textbox(label="Advice for the farmer", lines=5)
                out_aud = gr.Audio(label="Listen", type="filepath")
        go.click(run_video, [vid, th, tile, sty], [out_vid, out_sum, result]) \
          .then(update_advice, [lang, result], [out_adv, out_aud])
        lang.change(update_advice, [lang, result], [out_adv, out_aud])
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
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
