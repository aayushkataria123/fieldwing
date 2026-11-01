---
title: FieldWing Maize Disease Detection
emoji: 🌽
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# FieldWing

Maize leaf-disease detection from drone video. Upload a short field video: the model marks diseased zones
(red), healthy zones (green) and unsure zones (orange) directly on the video, then explains what to do in the
farmer's language, with voice.

Model: MobileNetV2 (TFLite, quantized) fine-tuned on the "Corn or Maize Leaf Disease Dataset"
(derived from PlantVillage and PlantDoc). 4 classes: Blight, Common Rust, Gray Leaf Spot, Healthy.

**Prototype - AI suggestions only, not a diagnosis. Always confirm treatment with a local agriculture officer (KVK) .**
