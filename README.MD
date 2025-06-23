# 🧠 Profile Image Verification Pipeline

This Python script performs **automated profile image verification** using face detection and NSFW (Not Safe For Work) content detection.

It is designed to:
- ✅ Check if exactly one face is present (human & single person detection)
- 🔒 Ensure the image is **safe** (non-explicit) using AI
- 🚫 Skip gender classification (you can enable it later if needed)

---

## 📁 Features

- **Face Detection** using OpenCV
- **NSFW Detection** using [NudeNet](https://github.com/bedapudi678/nudenet)
- Easy to run as a standalone script
- Outputs a clear result with safety status and face count

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
