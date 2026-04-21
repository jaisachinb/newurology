"""
NeuroScan — Brain CT Hemorrhage Detection & Classification
Flask Backend
"""

import os, cv2, numpy as np, json, time, base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# ── Hemorrhage Taxonomy ───────────────────────────────────────────────────────
HEMORRHAGE_TAXONOMY = {
    "source":   ["Arterial", "Venous", "Capillary"],
    "location": ["External", "Internal"],
    "cause":    ["Traumatic", "Spontaneous"],
    "region":   ["Intracranial", "Subarachnoid", "Epidural",
                 "Subdural", "Gastrointestinal", "Postpartum"],
}

CATEGORY_META = {
    "source":   {"icon": "🩸", "label": "By Source",   "color": "#f59e0b"},
    "location": {"icon": "📍", "label": "By Location", "color": "#3b82f6"},
    "cause":    {"icon": "⚡", "label": "By Cause",    "color": "#8b5cf6"},
    "region":   {"icon": "🧠", "label": "By Region",   "color": "#ef4444"},
}

# ── Model loader ──────────────────────────────────────────────────────────────
models_loaded = {}

def try_load_models():
    """Attempt to load real trained models; fall back gracefully."""
    global models_loaded
    try:
        import joblib
        from tensorflow.keras.models import model_from_json

        models_loaded["svm"]    = joblib.load("models/svm_model_linear_pca.pkl")
        models_loaded["scaler"] = joblib.load("models/scaler.pkl")
        models_loaded["pca"]    = joblib.load("models/pca.pkl")
        models_loaded["le"]     = joblib.load("models/label_encoder.pkl")

        with open("models/model_vgg.json") as f:
            vgg = model_from_json(f.read())
        vgg.load_weights("models/model_vgg.weights.h5")
        models_loaded["vgg_binary"] = vgg

        with open("models/subtype_model_vgg.json") as f:
            sub = model_from_json(f.read())
        sub.load_weights("models/subtype_model_vgg.weights.h5")
        models_loaded["subtype"] = sub

        with open("models/subtype_class_index.json") as f:
            ci = json.load(f)
        models_loaded["idx_to_class"] = {v: k for k, v in ci.items()}

        print("✅ All models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Models not found ({e}). Running in DEMO mode.")
        return False

MODELS_READY = try_load_models()

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def demo_predict(image_path):
    """
    Demo prediction used when real models are not loaded.
    Returns realistic-looking results based on simple image analysis.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simple heuristic: bright regions may indicate hemorrhage
    mean_val = float(np.mean(gray))
    bright_pixels = int(np.sum(gray > 180))
    total_pixels = gray.shape[0] * gray.shape[1]
    bright_ratio = bright_pixels / total_pixels

    is_hemorrhagic = bright_ratio > 0.03 or mean_val > 45
    base_conf = 72 + bright_ratio * 200
    base_conf = min(max(base_conf, 60), 97)

    if is_hemorrhagic:
        detection = {
            "overall": "Hemorrhagic",
            "svm_pred": "Hemorrhagic", "svm_conf": round(base_conf - 2, 1),
            "vgg_pred": "Hemorrhagic", "vgg_conf": round(base_conf + 1, 1),
            "avg_conf": round(base_conf, 1),
        }
        # Subtype demo
        np.random.seed(int(mean_val * 100) % 2**31)
        by_category = {}
        for cat, classes in HEMORRHAGE_TAXONOMY.items():
            probs = np.random.dirichlet(np.ones(len(classes)) * 0.8)
            best  = int(np.argmax(probs))
            by_category[cat] = {
                "predicted":  classes[best],
                "confidence": round(float(probs[best]) * 100, 1),
                "all_probs":  {c: round(float(p) * 100, 1)
                               for c, p in zip(classes, probs)},
            }
        top_preds = [
            {"class": by_category["region"]["predicted"],
             "confidence": by_category["region"]["confidence"]},
            {"class": by_category["source"]["predicted"],
             "confidence": by_category["source"]["confidence"]},
            {"class": by_category["cause"]["predicted"],
             "confidence": by_category["cause"]["confidence"]},
        ]
    else:
        detection = {
            "overall": "NORMAL",
            "svm_pred": "NORMAL", "svm_conf": round(base_conf + 3, 1),
            "vgg_pred": "NORMAL", "vgg_conf": round(base_conf + 1, 1),
            "avg_conf": round(base_conf + 2, 1),
        }
        by_category = {}
        top_preds   = []

    return {"detection": detection, "by_category": by_category,
            "top_predictions": top_preds, "demo_mode": True}


def real_predict(image_path):
    """Run actual model inference."""
    VGG_SIZE = (150, 150)
    result = {}

    # Binary detection
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_f = cv2.resize(gray, (90, 90)).flatten().reshape(1, -1)
    gray_f = models_loaded["pca"].transform(
             models_loaded["scaler"].transform(gray_f))
    svm_pred = models_loaded["le"].inverse_transform(
               [int(models_loaded["svm"].predict(gray_f)[0])])[0]
    svm_conf = round(float(np.max(
               models_loaded["svm"].predict_proba(gray_f))) * 100, 1)

    rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    vgg_in = cv2.resize(rgb, VGG_SIZE) / 255.0
    probs  = models_loaded["vgg_binary"].predict(
             np.expand_dims(vgg_in, 0), verbose=0)[0]
    vgg_classes = ["Hemorrhagic", "NORMAL"]
    vgg_pred = vgg_classes[int(np.argmax(probs[:2]))]
    vgg_conf = round(float(np.max(probs[:2])) * 100, 1)
    overall  = ("Hemorrhagic"
                if "em" in svm_pred.lower() or "em" in vgg_pred.lower()
                else "NORMAL")

    result["detection"] = {
        "overall": overall,
        "svm_pred": svm_pred, "svm_conf": svm_conf,
        "vgg_pred": vgg_pred, "vgg_conf": vgg_conf,
        "avg_conf": round((svm_conf + vgg_conf) / 2, 1),
    }

    if overall == "Hemorrhagic":
        sub_probs  = models_loaded["subtype"].predict(
                     np.expand_dims(vgg_in, 0), verbose=0)[0]
        idx2cls    = models_loaded["idx_to_class"]
        by_category = {}
        for cat, classes in HEMORRHAGE_TAXONOMY.items():
            cat_idx = [i for i, c in idx2cls.items() if c in classes]
            best    = max(cat_idx, key=lambda i: sub_probs[i])
            by_category[cat] = {
                "predicted":  idx2cls[best],
                "confidence": round(float(sub_probs[best]) * 100, 1),
                "all_probs":  {idx2cls[i]: round(float(sub_probs[i]) * 100, 1)
                               for i in cat_idx},
            }
        top_k = np.argsort(sub_probs)[::-1][:3]
        result["top_predictions"] = [
            {"class": idx2cls[int(i)], "confidence": round(float(sub_probs[i]) * 100, 1)}
            for i in top_k
        ]
        result["by_category"] = by_category
    else:
        result["by_category"]    = {}
        result["top_predictions"] = []

    result["demo_mode"] = False
    return result

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           models_ready=MODELS_READY,
                           taxonomy=HEMORRHAGE_TAXONOMY,
                           category_meta=CATEGORY_META)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG."}), 400

    filename  = f"{int(time.time())}_{secure_filename(file.filename)}"
    filepath  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        if MODELS_READY:
            result = real_predict(filepath)
        else:
            result = demo_predict(filepath)

        result["image_url"] = f"/static/uploads/{filename}"
        result["category_meta"] = CATEGORY_META
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/info")
def api_info():
    return jsonify({
        "name":        "NeuroScan API",
        "version":     "1.0.0",
        "models_ready": MODELS_READY,
        "taxonomy":    HEMORRHAGE_TAXONOMY,
        "endpoints": {
            "POST /analyze": "Upload CT image for analysis",
            "GET /":         "Web interface",
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
