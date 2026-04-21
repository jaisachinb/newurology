# NeuroScan — Brain CT Hemorrhage Detection

## ⚠️ Bug Fix Applied
The original project had `index.html` in the root folder, but Flask requires
templates to be inside a `templates/` folder. This has been fixed.

## Folder Structure (CORRECT)
```
NeuroScan-Fixed/
├── app.py                  ← Flask backend
├── requirements.txt
├── templates/
│   └── index.html          ← ✅ Moved here (was in root — that was the bug!)
├── static/
│   └── uploads/            ← Uploaded images saved here
└── models/                 ← Place your trained models here (optional)
    ├── svm_model_linear_pca.pkl
    ├── scaler.pkl
    ├── pca.pkl
    ├── label_encoder.pkl
    ├── model_vgg.json
    ├── model_vgg.weights.h5
    ├── subtype_model_vgg.json
    ├── subtype_model_vgg.weights.h5
    └── subtype_class_index.json
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Flask server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

## Notes
- If model files are NOT in the `models/` folder, it runs in **DEMO MODE**
  (still shows predictions based on image brightness heuristics).
- **Never open `index.html` directly in a browser** — it won't work because
  the `/analyze` endpoint only exists when Flask is running.
- Make sure you run `python app.py` FIRST, then open `http://localhost:5000`.
