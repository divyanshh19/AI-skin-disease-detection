# SkinScan AI — Skin Disease Detection System

CNN + Vision Transformer ensemble for classifying dermoscopic images across
the 7 HAM10000 diagnostic categories, with Grad-CAM explainability, calibrated
confidence scores, a FastAPI backend, and a React frontend.

⚠ **Not a medical device.** For education/research only.

## Project layout

```
backend/
  app/            FastAPI application (routes, services, schemas)
  ml/             training scripts, dataset loaders, preprocessing, evaluation
frontend/
  src/            React application
data/             (gitignored) raw datasets go here
```

## 1. Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

```bash
cd backend/ml/datasets
# Requires ~/.kaggle/kaggle.json (see docstring in the script for setup steps)
python download_ham10000.py --out ../../../data/ham10000

# Optional: augment underrepresented classes from the ISIC archive
python download_isic_subset.py --diagnosis "dermatofibroma" --limit 200 --out ../../../data/isic_subset
```

## 3. Train the models (CPU-friendly)

```bash
cd backend/ml/training
python train_cnn.py --data-root ../../../data/ham10000 --epochs 8 --fine-tune-epochs 3
python train_vit.py --data-root ../../../data/ham10000 --epochs 8 --fine-tune-epochs 3
python train_ensemble_calibration.py --data-root ../../../data/ham10000
```

This produces:
- `ml/saved_models/cnn_efficientnet_b0.pth`
- `ml/saved_models/vit_deit_tiny.pth`
- `ml/saved_models/ensemble_config.json` (ensemble weight + calibration temperatures)

## 4. Evaluate

```bash
cd backend/ml/evaluation
python evaluate.py --model ensemble --data-root ../../../data/ham10000 \
  --cnn-checkpoint ../saved_models/cnn_efficientnet_b0.pth \
  --vit-checkpoint ../saved_models/vit_deit_tiny.pth
```

Outputs accuracy/precision/recall/F1/ROC-AUC and saves confusion matrices to
`ml/evaluation/results/`.

## 5. Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

App at `http://localhost:5173`.

## 7. Docker (both services)

```bash
docker-compose up --build
```

## Testing

```bash
cd backend
pytest tests/
```

## Notes on CPU-only training

Both models freeze their pretrained backbones and train only a lightweight
classifier head first, then optionally unfreeze the last block/blocks for a
short low-LR fine-tuning phase. This keeps backward passes cheap. Expect
roughly 15–30 min/epoch on a modern CPU for the full HAM10000 set.
