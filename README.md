# SatQuery AI

Agentic multi-modal remote sensing vision-language assistant for analyzing single and paired satellite imagery through natural-language queries.

## Features

- **Single-Image Analysis**: Remote-sensing VQA and text-guided spatial region grounding with bounding box generation.
- **Bi-Temporal Change Analysis**: Change vector analysis between $T_1$ and $T_2$ pairs with spatial change heatmaps.
- **Optical-SAR Cross-Modal Fusion**: Joint feature extraction fusing optical spectral reflectance with radar microwave backscatter through cloud cover.
- **Agentic Orchestration**: Input validation, predefined specialist tool registry, and auditable execution logging.

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
```
Add your `GEMINI_API_KEY` to `.env`.

3. Run the application:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open `http://localhost:8000/app` in your browser.

## Deploying to Railway

1. Push this folder to a GitHub repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/satquery-ai.git
git push -u origin main
```

2. On [Railway](https://railway.app):
   - Click **New Project** -> **Deploy from GitHub repo**.
   - Select your repository.
   - In the **Variables** tab, set:
     - `GEMINI_API_KEY`: your Google Gemini API key.
     - `GEMINI_MODEL`: `gemini-3.5-flash` (optional, defaults to `gemini-3.5-flash`).
     - `HF_REPO_ID`: `KowhickMaran/rs-eurosat-classifier` (optional, defaults to `KowhickMaran/rs-eurosat-classifier`).
   - Under **Settings** -> **Networking**, click **Generate Domain**.
   - Access the web app at `https://your-domain.up.railway.app/app`.

## Project Structure

```
├── Procfile
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── agent/
│   ├── audit_log.py
│   ├── gemini_client.py
│   ├── router.py
│   └── tool_registry.py
├── backend/
│   └── main.py
├── inference/
│   ├── rs_inference.py
│   ├── grounding_engine.py
│   ├── change_engine.py
│   └── sar_fusion_engine.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── data/
    └── sample_images/
```
