# SatQuery AI

Plain-English satellite image analysis. Upload an image, ask a question,
get an answer backed by a real remote-sensing-adapted model — not a
generic AI guess.

## How it works

1. User uploads an image + asks a question
2. `agent/router.py` decides what kind of task this is
3. The right specialist model analyzes the image
   (`inference/rs_inference.py` — fine-tuned CLIP classifier)
4. `agent/gemini_client.py` turns the structured result into a natural answer
5. `agent/audit_log.py` records every step (the "not a black box" evidence)

## Local setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # then fill in your real GEMINI_API_KEY
uvicorn backend.main:app --reload
```

First request will auto-download the model from Hugging Face
(`KowhickMaran/rs-eurosat-classifier`) into `models/rs-eurosat-classifier/`.

Test it:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "question=What kind of land is this?" \
  -F "image=@data/sample_images/test.png"
```

## Deploying to Railway

1. Push this folder to a GitHub repo
2. Railway → New Project → Deploy from GitHub repo
3. Railway auto-detects the `Procfile` and runs it
4. Go to your service → **Variables** tab → add:
   - `GEMINI_API_KEY` = your real key
   - `HF_REPO_ID` = `KowhickMaran/rs-eurosat-classifier` (only if different from default)
5. Deploy — first request will be slower (downloading the model), after that it's fast

## Status

| Feature | Status |
|---|---|
| RS-adapted specialist model | Done |
| Local inference | Done |
| Backend + Gemini integration | Done (single-image path) |
| Agent routing | Basic version done |
| Audit trail | Done |
| Change detection | Not built |
| Optical+SAR fusion | Not built |
| Frontend | Not built |
