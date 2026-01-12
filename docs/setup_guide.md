# 🚀 Complete Setup & Deployment Guide

## End-to-End Instructions for Running the Smart Housing Price Predictor

---

## ✅ Pre-Flight Checklist

- [ ] Python 3.9 or higher installed
- [ ] `pip` available in terminal
- [ ] Git installed (to clone repo)
- [ ] ~500MB free disk space
- [ ] Access to ports 8000 (API) and 8501 (Streamlit)

**Check Python version:**
```bash
python --version
# Should output: Python 3.9.x or higher
```

---

## 📥 Step 1: Clone & Setup Environment

### 1.1 Clone the Repository

```bash
git clone https://github.com/your-username/smart-housing-price-predictor.git
cd smart-housing-price-predictor
```

### 1.2 Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (choose based on your OS)
# Linux/Mac:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (Command Prompt):
venv\Scripts\activate.bat
```

**Verify activation:** Prompt should show `(venv)` prefix

### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:**
```
Collecting scikit-learn...
Collecting fastapi...
Collecting streamlit...
...
Successfully installed numpy-1.24.3 pandas-2.0.3 scikit-learn-1.3.2 fastapi-0.104.1 streamlit-1.28.1
```

---

## 🤖 Step 2: Train the ML Model

### 2.1 Prepare Data

The training data (`housing_data.csv`) should already exist. Verify:

```bash
ls data/raw/housing_data.csv
# Or on Windows:
dir data\raw\housing_data.csv
```

### 2.2 Run Training Pipeline

```bash
python models/train.py
```

**Expected output:**
```
=======================================================================
🏠 HOUSING PRICE PREDICTION - TRAINING PIPELINE
=======================================================================
✓ Loaded 200 samples from housing_data.csv
✓ Training set: 160 samples
✓ Test set: 40 samples

📊 Training Primary Model: Random Forest Regressor...
  R² Score:  0.9247
  RMSE:      $48,250
  MAE:       $38,420

📊 Training Baseline Model: Linear Regression...
  R² Score:  0.8891
  RMSE:      $62,150
  MAE:       $44,680

✓ Model saved: ./models/housing_model.pkl
✓ Scaler saved: ./models/scaler.pkl

=======================================================================
✅ TRAINING COMPLETE - Models ready for deployment
=======================================================================
```

### 2.3 Verify Model Files

```bash
ls -la models/
# Should show:
# housing_model.pkl (random forest model)
# scaler.pkl (feature scaling transformer)
```

---

## 🔌 Step 3: Start Backend API

### 3.1 Navigate to Backend Directory

```bash
cd backend
```

### 3.2 Run FastAPI Server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete [lifespan]
INFO:     ✓ Models loaded successfully. Model version: 1.0.0
```

### 3.3 Verify API is Running

**In a new terminal/PowerShell:**

```bash
curl http://localhost:8000/
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-01-01T17:13:00Z"
}
```

### 3.4 View Interactive API Docs

Open browser: http://localhost:8000/docs

You should see Swagger UI with:
- `/` endpoint (health check)
- `/predict` endpoint (make predictions)
- Test interface to try requests

---

## 🎨 Step 4: Start Frontend UI

### 4.1 Open New Terminal (Keep Backend Running)

```bash
cd frontend
```

### 4.2 Run Streamlit App

```bash
streamlit run app.py
```

**Expected output:**
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501

  Ready to accept connections...
```

### 4.3 Access the UI

Open browser: http://localhost:8501

You should see:
- 🏠 Title: "Housing Price Intelligence Predictor"
- API status: "✅ Backend API: Connected"
- Property input sliders
- "Get Price Prediction" button

---

## 🧪 Step 5: Test the System

### 5.1 Test via UI (Easiest)

1. Go to http://localhost:8501
2. Adjust sliders:
   - Area: 2500 sq ft
   - Bedrooms: 3
   - Bathrooms: 2.5
   - Location Score: 7.5
3. Click "🔮 Get Price Prediction"
4. See estimated price with confidence interval

**Expected result:**
```
💰 Price Estimate
$625,000

Confidence: High (92%)

Low Estimate: $575,000 (95% confident)
High Estimate: $675,000 (95% confident)
```

### 5.2 Test via API (Manual)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 2500,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "location_score": 7.5
  }'
```

**Expected response:**
```json
{
  "predicted_price": 625000.00,
  "estimated_range_low": 575000.00,
  "estimated_range_high": 675000.00,
  "confidence_score": 0.92,
  "model_version": "1.0.0",
  "timestamp": "2026-01-01T17:13:00Z",
  "input_features": {
    "area": 2500,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "location_score": 7.5
  }
}
```

### 5.3 Test Input Validation

Try sending invalid data:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 100,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "location_score": 7.5
  }'
```

**Expected error (422):**
```json
{
  "detail": [
    {
      "msg": "Area too small - minimum 300 sq ft for residential",
      "type": "value_error"
    }
  ]
}
```

---

## 🔄 Full System in Action

### Terminal 1: Backend API
```bash
$ python -m uvicorn backend.main:app --reload
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     ✓ Models loaded successfully
```

### Terminal 2: Frontend UI
```bash
$ streamlit run frontend/app.py
Local URL: http://localhost:8501
```

### Terminal 3: Test Client
```bash
$ curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"area": 2500, "bedrooms": 3, "bathrooms": 2.5, "location_score": 7.5}'

{"predicted_price": 625000.0, ...}
```

---

## 📊 Run Tests (Optional)

```bash
# From project root
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=backend --cov=models
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

### Problem: "Connection refused" when accessing API from UI

**Check:**
1. Is backend running? (Terminal should show "Uvicorn running...")
2. Is API address correct? (Should be http://localhost:8000)
3. Did you forget to activate venv?

**Fix:**
```bash
# Kill existing process
lsof -i :8000
kill -9 <PID>

# Restart backend
python -m uvicorn backend.main:app --reload
```

### Problem: Port 8000 already in use

**Solution:**
```bash
# Use different port
python -m uvicorn backend.main:app --port 8001

# Update frontend API_BASE_URL in app.py:
API_BASE_URL = "http://localhost:8001"
```

### Problem: Port 8501 (Streamlit) already in use

**Solution:**
```bash
streamlit run frontend/app.py --server.port 8502
```

### Problem: "Model not loaded at startup"

**Check:**
1. Did you run `python models/train.py`?
2. Do these files exist?
   ```bash
   ls models/housing_model.pkl
   ls models/scaler.pkl
   ```
3. Check backend logs for errors

### Problem: Prediction returns very different price

**Possible causes:**
1. Training data might have changed
2. Model files might be corrupt
3. Feature scaling might be incorrect

**Solution:** Retrain model:
```bash
python models/train.py
# Restart backend (Ctrl+C, then re-run)
```

---

## 🚢 Deployment Options

### Option 1: Docker (Local)

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0"]
```

**Build & run:**
```bash
docker build -t housing-predictor .
docker run -p 8000:8000 housing-predictor
```

### Option 2: Render.com (Free)

1. Push to GitHub
2. Create new Web Service on Render.com
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python -m uvicorn backend.main:app --host 0.0.0.0`
6. Deploy!

### Option 3: Heroku (Legacy)

```bash
# Create Procfile
echo "web: python -m uvicorn backend.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
git push heroku main
```

### Option 4: AWS EC2

```bash
# On EC2 instance
sudo apt-get update
sudo apt-get install python3-pip
git clone <repo>
cd <repo>
pip install -r requirements.txt
python models/train.py
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 📚 Project File Structure

After setup, your directory should look like:

```
smart-housing-price-predictor/
├── README.md
├── requirements.txt
├── setup_guide.md                 # This file
├── architecture.md                # Design decisions
│
├── data/
│   └── raw/
│       └── housing_data.csv       # Training data (200 samples)
│
├── models/
│   ├── train.py                   # Training script
│   ├── housing_model.pkl          # ✓ Generated after training
│   └── scaler.pkl                 # ✓ Generated after training
│
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── schemas.py                 # Pydantic models
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                     # Streamlit UI
│   └── requirements.txt
│
├── tests/
│   ├── test_api.py
│   └── test_integration.py
│
└── venv/                          # Virtual environment (auto-created)
```

---

## 🎓 Learning Checkpoints

### After Step 2 (Training)
- [ ] Understand Random Forest vs Linear Regression
- [ ] Know why we split train/test data
- [ ] Understand feature scaling and why it matters
- [ ] Can explain pickle serialization

### After Step 3 (Backend)
- [ ] Know what RESTful API means
- [ ] Understand Pydantic validation
- [ ] Know why models load at startup (not per request)
- [ ] Can explain error handling strategy

### After Step 4 (Frontend)
- [ ] Can explain API-first design
- [ ] Know why frontend and ML are separate
- [ ] Understand async/await in Python
- [ ] Can describe confidence intervals

### After Step 5 (Testing)
- [ ] Can debug API issues
- [ ] Understand HTTP status codes (200, 422, 503, 500)
- [ ] Know how to validate inputs
- [ ] Can test end-to-end flows

---

## 🎯 Interview Question Prep

After completing this setup, practice explaining:

1. **"Walk me through your system architecture"**
   - Three layers: ML, Backend, Frontend
   - Why separate? (independence, scalability)
   - How do they communicate? (REST API)

2. **"How does prediction work?"**
   - Load features from input
   - Scale using training scaler
   - Run through Random Forest
   - Return prediction + confidence

3. **"What happens if the API fails?"**
   - Graceful error message
   - Logs show root cause
   - User sees "offline" status
   - Can retry

4. **"How would you scale this?"**
   - Start: Single server (current)
   - Next: Multi-worker FastAPI
   - Later: Kubernetes
   - Eventually: ML serving platform

---

## ✨ What You've Built

You now have a **production-grade ML system** that demonstrates:

✅ End-to-end ML pipeline  
✅ REST API design (FastAPI)  
✅ Input validation (Pydantic)  
✅ Proper error handling  
✅ Separation of concerns  
✅ Type hints & documentation  
✅ Scalable architecture  
✅ User-friendly UI (Streamlit)  

This is **exactly what senior engineers look for** in interviews.

---

## 🤝 Next Steps

1. **Deploy to cloud** (Render, Heroku, AWS)
2. **Add monitoring** (Prometheus, Grafana)
3. **Implement authentication** (API keys)
4. **Add rate limiting** (prevent abuse)
5. **Build CI/CD pipeline** (automated testing/deployment)
6. **Integrate real data** (actual housing market data)
7. **Add more features** (build year, neighborhood, etc.)

---

**Questions? Stuck?** Check the architecture document or review the inline code comments.

**Good luck with your interviews!** 🚀
