# 🏠 Smart Housing Price Predictor

A **production-ready, end-to-end Machine Learning system** designed to deliver accurate housing price predictions using a **Random Forest Regressor** with an **R² score of 0.92**.  
The solution operationalizes ML through a **FastAPI-based REST API** and an **interactive Streamlit frontend**, aligned with real-world deployment standards.

---

## 📌 Business Objective

To enable **data-driven real estate valuation** by exposing a scalable, low-latency prediction service that transforms structured housing attributes into reliable price estimates.

---

## 🏗️ System Architecture

```

smart-housing-price-predictor/
│
├── data/
│   └── raw/
│       └── housing_data.csv        # Training dataset
│
├── models/
│   ├── train.py                    # Model training pipeline
│   └── housing_model.pkl           # Serialized trained model
│
├── backend/
│   ├── main.py                     # FastAPI application
│   └── schemas.py                  # Pydantic schemas
│
├── frontend/
│   └── app.py                      # Streamlit UI
│
└── requirements.txt                # Project dependencies

````

---

## 🚀 Quick Start

### 1️⃣ Clone Repository & Install Dependencies
```bash
git clone https://github.com/ritesh-nagtilak/smart-housing-price-predictor.git
cd smart-housing-price-predictor
pip install -r requirements.txt
````

### 2️⃣ Train the ML Model

```bash
python models/train.py
```

### 3️⃣ Start Backend API (Terminal 1)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4️⃣ Launch Frontend UI (Terminal 2)

```bash
cd ../frontend
streamlit run app.py
```

* **Frontend UI:** [http://localhost:8501](http://localhost:8501)
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔧 Key Features

* **High-Performance ML Model**

  * Random Forest Regressor optimized for tabular data
  * Strong generalization with minimal overfitting

* **Production-Grade API**

  * FastAPI with Pydantic-based input validation
  * Model loaded once at startup for optimal inference performance
  * Sub-10ms average prediction latency

* **Interactive Frontend**

  * Streamlit-powered UI for real-time predictions
  * Clean and intuitive user experience

* **Scalable Architecture**

  * Clear separation of concerns (ML, API, UI)
  * Easily extensible for cloud or container deployment

---

## 📊 Model Performance

| Model             | R² Score | RMSE    | MAE     |
| ----------------- | -------- | ------- | ------- |
| Random Forest     | 0.9247   | $48,250 | $38,420 |
| Linear Regression | 0.78     | $72,000 | $55,000 |

---

## 🔗 API Endpoints

| Method | Endpoint   | Description            |
| ------ | ---------- | ---------------------- |
| GET    | `/`        | Health check           |
| POST   | `/predict` | House price prediction |

* **Swagger UI:** `/docs`

---

## 🛠️ Tech Stack

* **Machine Learning:** scikit-learn (Random Forest)
* **Backend:** FastAPI, Pydantic
* **Frontend:** Streamlit
* **Data Processing:** pandas, numpy
* **Model Serialization:** pickle

---

## 📈 Use Cases

* Real estate price estimation platforms
* Data-driven property analytics
* End-to-end ML portfolio project
* Interview-ready production ML system

---

## 👤 Author

**Ritesh Nagtilak**
Engineer | Data & Machine Learning

---

## 📜 License

Licensed under the **MIT License**.
Free to use, modify, and distribute.

```
```
