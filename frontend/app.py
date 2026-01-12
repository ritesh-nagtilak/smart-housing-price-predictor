import streamlit as st
import requests
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .price-display {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .confidence-badge {
        display: inline-block;
        padding: 10px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 5px;
    }
    .high-confidence {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-confidence {
        background-color: #fff3cd;
        color: #856404;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10

# Feature constraints for UI
CONSTRAINTS = {
    'area': {'min': 300, 'max': 10000, 'step': 100, 'default': 2500},
    'bedrooms': {'min': 1, 'max': 10, 'step': 1, 'default': 3},
    'bathrooms': {'min': 1.0, 'max': 6.0, 'step': 0.5, 'default': 2.0},
    'location_score': {'min': 1.0, 'max': 10.0, 'step': 0.5, 'default': 7.0}
}

def check_api_health() -> bool:

    try:
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=TIMEOUT
        )
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False


def get_prediction(area: float, bedrooms: int, bathrooms: float, location_score: float) -> dict:
    try:
        payload = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location_score": location_score
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        else:
            return {
                'success': False,
                'error': f"API Error {response.status_code}: {response.text}"
            }
    
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': f"Cannot connect to API at {API_BASE_URL}. Is the backend running?"
        }
    
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': "API request timeout. Try again."
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error: {str(e)}"
        }


def format_currency(value: float) -> str:
    """Format number as USD currency."""
    return f"${value:,.0f}"


def get_confidence_badge(score: float) -> str:

    percentage = score * 100
    if score >= 0.85:
        css_class = "high-confidence"
        label = "High"
    elif score >= 0.70:
        css_class = "medium-confidence"
        label = "Medium"
    else:
        css_class = "low-confidence"
        label = "Low"
    
    return f'<span class="confidence-badge {css_class}">{label} ({percentage:.0f}%)</span>'

# Header
st.title("🏠 Housing Price Intelligence Predictor")
st.markdown("""
**Predict accurate property prices in seconds** using AI-powered analysis.
Get instant valuations with confidence intervals for better decision-making.
""")

# API Status
col1, col2, col3 = st.columns(3)
with col1:
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ Backend API: Connected")
    else:
        st.error("❌ Backend API: Offline")
        st.stop()

# Main content area
col_input, col_result = st.columns([2, 2], gap="large")

with col_input:
    st.subheader("📋 Property Details")
    
    # Area input
    area = st.slider(
        "Square Footage",
        min_value=CONSTRAINTS['area']['min'],
        max_value=CONSTRAINTS['area']['max'],
        value=CONSTRAINTS['area']['default'],
        step=CONSTRAINTS['area']['step'],
        help="Total property area in square feet"
    )
    st.caption(f"*{area:,} sq ft*")
    
    # Bedrooms
    bedrooms = st.select_slider(
        "Bedrooms",
        options=range(
            CONSTRAINTS['bedrooms']['min'],
            CONSTRAINTS['bedrooms']['max'] + 1
        ),
        value=CONSTRAINTS['bedrooms']['default'],
        help="Number of bedrooms"
    )
    
    # Bathrooms
    bathrooms = st.slider(
        "Bathrooms",
        min_value=CONSTRAINTS['bathrooms']['min'],
        max_value=CONSTRAINTS['bathrooms']['max'],
        value=CONSTRAINTS['bathrooms']['default'],
        step=CONSTRAINTS['bathrooms']['step'],
        help="Number of bathrooms"
    )
    
    # Location score
    location_score = st.slider(
        "Location Score",
        min_value=CONSTRAINTS['location_score']['min'],
        max_value=CONSTRAINTS['location_score']['max'],
        value=CONSTRAINTS['location_score']['default'],
        step=CONSTRAINTS['location_score']['step'],
        help="1 = Poor location, 10 = Excellent location"
    )
    predict_button = st.button(
        "🔮 Get Price Prediction",
        use_container_width=True,
        type="primary"
    )

with col_result:
    st.subheader("💰 Price Estimate")
    
    if predict_button:
        with st.spinner("Analyzing property..."):
            result = get_prediction(area, bedrooms, bathrooms, location_score)
        
        if result['success']:
            pred = result['data']
            
            st.markdown(f"""
            <div class="price-display">
                {format_currency(pred['predicted_price'])}
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence
            st.markdown(
                f"**Confidence:** {get_confidence_badge(pred['confidence_score'])}",
                unsafe_allow_html=True
            )
            st.divider()
            
            # Price range
            col_low, col_high = st.columns(2)
            with col_low:
                st.metric(
                    "Low Estimate",
                    format_currency(pred['estimated_range_low']),
                    delta="95% confident"
                )
            
            with col_high:
                st.metric(
                    "High Estimate",
                    format_currency(pred['estimated_range_high']),
                    delta="95% confident"
                )
            
            st.divider()
            
            # Model info
            st.caption(f"""
            **Model Version:** {pred['model_version']}  
            **Predicted At:** {datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}
            """)
            
            # Store result in session state for visualization
            st.session_state.last_prediction = pred
        
        else:
            st.error(f"Prediction failed: {result['error']}")
    
    else:
        st.info(
            "👈 **Enter property details and click 'Get Price Prediction'** "
            "to see the estimated market price.",
            icon="ℹ️"
        )

if 'last_prediction' in st.session_state:
    st.divider()
    st.subheader("📊 Price Analysis")
    
    pred = st.session_state.last_prediction
    
    # Create confidence interval visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot range
    price_low = pred['estimated_range_low']
    price_high = pred['estimated_range_high']
    price_mid = pred['predicted_price']
    
    ax.barh([0], [price_high - price_low], left=price_low, height=0.5, 
            color='lightblue', label='95% Confidence Range')
    ax.plot([price_mid], [0], 'o', color='darkblue', markersize=15, 
            label='Predicted Price', zorder=5)
    
    # Formatting
    ax.set_xlim(price_low * 0.95, price_high * 1.05)
    ax.set_yticks([])
    ax.set_xlabel('Estimated Price ($)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    st.pyplot(fig, use_container_width=True)
    
    # Comparison metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Price per Sq Ft", f"${pred['predicted_price']/area:.2f}")
    with col2:
        st.metric("Uncertainty Range", 
                 f"±{((price_high - price_low) / 2 / price_mid * 100):.1f}%")
    with col3:
        st.metric("Model Confidence", 
                 f"{pred['confidence_score']*100:.0f}%")

st.markdown("""
---
**Smart Housing Price Predictor** | *Production-Ready ML System*
Built with Python, FastAPI, Streamlit, and Scikit-learn for AI/ML interviews.
""")
