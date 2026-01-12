import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime


class HousingModelTrainer: 
    def __init__(self, data_path: str, model_dir: str = './models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.primary_model = None
        self.baseline_model = None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def load_data(self) -> bool:
        try:
            self.df = pd.read_csv(self.data_path)
            
            required_cols = ['area', 'bedrooms', 'bathrooms', 'location_score', 'price']
            missing = [col for col in required_cols if col not in self.df.columns]
            
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if self.df.isnull().any().any():
                print("⚠️  Warning: Missing values found. Dropping rows...")
                self.df = self.df.dropna()
            
            print(f"✓ Loaded {len(self.df)} samples from {self.data_path}")
            return True
        
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def prepare_features(self) -> bool:
        try:
            X = self.df[['area', 'bedrooms', 'bathrooms', 'location_score']]
            y = self.df['price']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"✓ Training set: {len(self.X_train)} samples")
            print(f"✓ Test set: {len(self.X_test)} samples")
            return True
        
        except Exception as e:
            print(f"✗ Error preparing features: {str(e)}")
            return False
    
    def train_primary_model(self) -> bool:
        try:
            print("\n📊 Training Primary Model: Random Forest Regressor...")

            self.primary_model = RandomForestRegressor(
                n_estimators=100,        
                max_depth=15,            
                min_samples_split=5,     
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1                
            )
            
            self.primary_model.fit(self.X_train, self.y_train)

            y_pred = self.primary_model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            
            print(f"  R² Score:  {r2:.4f}")
            print(f"  RMSE:      ${rmse:,.0f}")
            print(f"  MAE:       ${mae:,.0f}")
            
            return True
        
        except Exception as e:
            print(f"✗ Error training primary model: {str(e)}")
            return False
    
    def train_baseline_model(self) -> bool:
        try:
            print("\n📊 Training Baseline Model: Linear Regression...")
            
            self.baseline_model = LinearRegression()
            self.baseline_model.fit(self.X_train, self.y_train)
            
            y_pred = self.baseline_model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            
            print(f"  R² Score:  {r2:.4f}")
            print(f"  RMSE:      ${rmse:,.0f}")
            print(f"  MAE:       ${mae:,.0f}")
            
            return True
        
        except Exception as e:
            print(f"✗ Error training baseline model: {str(e)}")
            return False
    
    def save_model(self, model, filename: str) -> bool:
        try:
            filepath = os.path.join(self.model_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Model saved: {filepath}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
            return False
    
    def save_scaler(self, filename: str = 'scaler.pkl') -> bool:
        try:
            filepath = os.path.join(self.model_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"✓ Scaler saved: {filepath}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving scaler: {str(e)}")
            return False
    
    def run_pipeline(self):
        print("\n" + "="*70)
        print("🏠 HOUSING PRICE PREDICTION - TRAINING PIPELINE")
        print("="*70)
        
        if not self.load_data():
            return False

        if not self.prepare_features():
            return False

        if not self.train_primary_model():
            return False
        
        if not self.train_baseline_model():
            return False

        if not self.save_model(self.primary_model, 'housing_model.pkl'):
            return False
        
        if not self.save_scaler():
            return False
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE - Models ready for deployment")
        print("="*70)
        
        return True

if __name__ == "__main__":
    trainer = HousingModelTrainer(
        data_path='./data/raw/housing_data.csv',
        model_dir='./models'
    )
    
    success = trainer.run_pipeline()
    exit(0 if success else 1)
