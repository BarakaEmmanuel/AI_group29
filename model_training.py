import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os

def train_and_evaluate(df, features, target_col, dataset_name):
    """Trains Random Forest and XGBoost, then evaluates using MAE."""
    print(f"\n--- Training Models for {dataset_name} ---")
    
    # Ensure features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df[target_col]
    
    # Split testing and training sets (80% train, 20% test on unseen data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    
    # 2. XGBoost Regressor
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    
    print(f"Random Forest Mean Absolute Error (MAE): {rf_mae:,.2f}")
    print(f"XGBoost Mean Absolute Error (MAE):       {xgb_mae:,.2f}")

def main():
    # Evaluate Kenya Housing
    if os.path.exists('cleaned_Kenya_housing.csv'):
        kenya_df = pd.read_csv('cleaned_Kenya_housing.csv')
        train_and_evaluate(
            df=kenya_df,
            features=['Neighborhood', 'Bedrooms', 'Bathrooms', 'sq_mtrs'],
            target_col='Price',
            dataset_name='Kenya Housing Dataset'
        )

    # Evaluate Apartments
    if os.path.exists('cleaned_apartments.csv'):
        apt_df = pd.read_csv('cleaned_apartments.csv')
        train_and_evaluate(
            df=apt_df,
            features=['location', 'bedrooms', 'bathrooms'],
            target_col='price',
            dataset_name='Nairobi Apartments Dataset'
        )

    # Evaluate Nairobi Property Prices
    if os.path.exists('cleaned_Nairobi_propertyprices.csv'):
        nrb_df = pd.read_csv('cleaned_Nairobi_propertyprices.csv')
        train_and_evaluate(
            df=nrb_df,
            features=['Location', 'Bedroom', 'bathroom', 'propertyType'],
            target_col='Price',
            dataset_name='Nairobi Property Prices'
        )

    # Evaluate Ames Housing (Testing Structural Details)
    if os.path.exists('cleaned_AmesHousing.csv'):
        ames_df = pd.read_csv('cleaned_AmesHousing.csv')
        # Using a selection of structural features for Ames
        ames_features = ['Neighborhood', 'Overall Qual', 'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'Year Built', 'Bldg Type']
        train_and_evaluate(
            df=ames_df,
            features=ames_features,
            target_col='SalePrice',
            dataset_name='Ames Housing Dataset'
        )

if __name__ == "__main__":
    main()
