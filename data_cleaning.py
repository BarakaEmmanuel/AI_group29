import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_price(price_col):
    """Removes currency symbols, commas, and spaces from price strings and converts to numeric."""
    # Force conversion to string to bypass fragile dtype checks
    price_col = price_col.astype(str)
    # Convert to lowercase to catch 'Ksh', 'KSH', 'ksh', etc.
    price_col = price_col.str.lower()
    price_col = price_col.str.replace('ksh', '', regex=False)
    price_col = price_col.str.replace(',', '', regex=False)
    price_col = price_col.str.replace(' ', '', regex=False)
    # Convert to numeric, coercing unparseable data to NaN
    price_col = pd.to_numeric(price_col, errors='coerce')
    return price_col

def clean_dataset(df, price_col_name, categorical_cols, numeric_cols):
    """Generic cleaning function to handle duplicates, missing values, and encoding."""
    # 1. Remove duplicate entries
    df = df.drop_duplicates()

    # 2. Clean the price target column
    if price_col_name in df.columns:
        df[price_col_name] = clean_price(df[price_col_name])
        # Drop rows where price couldn't be parsed
        df = df.dropna(subset=[price_col_name])

    # 3. Handle Missing Values in numeric columns (Fill with average)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # 4. Convert text data into numbers (Label Encoding for Neighborhoods/Locations)
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    return df

def main():
    print("Starting data cleaning process...")
    
    # Process Apartments Dataset
    if os.path.exists('apartments.csv'):
        apt_df = pd.read_csv('apartments.csv')
        apt_df = clean_dataset(
            apt_df, 
            price_col_name='price', 
            categorical_cols=['location', 'title', 'rate'], 
            numeric_cols=['bedrooms', 'bathrooms']
        )
        apt_df.to_csv('cleaned_apartments.csv', index=False)
        print("Cleaned apartments.csv")

    # Process Kenya Housing Dataset
    if os.path.exists('Kenya_housing.csv'):
        kenya_df = pd.read_csv('Kenya_housing.csv')
        kenya_df = clean_dataset(
            kenya_df, 
            price_col_name='Price', 
            categorical_cols=['Neighborhood', 'Agency'], 
            numeric_cols=['Bedrooms', 'Bathrooms', 'sq_mtrs']
        )
        kenya_df.to_csv('cleaned_Kenya_housing.csv', index=False)
        print("Cleaned Kenya_housing.csv")

    # Process Nairobi Property Prices Dataset
    if os.path.exists('Nairobi propertyprices - Sheet1.csv'):
        nrb_df = pd.read_csv('Nairobi propertyprices - Sheet1.csv')
        nrb_df = clean_dataset(
            nrb_df, 
            price_col_name='Price', 
            categorical_cols=['Location', 'propertyType'], 
            numeric_cols=['Bedroom', 'bathroom']
        )
        nrb_df.to_csv('cleaned_Nairobi_propertyprices.csv', index=False)
        print("Cleaned Nairobi propertyprices - Sheet1.csv")

    # Process Ames Housing Dataset
    if os.path.exists('AmesHousing.csv'):
        ames_df = pd.read_csv('AmesHousing.csv')
        categorical_ames = ames_df.select_dtypes(include=['object']).columns.tolist()
        numeric_ames = ames_df.select_dtypes(exclude=['object']).columns.tolist()
        numeric_ames.remove('SalePrice') if 'SalePrice' in numeric_ames else None
        
        ames_df = clean_dataset(
            ames_df,
            price_col_name='SalePrice',
            categorical_cols=categorical_ames,
            numeric_cols=numeric_ames
        )
        ames_df.to_csv('cleaned_AmesHousing.csv', index=False)
        print("Cleaned AmesHousing.csv")

    print("Data pre-processing complete. Cleaned files saved.")

if __name__ == "__main__":
    main()
