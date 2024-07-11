import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

class LuxuryBrandTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, luxury_brands):
        self.luxury_brands = luxury_brands
        self._new_columns = ['is_luxury_manufacturer']

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['is_luxury_manufacturer'] = X['manufacturer'].apply(lambda x: x in self.luxury_brands)
        return X
    
    def get_feature_names(self):
        return self._new_columns

class MileageConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X['mileage_miles'] = X['mileage'].str.replace(' km', '').astype(int) * 0.621371
        except Exception as e:
            print("Error converting mileage in km to miles: ", e)
            X['mileage_miles'] = X['mileage'].str.replace(' km', '').astype(int) * 0.621371
        return X

class CarAgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, current_year):
        self.current_year = current_year
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['car_age'] = self.current_year - X['prod_year']
        return X
    
class CapCarPrice(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.percentile_97_5 = np.percentile(X['price'], 97.5)
        return self
    
    def transform(self, X):
        X = X.copy()
        X['capped_price'] = X['price'].clip(upper=self.percentile_97_5)
        return X

class DebugTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.head())
        return X

luxury_brands = ['LEXUS', 'MERCEDES-BENZ', 'PORSCHE', 'BMW', 'AUDI', 'INFINITI',
                 'ALFA ROMEO', 'ACURA', 'LINCOLN', 'LAND ROVER', 'JAGUAR', 'CADILLAC',
                 'BENTLEY', 'VOLVO', 'MASERATI', 'FERRARI', 'LAMBORGHINI', 'ROLLS-ROYCE',
                 'ASTON MARTIN', 'TESLA']

color_mapping = {'Carnelian red': 'Red', 'Beige': 'Brown', 'Golden': 'Yellow',
                 'Pink': 'Other', 'Green': 'Other', 'Purple': 'Other', 'Orange': 'Other', 'Sky blue': 'Blue'}

gear_box_mapping = {'Automatic': 1, 'Tiptronic': 1, 'Variator': 1, 'Manual': 0}

def rename_columns(data):
    data.columns = ['car_id', 'price', 'levy', 'manufacturer', 'model', 'prod_year', 'category',
                    'leather_interior', 'fuel_type', 'engine_volume', 'mileage', 'cylinders', 'gear_box_type',
                    'drive_wheels','doors', 'wheel', 'color', 'airbags']
    return data

def clean_transform_data(data):
    categorical_features = ['category','fuel_type','gear_box_type','drive_wheels', 'color']
    categorical_transformer = ColumnTransformer(
        transformers = [
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('luxury_brand', LuxuryBrandTransformer(luxury_brands)),
        ('color_map', FunctionTransformer(lambda df: df.assign(color=df['color'].map(color_mapping).fillna(df['color'])),validate=False)),
        ('mileage_convert', MileageConverter()),
        ('car_age_calc', CarAgeCalculator(current_year=2021)),
        ('price_cap', CapCarPrice()),
        ('categorical_encode', categorical_transformer)
    ])

    pipeline2 = Pipeline([
        ('luxury_brand', LuxuryBrandTransformer(luxury_brands))
    ])

    # Fit and transform data using pipeline
    clean_data = pipeline.fit_transform(data)

    # Retrieve the encoder from the pipeline
    encoder = categorical_transformer.named_transformers_['categorical']

    # Get feature names for the categorical columns
    categorical_cols = encoder.get_feature_names_out(input_features=categorical_features)

    # Get names of passthrough columns
    passthrough_cols = [col for col in data.columns if col not in categorical_features]

    # Combine all column names
    new_column_names = list(categorical_cols) + passthrough_cols + ['is_luxury_manufacturer', 'mileage_miles', 'car_age', 'capped_price']

    # Create a new DataFrame with the transformed data and new column names
    clean_data = pd.DataFrame(clean_data, columns=new_column_names, index=data.index)
    
    return clean_data


    