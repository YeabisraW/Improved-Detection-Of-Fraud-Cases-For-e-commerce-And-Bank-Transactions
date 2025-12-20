# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def scale_numeric(df, numeric_features):
    """
    Scale numeric features using StandardScaler.
    """
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df

def encode_categorical(df, categorical_features):
    """
    Encode categorical features using one-hot encoding.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]),
                           columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(columns=categorical_features)
    df = pd.concat([df.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    return df
