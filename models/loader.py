import joblib

def load_model(number):
    return joblib.load(f"models/model{number}.pkl")

def load_scaler(number):
    return joblib.load(f"models/scaler_model{number}.pkl")
