from fastapi import FastAPI
import os
import csv
from typing import Optional, List, Dict
import time
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import glob
from fastapi.middleware.cors import CORSMiddleware
from models.loader import load_model,load_scaler
from sklearn.preprocessing import StandardScaler



app = FastAPI(
    title="FastAPI Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ou ["*"] pour tout autoriser (moins sécurisé)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model1=load_model(1)
model2=load_model(2)
model3=load_model(3)

scaler_model1=load_scaler(1)
scaler_model2=load_scaler(2)
scaler_model3=load_scaler(3)


def read_csv(path: str):
    print("hi")
    df = pd.read_csv(path, encoding="utf-8")
    print(df)
    return df
   

class MultiCSVCache:
    def __init__(self):
        self.caches = {}  # { "file.csv": {...cache...} }

    def register(self, name: str, path: str):
        """Déclare un CSV à mettre en cache."""
        self.caches[name] = {
            "path": path,
            "data": None,
        }

    def get(self, name: str):
        """Récupère un CSV depuis le cache."""
        if name not in self.caches:
           self.register(name,f"{name}.csv")
        info = self.caches[name]
        path = info["path"]
       

        should_reload = (
            info["data"] is None 
            
        )
        if should_reload:
            info["data"] = read_csv(path)


        return info["data"]

csv_manager=MultiCSVCache()    
    

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.get("/load_csv/{csv_name}")
def load_csv(csv_name):
    file_path=f"{csv_name}.csv"
    if os.path.exists(file_path):
        df = csv_manager.get(csv_name)
        vehicules = df["vehicle_number"].drop_duplicates().tolist()
        laps_recorded = len(df["lap_number"].drop_duplicates())
        return JSONResponse(
        status_code=200,
        content={"vehicules":vehicules,"laps_recorded":laps_recorded}
       )

    return JSONResponse(
        status_code=404,
        content={"error":"error"}
    )    

@app.get("/races")
def races():
   try: 
    csv_files = glob.glob("*.csv")
    csv_names = []
    for csv_file in csv_files:
        name, ext = os.path.splitext(csv_file)
        csv_names.append(name)
    return JSONResponse(
        status_code=200,
        content={"races":csv_names}
    )    
    
   except:
       return JSONResponse(
        status_code=500,
    )    


@app.post("/start")
async def start_simulation(request:Request):
    #first lap of the vehicle number that we get it in input from csv file in the input
    data = await request.json()
    print(data)
    csv_name=data['name']
    vehicle_number=data['vehicle_number']
    df=csv_manager.get(csv_name).copy()
    df["lap_time"] = pd.to_timedelta(df["lap_time"], errors="coerce")
    df["lap_time"] = df["lap_time"].dt.total_seconds()
    df["lap_time_variance"] = df.groupby("vehicle_number")["lap_time"].transform("var")
    df["remaining_laps"]=df["lap_number"].max() - df["lap_number"]
    first_lap = df[(df["vehicle_number"] == vehicle_number) & (df["lap_number"] == 1)].copy()
    print(first_lap["lap_time"])
    features = ["lap_number",'lap_time' ,"accx_can","Steering_Angle", "accy_can", "pbrake_f", "pbrake_r", "tyre_age"]
    first_lap_model1=first_lap[features]
    X_scaled = scaler_model1.transform(first_lap_model1)

    first_lap_model1 = pd.DataFrame(X_scaled, columns=features)
    print(first_lap_model1)
    loss_per_lap=model1.predict(first_lap_model1)
    first_lap["loss_per_lap"]=loss_per_lap

    features = ["accx_can", "accy_can", "Steering_Angle", "pbrake_f", "speed", "lap_time_variance","tyre_age"]
    first_lap_model2=first_lap[features]
    X_scaled = scaler_model2.transform(first_lap_model2)

    first_lap_model2 = pd.DataFrame(X_scaled, columns=features)
    print(first_lap_model2)
    incident=model2.predict_proba(first_lap_model2)[:, 1]
    first_lap["incident"]=incident
    features = ["lap_number", 'lap_time','remaining_laps',"accx_can", "accy_can", "Steering_Angle", "pbrake_f", "speed","incident","tyre_age"]
    first_lap_model3=first_lap[features]
    scaler = StandardScaler()
    X_scaled = scaler_model3.transform(first_lap_model3)

    first_lap_model3 = pd.DataFrame(X_scaled, columns=features)
    print(first_lap_model3)
    expected_gain_if_pit_now=model3.predict(first_lap_model3)
    first_lap["expected_gain_if_pit_now"]=expected_gain_if_pit_now
    print(first_lap)
    row = first_lap.iloc[0]  # <- récupérer la ligne

    return JSONResponse(
      status_code=200,
      content={
        "lapTime": round(float(row["lap_time"]),2),
        "lapTimeGain": round(float(row["loss_per_lap"]),2),
        "speed": round(float(row["speed"]),2),
        "tyreAge": int(row["tyre_age"]),
        "expectedGain": round(float(row["expected_gain_if_pit_now"]),2),
        "incidentRisk": round(float(row["incident"]), 2)
      }
    ) 


@app.post("/next_lap")
async def run(request:Request):
    #get the row which correspond lap number and vehicule
    data = await request.json()
    csv_name=data['name']
    vehicle_number=data['vehicle_number']
    lap_number=data["lap_number"]
    pit_lap=data["pit_lap"]
    gain=data["gain"]
    df=csv_manager.get(csv_name).copy()
    df["lap_time"] = pd.to_timedelta(df["lap_time"], errors="coerce")
    df["lap_time"] = df["lap_time"].dt.total_seconds()
    df["lap_time_variance"] = df.groupby("vehicle_number")["lap_time"].transform("var")
    df["remaining_laps"]=df["lap_number"].max() - df["lap_number"]
    mask = (df["vehicle_number"] == vehicle_number) & (df["lap_number"] == lap_number)

# Récupérer la valeur pit_time
    pit_time = df.loc[mask, "pit_time"].values[0]

# Condition : pit_time existe et n'est pas vide
    if pd.notna(pit_time) and pit_time != "":
    # Si pit_lap == lap_no
     if pit_lap == lap_number:
        df.loc[mask, "lap_time"] += gain
    else:
    # pit_time n'existe pas ou est vide
     if pit_lap != lap_number:
        df.loc[mask, "lap_time"] += gain
    df["remaining_laps"]=df["lap_number"].max() - df["lap_number"]
    first_lap = df[(df["vehicle_number"] == vehicle_number) & (df["lap_number"] == lap_number)].copy()
    first_lap["tyre_age"]=lap_number-pit_lap
    print(first_lap['lap_time'])
    # change the necessary as pit and tyre age 
    # launch the three models 
    # return informations
   
    features = ["lap_number",'lap_time' ,"accx_can","Steering_Angle", "accy_can", "pbrake_f", "pbrake_r", "tyre_age"]
    first_lap_model1=first_lap[features]
    X_scaled = scaler_model1.transform(first_lap_model1)

    first_lap_model1 = pd.DataFrame(X_scaled, columns=features)
    loss_per_lap=model1.predict(first_lap_model1)
    first_lap["loss_per_lap"]=loss_per_lap

    features = ["accx_can", "accy_can", "Steering_Angle", "pbrake_f", "speed", "lap_time_variance","tyre_age"]
    first_lap_model2=first_lap[features]
    X_scaled = scaler_model2.transform(first_lap_model2)

    first_lap_model2 = pd.DataFrame(X_scaled, columns=features)
    incident=model2.predict_proba(first_lap_model2)[:, 1]
    first_lap["incident"]=incident
    features = ["lap_number", 'lap_time','remaining_laps',"accx_can", "accy_can", "Steering_Angle", "pbrake_f", "speed","incident","tyre_age"]
    first_lap_model3=first_lap[features]
    X_scaled = scaler_model3.transform(first_lap_model3)

    first_lap_model3 = pd.DataFrame(X_scaled, columns=features)
    expected_gain_if_pit_now=model3.predict(first_lap_model3)
    first_lap["expected_gain_if_pit_now"]=expected_gain_if_pit_now
     
    print(first_lap_model3)
    print(first_lap["incident"])
    row = first_lap.iloc[0]  # <- récupérer la ligne
    print(round(float(row["lap_time"]),2))
    return JSONResponse(
      status_code=200,
      content={
        "lapTime": round(float(row["lap_time"]),2),
        "lapTimeGain": round(float(row["loss_per_lap"]),2),
        "speed": round(float(row["speed"]),2),
        "tyreAge": int(row["tyre_age"]),
        "expectedGain": round(float(row["expected_gain_if_pit_now"]),2),
        "incidentRisk": round(float(row["incident"]), 2)
      }
    ) 
