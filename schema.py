from pydantic import BaseModel

class InputData1(BaseModel):
    lap_number: int
    lap_time: float
    accx_can: float
    Steering_Angle:float
    accy_can:float
    pbrake_f:float
    pbrake_r:float
    tyre_age:int

