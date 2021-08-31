from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def index():
    #load a ML model
    #model.predict
    
    return {"ok":"whatever"}

@app.get("/predict")
def predict(day_of_week, time):
    # compute `wait_prediction` from `day_of_week` and `time`
    return {'wait': int(day_of_week) * int(time)}

