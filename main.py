from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib

# Create FastAPI App
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using XGBoost",
    version="1.0"
)

# Load Model
model = joblib.load("churn_model.pkl")

# Templates & Static Files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Home Page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "probability": None
        }
    )


# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Model Information
@app.get("/model/info")
def model_info():
    return {
        "model": "XGBoost",
        "version": "1.0",
        "features": [
            "AccountAge",
            "MonthlyCharges",
            "ViewingHoursPerWeek",
            "UserRating",
            "SupportTicketsPerMonth",
            "SubscriptionType",
            "MultiDeviceAccess",
            "PaymentMethod",
            "WatchlistSize",
            "GenrePreference"
        ]
    }


# Prediction Endpoint
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,

    AccountAge: float = Form(...),
    MonthlyCharges: float = Form(...),
    ViewingHoursPerWeek: float = Form(...),
    UserRating: float = Form(...),
    SupportTicketsPerMonth: float = Form(...),

    SubscriptionType: str = Form(...),
    MultiDeviceAccess: str = Form(...),
    PaymentMethod: str = Form(...),

    WatchlistSize: int = Form(...),

    GenrePreference: str = Form(...)
):

    try:

        input_data = pd.DataFrame([{
            "AccountAge": AccountAge,
            "MonthlyCharges": MonthlyCharges,
            "ViewingHoursPerWeek": ViewingHoursPerWeek,
            "UserRating": UserRating,
            "SupportTicketsPerMonth": SupportTicketsPerMonth,
            "SubscriptionType": SubscriptionType,
            "MultiDeviceAccess": MultiDeviceAccess,
            "PaymentMethod": PaymentMethod,
            "WatchlistSize": WatchlistSize,
            "GenrePreference": GenrePreference
        }])

        print("\n========== INPUT DATA ==========")
        print(input_data)
        print("================================\n")

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        result = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": result,
                "probability": round(probability * 100, 2)
            }
        )

    except Exception as e:

        print("\n========== ERROR ==========")
        print(type(e))
        print(str(e))
        print("===========================\n")

        return HTMLResponse(
            content=f"""
            <html>
            <body style="font-family:Arial;padding:30px;">
                <h2>Prediction Error</h2>
                <p>{str(e)}</p>
            </body>
            </html>
            """,
            status_code=500
        )