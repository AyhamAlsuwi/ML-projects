
# USA Used Car Price Prediction API

## Overview
I deployed the core XGBoost prediction pipeline as a fully containerized REST API using FastAPI. This allows external applications to instantly query the model and receive real-time auction price predictions.

👉 **[Access the Live API & Interactive Swagger UI Here](https://ayham-zeyad-usa-used-cars-price-prediction.hf.space/docs)**

## How It Works
The API is built with FastAPI and uses Pydantic for strict data validation. It accepts a JSON payload containing 11 specific car features, passes them through the custom Scikit-Learn preprocessing pipeline, and executes the XGBoost regression model.

It returns the predicted US Dollar value rounded to two decimal places.

## API Endpoint
`POST /predict`

## Expected Input Payload (JSON)
The endpoint requires the following schema. Missing categorical fields will be handled by the pipeline's internal imputers, but providing accurate data yields the best predictions.

```json
{
  "saledate": "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)",
  "transmission": "automatic",
  "color": "black",
  "interior": "black",
  "state": "ca",
  "body": "sedan",
  "model": "camry",
  "trim": "se",
  "year": 2015,
  "condition": 3.5,
  "odometer": 50000
}
```

## Response (JSON)
The API strips out unnecessary metadata and returns exactly what a frontend application or business intelligence tool needs: the price.

```json
{
  "predicted_price_usd": 13474.69
}
```

## Testing the API Locally via CLI
You can test the live Hugging Face deployment directly from your terminal using `curl`:

```bash
curl -X 'POST' \
  '[https://ayham-zeyad-usa-used-cars-price-prediction.hf.space/predict](https://ayham-zeyad-usa-used-cars-price-prediction.hf.space/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "saledate": "Tue Dec 16 2014",
  "transmission": "automatic",
  "color": "white",
  "interior": "black",
  "state": "tx",
  "body": "suv",
  "model": "f-150",
  "trim": "base",
  "year": 2018,
  "condition": 4.0,
  "odometer": 35000
}'
```
