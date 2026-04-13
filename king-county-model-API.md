# API Documentation: King County Housing Predictor

This document outlines the technical specifications for interacting with the live machine learning API.

* **🚀 Live Base URL:** `https://ayham-zeyad-king-county-housing-price-prediction.hf.space`
* **📚 Interactive Swagger UI:** [View Docs](https://ayham-zeyad-king-county-housing-price-prediction.hf.space/docs)

---

## Endpoint: Predict Price
Returns the predicted price of a house based on its features.

* **Path:** `/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`

### Request Payload Schema
The API strictly validates incoming JSON payloads using **Pydantic**. All 17 fields are required (though defaults are provided in the Swagger UI if left blank).

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `date` | String | "20141013T000000" | The date the house was sold. |
| `bedrooms` | Integer | 3 | Number of bedrooms. |
| `bathrooms` | Float | 2.0 | Number of bathrooms. |
| `sqft_living` | Integer | 2000 | Square footage of the home. |
| `sqft_lot` | Integer | 5000 | Square footage of the lot. |
| `floors` | Float | 1.0 | Number of floors. |
| `waterfront` | Integer | 0 | 1 if waterfront, 0 if not. |
| `view` | Integer | 0 | Rating from 0 to 4. |
| `condition` | Integer | 3 | Rating from 1 to 5. |
| `sqft_above` | Integer | 1500 | Square footage apart from basement. |
| `sqft_basement` | Integer | 0 | Square footage of the basement. |
| `yr_built` | Integer | 1990 | Year the house was built. |
| `yr_renovated` | Integer | 0 | Year the house was renovated (0 if never). |
| `lat` | Float | 47.5 | Latitude coordinate. |
| `long` | Float | -122.2 | Longitude coordinate. |
| `sqft_living15` | Integer | 1500 | Living room area in 2015 (implies renovations). |
| `sqft_lot15` | Integer | 5000 | Lot area in 2015 (implies renovations). |

### Success Response
**Code:** `200 OK`

```json
{
  "predicted_price": 415159.93
}
