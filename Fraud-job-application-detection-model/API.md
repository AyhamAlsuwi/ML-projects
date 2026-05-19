
# Trust & Safety: Fraudulent Job Detection API


This document houses the complete deployed production-ready microservice built to act as a Trust & Safety firewall for job boards. It accepts messy JSON payloads,  sanitizes the data, and returns a definitive, un-black-boxed business decision. You don't need to understand the underlying ML architecture to use this; you just need to know how to send a POST request.

**[Interact with the Live API (Swagger UI)](https://ayham-zeyad-fraud-job-application-detection.hf.space/docs)**

---

## The Mechanism

This API is designed to sit between your frontend submission form and your database. Here is exactly what happens when you hit the endpoint:

### 1. Payload Sanitization
Web frontends are notorious for sending empty strings (`""`) when a user leaves a field blank. If passed directly to an automated system, these empty strings cause massive data leakage (the system thinks data is present when it isn't). 
Before any evaluation happens, this API intercepts the payload, runs a matrix scan, and mathematically destroys all empty or whitespace-only strings, converting them into true `null` values to ensure strict evaluation integrity.

### 2. Action Routing
A raw probability score (e.g., `0.65`) is useless to a backend developer. Instead of forcing your application layer to figure out what a score means, this API routes the calculated risk through strict Trust & Safety business thresholds:
* **CRITICAL:** The job is highly toxic. Action: `DELETE_ACCOUNT`.
* **HIGH:** The job is suspicious. Action: `SEND_TO_MODERATION_QUEUE`.
* **ELEVATED:** The job has missing or strange data. Action: `FLAG_FOR_SECONDARY_SCAN`.
* **LOW:** The job looks clean. Action: `PUBLISH_JOB`.

---

## How to Use It

### The Endpoint
Make a `POST` request to the `/predict` route. 

The API expects a standard JSON payload representing a job posting. All fields are technically optional, but the more data you provide, the more accurate the firewall becomes.

**Example Request Body:**
```json
{
  "title": "URGENT: Work From Home Data Entry Clerk",
  "location": "US, CA, Los Angeles",
  "department": "Admin",
  "salary_range": "50000 - 150000",
  "company_profile": "",
  "description": "We are looking for immediate hires. Cash paid weekly. Click the link to apply on Telegram!",
  "requirements": "Must have a valid bank account to receive company funds for home office setup.",
  "telecommuting": 1,
  "has_company_logo": 0,
  "has_questions": 0,
  "employment_type": "Full-time"
}
```

### What It Returns
Instead of hiding the logic, the API returns a comprehensive diagnostic JSON payload. It tells your backend exactly what action to take and exposes the threshold switches that were triggered.

**Example Response:**
```json
{
  "fraud_probability": 0.6500,
  "risk_level": "HIGH",
  "recommended_action": "SEND_TO_MODERATION_QUEUE",
  "threshold_diagnostics": {
    "triggered_auto_ban": false,
    "triggered_human_review": true,
    "triggered_initial_filter": true
  }
}
```

**Response Breakdown:**
* `fraud_probability`: The raw 0-to-1 risk score.
* `risk_level`: The severity tier (`LOW`, `ELEVATED`, `HIGH`, `CRITICAL`).
* `recommended_action`: The exact operation your backend should execute.
* `threshold_diagnostics`: A boolean breakdown showing exactly *why* it recommended that action, allowing your frontend to display clear error messages if needed.

---

## Running Locally

To run this inference server on your own machine or internal network:

1. Clone the repository.
2. Build the Docker container:
   ```bash
   docker build -t fraud-api .
   ```
3. Run the container, mapping port 7860 to your local machine:
   ```bash
   docker run -p 7860:7860 fraud-api
   ```
4. Visit `http://localhost:7860/docs` to access the interactive Swagger dashboard.
```
