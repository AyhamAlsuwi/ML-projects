# Intelligent IT Ticket Routing API

This document houses the complete, deployed, production-ready microservice built to act as an automated dispatcher for your IT Helpdesk. It accepts natural language support tickets, intercepts garbage or vague inputs, and returns a definitive, un-black-boxed department routing decision. You don't need to understand the underlying Logistic Regression architecture or TF-IDF math to use this; you just need to know how to send a POST request.

 *[Interact with the Live API Swagger UI](https://ayham-zeyad-it-ticket-routing-model.hf.space/docs#/)*

---

## The Mechanism

This API is designed to sit between your frontend ticketing portal (or email parser) and your IT management database (like ServiceNow or Jira). Here is exactly what happens when you hit the endpoint:

### 1. The Garbage & Noise Filter
Users frequently submit hopelessly vague tickets (e.g., "it is broken") or total keyboard smashes. If passed directly to an automated routing system, these contaminate the queues of actual IT departments. Before any mathematical routing happens, this API runs two strict safety nets:
* **The Zero-Vector Catcher:** Scans the ticket for known IT vocabulary. If no recognized words exist, it forcefully halts the math and flags the ticket.
* **The Noise Floor:** Evaluates the model's confidence. If the highest probability falls below the strict `0.24` baseline, the API recognizes the model is making a blind guess and intercepts it.

### 2. Department Classification
If a ticket passes the noise filters, the API applies a custom-scaled evaluation to route the issue to one of the following exact categories:
* `Access`
* `Administrative rights` *(Note: The API specifically boosts sensitivity for this hard-to-catch class)*
* `HR Support`
* `Hardware`
* `Internal Project`
* `Purchase`
* `Storage`
* `Miscellaneous` *(Used exclusively as the fallback for intercepted garbage/noise)*

---

## How to Use It

### The Endpoint
Make a `POST` request to the `/predict` route.

The API expects a highly simplified JSON payload containing only the raw text of the ticket. The model handles all text flattening, cleaning, and vectorization on the server side.

**Example Request Body:**
```json
{
  "document": "Hello, I need to change my Oracle timesheet approval format, please help."
}
