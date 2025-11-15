# Setup Instructions

## Prerequisites
- Node.js and npm installed
- MongoDB Atlas account configured

## Getting Started

1. Install dependencies with `npm install`
2. Configure your `.env` file with `MONGO_URI` and other required variables
3. Start the server with `npm run dev` (runs on port 3000 by default)

## Using the Android App

- The Android app can be run from Android Studio or installed on a device.  
- Make sure to update the base URL in the app's `NetworkModule` to point to your backend server:
  - Local development: `http://YOUR_LOCAL_IP:3000`
  - Deployed server: `https://your-server-url.com`

> **Note:** The server can run locally or be deployed to any hosting provider. The URL depends on where you run the backend.

## Important
- Whitelist your server or client IP in MongoDB Atlas
- Never commit your `.env` file to the repository

##

<img width="1295" height="971" alt="image" src="https://github.com/user-attachments/assets/a286f8d1-707d-4446-a857-6be8657bd763" />

##
<br><br>
<img width="1609" height="839" alt="image" src="https://github.com/user-attachments/assets/ef814775-ccfa-4e47-a69e-f1397d891fa4" />
<br><br>

##

<h2 align="center">AI Model Architecture</h1>
<img width="1995" height="5428" alt="AI-Architecture-audio_classification_diagram" src="https://github.com/user-attachments/assets/4b13bc9a-cc88-4463-b1da-1bb73bc3217f" />

##

<h2 align="center">Android - MVVM Diagram</h1>
<img width="3715" height="1468" alt="MVVM_diagram" src="https://github.com/user-attachments/assets/d252ef1d-a4b3-49f6-bcfd-564b61bf5e9b" />

##

<h2 align="center">Backend Architecture</h1>
<img width="1809" height="1492" alt="mermaid-diagram-2025-11-12-150521" src="https://github.com/user-attachments/assets/cb1c8045-b219-4ce2-8d03-9bd185b318ef" />



