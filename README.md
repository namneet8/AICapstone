# Setup Instructions

## Prerequisites
- Node.js and npm installed
- MongoDB Atlas account configured
- ngrok installed (for Android app testing)

## Getting Started

1. Install dependencies with `npm install`
2. Configure your `.env` file with `MONGO_URI` and other required variables
3. Start the server with `npm run dev` (runs on port 3000 by default)

## Testing with Android App

1. Run `ngrok http 3000` to expose your local server
2. Copy the generated ngrok URL (e.g., `https://abcd1234.ngrok.io`)
3. Update the URL in your Android app's `NetworkModule`

> **Note:** The ngrok URL changes on every restart and needs to be updated in the app accordingly.

## Important
- Whitelist your IP in MongoDB Atlas
- Never commit your `.env` file to the repository



<img width="1295" height="971" alt="image" src="https://github.com/user-attachments/assets/a286f8d1-707d-4446-a857-6be8657bd763" />

<br><br>
<img width="1609" height="839" alt="image" src="https://github.com/user-attachments/assets/ef814775-ccfa-4e47-a69e-f1397d891fa4" />
<br><br>

<h2 align="center">AI Model Architecture</h1>
<img width="1995" height="5428" alt="AI-Architecture-audio_classification_diagram" src="https://github.com/user-attachments/assets/4b13bc9a-cc88-4463-b1da-1bb73bc3217f" />

<h2 align="center">Android - MVVM Diagram</h1>
<img width="3715" height="1468" alt="MVVM_diagram" src="https://github.com/user-attachments/assets/d252ef1d-a4b3-49f6-bcfd-564b61bf5e9b" />

<h2 align="center">Backend Architecture</h1>
<img width="793" height="829" alt="image" src="https://github.com/user-attachments/assets/365fd5c3-8e6b-4b02-87ee-2eeb8801300e" />


