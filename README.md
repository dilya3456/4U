MedAI – AI-Powered Clinical Diagnosis System

MedAI is an advanced clinical intelligence platform that uses state-of-the-art AI and machine learning algorithms to assist medical professionals in diagnosing symptoms. By leveraging a hybrid approach combining TF-IDF (word + character) vectorization and clinical protocols, MedAI offers a highly accurate, fast, and reliable symptom-to-diagnosis system, providing the top ICD-10 hypotheses with evidence from clinical sources.

Key Features

AI-Powered Analysis: MedAI uses a Hybrid TF-IDF model to analyze input symptoms with both word and character-level vectorization.

Top ICD-10 Diagnoses: Get the top three possible diagnoses with detailed explanations, including ICD-10 codes and evidence from relevant clinical protocols.

User-Friendly Interface: A simple, intuitive, and interactive UI that allows quick symptom input and provides results in a visually appealing format.

FastAPI Backend: Robust and high-performance backend for handling requests in real time.

No External Network Calls: All computations are done locally, ensuring privacy and reliability.

Real-Time Diagnosis: The system is designed to return results instantly based on the symptoms provided.

Technologies Used

Backend: FastAPI, Uvicorn

Machine Learning: Scikit-learn (TF-IDF + Cosine Similarity)

Frontend: HTML, CSS, JavaScript (Responsive, mobile-friendly)

Containerization: Docker

Database: Local file-based storage of clinical protocols (no need for an external DB)

Requirements

Python 3.12+

Docker (optional for containerization)

Git for version control

Installation Guide
1. Clone the Repository

Start by cloning the project to your local machine:

git clone https://github.com/dilya3456/4U.git
2. Set up Python Environment

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate     # For Windows
3. Install Dependencies

Install the required Python dependencies:

pip install -r requirements.txt
4. Running the Application Locally

Run the FastAPI server:

uvicorn src.mock_server:app --host 0.0.0.0 --port 8000 --reload

Now, open your browser and visit:
http://127.0.0.1:8000

5. Using Docker (Optional)

Build the Docker Image:

docker build -t submission .

Run the Docker Container:

docker run -p 8080:8080 submission

Visit your app in the browser: http://127.0.0.1:8080

How It Works

Input Symptoms: Enter a list of symptoms (e.g., "fever, headache, dizziness") into the input box.

Diagnose: MedAI processes the symptoms using a hybrid machine learning model to identify potential diagnoses.

Top Diagnoses: The system returns the top 3 hypotheses, including their corresponding ICD-10 codes, explanations, and clinical evidence.

Example Input:

"fever, headache, dizziness"

Example Output:

Rank 1: Hypothesis: G43.0 (Migraine with Aura)

Explanation: "Protocol: p_97a12ac54. Evidence: The patient has a history of migraines with aura. Symptoms include..."

Contributions

We welcome contributions! Here’s how you can help:

Fork the repository

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -am 'Add new feature')

Push to the branch (git push origin feature-name)

Create a new pull request

Why MedAI?

Accurate and Fast: MedAI offers fast, reliable diagnosis predictions based on real clinical protocols.

No External Dependencies: Everything is handled internally for faster processing and increased security.

Easy to Use: Simple web interface that provides actionable insights with just a few clicks.

License

This project is licensed under the MIT License – see the LICENSE
 file for details.