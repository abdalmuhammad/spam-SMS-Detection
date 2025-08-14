# 📩 SPAM SMS Detection Web App

A Flask-based machine learning web application to detect whether an SMS is **Spam** or **Not Spam**.  
The app uses Natural Language Processing (NLP) techniques and a trained ML model to classify incoming text messages.

---

## 🚀 Features
- **Real-time SMS classification**
- **Clean, responsive UI** with HTML/CSS
- **Flask backend** for serving predictions
- **Trained ML model** for spam detection
- **Lightweight and fast** deployment

---

## 📸 Screenshots

### 🔹 Home Page
![Home Page Screenshot](static/images/homepage.png)

## 🛠 Installation & Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/YourUsername/spam-sms-detection.git
cd spam-sms-detection

# 2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate   # For Windows
# source venv/bin/activate  # For Mac/Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Flask app
python app.py
