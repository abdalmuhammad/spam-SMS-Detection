from flask import Flask, render_template, request
import pickle

# Flask app init
app = Flask(__name__)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Transform message using TF-IDF
        data = tfidf.transform([message])
        # Predict (0 = Ham, 1 = Spam)
        prediction = model.predict(data)[0]
        
        result = "Spam" if prediction == 1 else "Ham (Not Spam)"
        
        return render_template('index.html', prediction_text=f'Message is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
