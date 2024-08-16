from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

#Loading the trained model and vectorizer
model = joblib.load('spam_filter.pkl')
cv = joblib.load('count_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def main_function():
    if request.method == 'POST':
        text = request.form['email']
        email_list = [text]
        email_count = cv.transform(email_list)
        prediction = model.predict(email_count)[0]

        return render_template("result.html", prediction = prediction)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
