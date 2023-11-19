from flask import Flask, request, jsonify
import pickle
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

with open('model/clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)


with open('model/tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

@app.route('/predict_category', methods=['POST'])
def predict_category():
    data = request.files['file']
    pdf = PdfReader(data)
    
    resume_text = ''
    for page in pdf.pages:
        resume_text += page.extract_text()

    # Clean the input resume
    cleaned_resume = cleanResume(resume_text)

    # Transform the cleaned resume using the trained TfidfVectorizer
    input_features = tfidf.transform([cleaned_resume])

    # Make the prediction using the loaded classifier
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")

    return jsonify({'predicted_category': category_name})

if __name__ == '__main__':
    app.run(debug=True)
