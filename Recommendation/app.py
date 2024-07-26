from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from models import db, Course
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import difflib
import os

# Initialize Flask app
app = Flask(__name__)

# Path to the SQLite database used by Django
basedir = os.path.abspath(os.path.dirname(__file__))
django_db_path = os.path.join(basedir, r'C:\Users\User\OneDrive\Desktop\proj\E-Learning-Platform\db.sqlite3')

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{django_db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize global variables
vectorizer = None
tfidf_matrix = None
cosine_sim = None
df_global = None

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_courses():
    global vectorizer, tfidf_matrix, cosine_sim, df_global
    # Preprocess course descriptions
    courses = Course.query.all()
    df = pd.DataFrame([{'id': course.id, 'title': course.title, 'overview': course.overview} for course in courses])
    
    df['processed_description'] = df['overview'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df_global = df


def get_closest_title(search_title, course_titles):
    closest_titles = difflib.get_close_matches(search_title, course_titles, n=1, cutoff=0.1)
    return closest_titles[0] if closest_titles else None

def get_recommendations_by_search_title(search_title):
    global df_global, cosine_sim
    course_titles = df_global['title'].tolist()
    matched_title = get_closest_title(search_title, course_titles)
    if matched_title is None:
        return "Course title not found."
    idx = df_global.index[df_global['title'] == matched_title].tolist()
    if not idx:
        return "Course title not found."
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:5]
    course_indices = [i[0] for i in sim_scores]
    return matched_title, df_global.iloc[course_indices][['id', 'title', 'overview']]


@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <title>Course Recommendation</title>
    <h1>Course Recommendation System</h1>
    <form action="/recommend" method="get">
        <label for="title">Enter Course Title:</label>
        <input type="text" id="title" name="title">
        <input type="submit" value="Get Recommendations">
    </form>
    ''')

@app.route('/recommend', methods=['GET'])
def recommend():
    search_title = request.args.get('title')
    if not search_title:
        return jsonify({'error': 'Missing "title" parameter'}), 400
    matched_title, recommended_courses = get_recommendations_by_search_title(search_title)
    if matched_title == "Course title not found.":
        return jsonify({'error': matched_title}), 404
    return jsonify({
        'searched_title': search_title,
        'matched_title': matched_title,
        'recommendations': recommended_courses.to_dict(orient='records')
    })

if __name__ == '__main__':
    with app.app_context():
        preprocess_courses()  # Preprocess the courses data before starting the server
    app.run(debug=True)
