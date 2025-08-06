from flask import Flask, render_template, request
from recommender import recommend_movies

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        try:
            recommendations = recommend_movies(user_id)
        except:
            recommendations = ["Invalid User ID or No Recommendations"]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
