# Creating sqlite DB and loading raw datasets + exploding genres

import sqlite3
import pandas as pd
from pathlib import Path

#Connect to database
db_path = Path("movie_ratings.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Connected to:", db_path.resolve())

#Got help from lab 4
def run_sql(sql, params=None, show=True):
    if params is None:
        params = ()
    try:
        cursor.execute(sql, params)
        conn.commit()

        if show and sql.lstrip().lower().startswith("select"):
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            
            try:
                from IPython.display import display
                display(pd.DataFrame(rows, columns=cols))
            except ImportError:
                print(pd.DataFrame(rows, columns=cols))
    except Exception as e:
        print("Error:", e)
        print("SQL was:\n", sql)


#Load CSVs into SQL tables 
movies_path = Path("datasets/movie.csv")
ratings_path = Path("datasets/rating.csv")

df_movies = pd.read_csv(movies_path)
print(df_movies.isna().sum())
df_movies = df_movies.drop_duplicates()

df_movies.to_sql("MoviesRaw", conn, if_exists="replace", index=False)
print("MoviesRaw:", len(df_movies), "rows")

run_sql("SELECT * FROM MoviesRaw LIMIT 5;")

#Ratings chunked loading
chunksize = 1_000_000
first = True

for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
    chunk.to_sql("RatingsRaw", conn, if_exists="replace" if first else "append", index=False)
    first = False
    print("Loaded another chunk of size:", len(chunk))

print("Finished loading RatingsRaw")

#Create an index on movieId so lookups and GROUP BY on this big ratings table run faster
run_sql("CREATE INDEX IF NOT EXISTS idx_ratings_movieId ON RatingsRaw(movieId);", show=False)

#Ratings for movieId = 1, in time order
run_sql("""
SELECT *
FROM RatingsRaw
WHERE movieId = 1
ORDER BY timestamp
LIMIT 10;
""")

#Checks
#Check if duplicate movieIds in MoviesRaw
run_sql("""
SELECT movieId, COUNT(*) AS m
FROM MoviesRaw
GROUP BY movieId
HAVING m > 1
LIMIT 10;
""")

#Look at some raw titles + genres
run_sql("""
SELECT movieId, title, genres
FROM MoviesRaw
LIMIT 10;
""")

#Aggregate all ratings for each movie: how many ratings and average rating
run_sql("""
CREATE TABLE IF NOT EXISTS MovieRatings AS
SELECT
    r.movieId,
    COUNT(*) AS num_ratings,
    AVG(r.rating) AS avg_rating
FROM RatingsRaw r
GROUP BY r.movieId;
""", show=False)

run_sql("SELECT * FROM MovieRatings LIMIT 10;")

#Combine movie metadata with aggregated ratings and remove the release year out of the title
#Filter out bad years if there (year < 1900 OR > 2100)
#Year extraction help from cgpt
run_sql("""
CREATE TABLE IF NOT EXISTS MovieRatingsFull AS
SELECT
    m.movieId,
    m.title,
    m.genres,
    CAST(substr(m.title, length(m.title)-4, 4) AS INTEGER) AS year, 
    mr.num_ratings,
    mr.avg_rating
FROM MoviesRaw m
JOIN MovieRatings mr ON m.movieId = mr.movieId;
""", show=False)

#Inspect a few movies with sus years 
run_sql("""
SELECT movieId, title, year
FROM MovieRatingsFull
WHERE year < 1900 OR year > 2100 OR year IS NULL
LIMIT 20;
""")

#Cleaned movie table
#Keep movies with enough ratings (num_ratings >= 50)
#Focus on a time window (1980–2024)
#Create a binary label high_rating = 1 if avg_rating >= 4.0
run_sql("DROP TABLE IF EXISTS MovieRatingsClean;", show=False)

run_sql("""
CREATE TABLE MovieRatingsClean AS
SELECT
    movieId,
    title,
    genres,
    year,
    num_ratings,
    avg_rating,
    CASE WHEN avg_rating >= 4.0 THEN 1 ELSE 0 END AS high_rating
FROM MovieRatingsFull
WHERE num_ratings >= 50
  AND year BETWEEN 1980 AND 2024;
""", show=False)

run_sql("SELECT * FROM MovieRatingsClean LIMIT 10;")

#See how movie average ratings are distributed
run_sql("""
SELECT
    ROUND(avg_rating, 1) AS rating_round,
    COUNT(*) AS n_movies
FROM MovieRatingsClean
GROUP BY rating_round
ORDER BY rating_round;
""")

#Turn multi-genre strings like "Action|Comedy" into one row per (movieId, genre)
df_clean = pd.read_sql_query("""
    SELECT movieId, title, genres, year, num_ratings, avg_rating, high_rating
    FROM MovieRatingsClean;
""", conn)

#Help from cgpt with explode
df_genre = (df_clean.assign(genre=df_clean["genres"].str.split("|")).explode("genre"))

#Save back into SQLite
df_genre.to_sql("MovieGenreRatings", conn, if_exists="replace", index=False)
print("MovieGenreRatings", len(df_genre), "rows")

#Check a sample of the movie–genre rows
run_sql("SELECT * FROM MovieGenreRatings LIMIT 10;")

#Summarize each genre:
#n_movies is how many movies have this genre
#avg_movie_rating is average avg_rating for those movies
#avg_num_ratings is number of ratings
#share_high_rating is fraction of movies that are high_rating = 1

run_sql("""
CREATE TABLE IF NOT EXISTS MovieGenreSummary AS
SELECT
    genre,
    COUNT(DISTINCT movieId) AS n_movies,
    AVG(avg_rating) AS avg_movie_rating,
    AVG(num_ratings) AS avg_num_ratings,
    AVG(high_rating) AS share_high_rating
FROM MovieGenreRatings
GROUP BY genre
HAVING n_movies >= 20;
""", show=False)

#View genres ordered from highest to lowest average movie rating
run_sql("""
SELECT *
FROM MovieGenreSummary
ORDER BY avg_movie_rating DESC;
""")

# conn.close()

df = pd.read_sql_query("""
    SELECT movieId, title, genres, year, num_ratings, avg_rating
    FROM MovieRatingsClean
""", conn)

'''
#Create a binary label which is 1 if the movie's average rating is at least 4.0 if not it is 0
df["high_rating"] = (df["avg_rating"] >= 4.0).astype(int)

genre_dummies = df["genres"].str.get_dummies(sep="|")

import numpy as np

#Build the matrix
X = pd.concat([df[["year", "num_ratings"]], genre_dummies], axis=1)
y = df["high_rating"].values

#Train on older movies (< 2015) and test on newer ones (>= 2015)
train = df["year"] < 2015
test = df["year"] >= 2015
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Train high_rating rate:", y_train.mean())
print("Test high_rating rate:", y_test.mean())
'''

import matplotlib.pyplot as plt

img_dir = Path("images")
img_dir.mkdir(exist_ok=True)

#Histogram of average rating from cleaned movie table
ratings_df = pd.read_sql_query("""
    SELECT avg_rating
    FROM MovieRatingsClean;
""", conn)

plt.figure(figsize=(10, 7))
plt.hist(ratings_df["avg_rating"], bins=50)
plt.xlabel("Average rating")
plt.ylabel("Number of movies")
plt.title("Histogram of Average Movie Ratings")
plt.tight_layout()
plt.savefig(img_dir / "rating_hist.png")
plt.show()
plt.close()

#Bar chart of average rating by genre
genre_summary = pd.read_sql_query("""
    SELECT genre, n_movies, avg_movie_rating, share_high_rating
    FROM MovieGenreSummary
    ORDER BY avg_movie_rating DESC;
""", conn)

plt.figure(figsize=(10, 20))
plt.bar(genre_summary["genre"], genre_summary["avg_movie_rating"])
plt.xlabel("Genre")
plt.ylabel("Average movie rating")
plt.title("Average Rating by Genre")
plt.show()
plt.tight_layout()
plt.savefig(img_dir / "genre_avg_rating.png")  
plt.show()
plt.close()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:",       f1_score(y_test, y_pred))

import matplotlib.pyplot as plt

df_clean["avg_rating"].hist(bins=20)
plt.title("Distribution of Average Movie Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Count of Movies")
plt.show()

import joblib
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "x_columns.pkl")

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
X_columns = joblib.load("x_columns.pkl")

st.title("Movie Rating Predictor")

year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2005)
num_ratings = st.number_input("Number of Ratings", min_value=0, max_value=500000, value=500)

genres_list = ['Action','Adventure','Animation','Drama','Comedy','Thriller','Sci-Fi','Romance']
genres_selected = st.multiselect("Select Genres", genres_list)

row = {col: 0 for col in X_columns}
row["year"] = year
row["num_ratings"] = num_ratings

for g in genres_selected:
    if g in row:
        row[g] = 1

row_df = pd.DataFrame([row])

if st.button("Predict Rating"):
    pred = model.predict(row_df)[0]
    label = "High Rating" if pred == 1 else "Not Highly Rated"
    st.subheader(label)


## Predict Popularity

import csv

def calculate_popularity(row):
    try:
        views = float(row["views"])
        likes = float(row["likes"])
        comments = float(row["comments"])
    except:
        return None

    popularity_score = (views * 0.5) + (likes * 3) + (comments * 4)
    return popularity_score

data = []

with open("data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        score = calculate_popularity(row)
        if score is not None:
            row["popularity_score"] = score
            data.append(row)

data_sorted = sorted(data, key=lambda x: x["popularity_score"], reverse=True)

for i, row in enumerate(data_sorted[:10], 1):
    print(f"{i}. {row['title']} — Popularity Score: {row['popularity_score']:.2f}")


#Filip maybe something like this for the predicition UI - got it from chatgpt so please double check
# -------------------------------
# 5. Very simple text-based "UI" for prediction
# -------------------------------

def predict_movie_interactive(model, feature_columns):
    """
    Simple text-based UI:
    asks for year, num_ratings, and genres,
    then prints the predicted probability and label.
    """
    print("=== Movie Rating Predictor ===")
    title = input("Movie title (optional, for display only): ")

    year = int(input("Year (e.g. 2010): "))
    num_ratings = int(input("Number of ratings (e.g. 1500): "))

    genres_input = input("Genres separated by | (e.g. Action|Sci-Fi): ")
    genres_list = [g.strip() for g in genres_input.split("|") if g.strip()]

    # Start with a row of zeros for all feature columns
    row = {col: 0 for col in feature_columns}

    # Set numeric features
    if "year" in row:
        row["year"] = year
    if "num_ratings" in row:
        row["num_ratings"] = num_ratings

    # Set genre indicator columns to 1 if they are selected
    for g in genres_list:
        if g in row:
            row[g] = 1

    # Build DataFrame with the same columns as X
    X_new = pd.DataFrame([row], columns=feature_columns)

    # Predict probability and label
    prob_high = model.predict_proba(X_new)[0, 1]
    label = int(prob_high >= 0.5)

    print("\nPrediction for:", title or "(untitled movie)")
    print("  Estimated probability of high rating (avg >= 4.0):", round(prob_high, 3))
    print("  Predicted label (1 = high-rated, 0 = not):", label)
    print()


# Call this once if you want to try the UI in the terminal / console
# (You can comment this out when you don't want to interact.)
predict_movie_interactive(model, X.columns.tolist())
