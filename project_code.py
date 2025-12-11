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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

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


# Movie Similarity Engine (using cosine similarity)

from sklearn.metrics.pairwise import cosine_similarity

movie_vectors = X  
similarity = cosine_similarity(movie_vectors)

def recommend_movie(movie_id, k=5):
    if movie_id not in df_clean["movieId"].values:
        print("Invalid movieId")
        return

    idx = df_clean.index[df_clean["movieId"] == movie_id][0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]

    print(f"\nMovies similar to: {df_clean.loc[idx, 'title']}\n")
    for i, score in sim_scores:
        print(f"{df_clean.loc[i, 'title']} (score={score:.3f})")



#Multi Page UI

import streamlit as st
import pandas as pd
import sqlite3
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import openai   
import os

conn = sqlite3.connect("movie_ratings.db")

df = pd.read_sql_query("""
    SELECT movieId, title, genres, year, num_ratings, avg_rating, high_rating
    FROM MovieRatingsClean
""", conn)

X_columns = joblib.load("x_columns.pkl")
model = joblib.load("model.pkl")

genre_dummies = df["genres"].str.get_dummies(sep="|")
X = pd.concat([df[["year", "num_ratings"]], genre_dummies], axis=1)

similarity = cosine_similarity(X)
df = df.reset_index(drop=True)

st.title("Movie Data Science Hub")

page = st.sidebar.selectbox("Navigation", [
    "High Rating Predictor",
    "Movie Recommender",
    "Genre Statistics",
    "SQL Explorer",
    "Ask an LLM About Your Movie",
    "Model Performance"
])

if page == "High Rating Predictor":
    st.header("Predict Whether a Movie Will Be Highly Rated")

    year = st.number_input("Release Year", 1980, 2025, 2005)
    num_ratings = st.number_input("Number of Ratings", 0, 500000, 500)

    genres = list(genre_dummies.columns)
    selected = st.multiselect("Genres", genres)

    row = {col: 0 for col in X_columns}
    row["year"] = year
    row["num_ratings"] = num_ratings

    for g in selected:
        if g in row:
            row[g] = 1

    row_df = pd.DataFrame([row])

    if st.button("Predict"):
        prob = model.predict_proba(row_df)[0, 1]
        label = "Highly Rated" if prob >= 0.5 else "Not Highly Rated"
        st.subheader(label)
        st.write("Probability:", round(prob, 3))

elif page == "Movie Recommender":
    st.header("Find Similar Movies")

    movie_names = df["title"].tolist()
    choice = st.selectbox("Select a Movie", movie_names)

    if st.button("Recommend"):
        idx = df.index[df["title"] == choice][0]
        sims = list(enumerate(similarity[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:6]

        st.subheader("Similar Movies:")
        for i, s in sims:
            st.write(f"{df.loc[i, 'title']} — similarity: {s:.3f}")

elif page == "Genre Statistics":
    st.header("Genre Rating Statistics")

    genre_summary = pd.read_sql_query("""
        SELECT genre, n_movies, avg_movie_rating, share_high_rating
        FROM MovieGenreSummary
        ORDER BY avg_movie_rating DESC
    """, conn)

    st.dataframe(genre_summary)

    st.bar_chart(
        genre_summary.set_index("genre")["avg_movie_rating"]
    )

elif page == "SQL Explorer":
    st.header("Run SQL on the Movie Database")

    user_sql = st.text_area("Write SQL here:", "SELECT * FROM MovieRatingsClean LIMIT 10;")

    if st.button("Execute"):
        try:
            result = pd.read_sql_query(user_sql, conn)
            st.dataframe(result)
        except Exception as e:
            st.error(f"SQL Error: {e}")

elif page == "Ask an LLM About Your Movie":
    st.header("LLM Movie Summary & Insights")

    openai.api_key = st.text_input("Enter OpenAI API Key:", type="password")

    selected_movie = st.selectbox("Choose a movie:", df["title"])

    prompt = f"""
    Provide a thoughtful analysis for the movie "{selected_movie}" based on:
    - Title
    - Genres
    - Release year
    - Average rating
    - Number of ratings
    Generate:
    • A short summary  
    • What type of audience might like it  
    • Whether it is likely high-rated or not based on metadata  
    """

    if st.button("Generate LLM Summary"):
        if not openai.api_key:
            st.error("API key required")
        else:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(str(e))
elif page == "Model Performance":
    st.header("Model Evaluation: F1 Score, Accuracy & Diagnostics")

    st.write("This evaluates the logistic regression model that predicts whether a movie achieves a high rating (≥ 4.0).")
    df_perf = df.copy()

    genre_dummies_perf = df_perf["genres"].str.get_dummies(sep="|")
    X_perf = pd.concat([df_perf[["year", "num_ratings"]], genre_dummies_perf], axis=1)
    y_perf = df_perf["high_rating"].astype(int).values

    train_mask = df_perf["year"] < 2015
    test_mask = df_perf["year"] >= 2015

    X_train_perf = X_perf[train_mask]
    X_test_perf = X_perf[test_mask]
    y_train_perf = y_perf[train_mask]
    y_test_perf = y_perf[test_mask]

    y_pred_perf = model.predict(X_test_perf)


    acc = accuracy_score(y_test_perf, y_pred_perf)
    f1 = f1_score(y_test_perf, y_pred_perf)

    st.subheader("Performance Metrics:")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test_perf, y_pred_perf, output_dict=True)

    st.subheader("Classification Report:")
    st.json(report)

    cm = confusion_matrix(y_test_perf, y_pred_perf)

    st.subheader("Confusion Matrix:")
    st.write(cm)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix Heatmap")

    st.pyplot(fig)
