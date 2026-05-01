import numpy as o
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from google import genai
from google.genai import types
import time, os
from google.api_core import exceptions

base_path = os.path.dirname(__file__)
ratings_path = os.path.join(base_path, 'ml-latest-mid', 'ratings.csv')
movies_path = os.path.join(base_path, 'ml-latest-mid', 'movies.csv')

# App Config
st.set_page_config(page_title="MovieMind: AI Recommendation Engine", layout="wide")

@st.cache_resource
def load_and_train():
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', encoding='latin-1', 
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(movies_path, sep='::', engine='python', encoding='latin-1',
                         names=['item_id', 'title', 'genres'])

    data = ratings.merge(movies, on='item_id')
    data_exploded = data.assign(genre=data['genres'].str.split('|')).explode('genre')
    user_genre_profiles = data_exploded.pivot_table(index='user_id', columns='genre', values='rating', aggfunc='mean', fill_value=0)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    user_genre_profiles['cluster'] = kmeans.fit_predict(user_genre_profiles)

    # --- EVAL: Silhouette Score for k = 2 to 7 ---
    from sklearn.metrics import silhouette_score
    features_only = user_genre_profiles.drop('cluster', axis=1)
    print("\n--- Silhouette Scores ---")
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(features_only)
        score = silhouette_score(features_only, labels)
        print(f" k={k}: {score:.4f}")
    print(f" [Using k=5] Score = {silhouette_score(features_only, user_genre_profiles['cluster']):.4f}")

    ratings['liked'] = (ratings['rating'] >= 4).astype(int)
    clf_data = ratings.merge(movies, on='item_id').merge(user_genre_profiles[['cluster']], on='user_id')
    genre_dummies = clf_data['genres'].str.get_dummies(sep='|')
    X = pd.concat([genre_dummies, clf_data[['cluster']].reset_index(drop=True)], axis=1)
    y = clf_data['liked']
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # --- EVAL: Train/test split ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    # --- EVAL: Classification report ---
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

    y_pred_rf = clf.predict(X_test)
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred_rf))

    baseline = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    baseline.fit(X_train, y_train)
    y_pred_lr = baseline.predict(X_test)
    print("--- Baseline Logistic Regression Report ---")
    print(classification_report(y_test, y_pred_lr))
    
    return ratings, movies, user_genre_profiles, clf, kmeans
with st.spinner("Training the MovieMind Engine... This takes about 15 seconds."):
    ratings, movies, user_genre_profiles, clf, kmeans = load_and_train()
itemid_to_title = pd.Series(movies.title.values, index=movies.item_id.values).to_dict()

global_mean = ratings['rating'].mean()
m_threshold = 10

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def build_system_prompt():
    return (
        "You are MovieMind, a sharp and enthusiastic AI film curator. "
        "Your job is to explain in exactly 2 punchy sentences why a specific movie is a great pick for a viewer, "
        "based on their taste profile and the genre that drove the recommendation. "
        "Be specific, warm, and avoid generic filler phrases like 'this film is a must-watch'. "
        "Always reference the genre driver and connect it to the user's viewing identity."
    )

def build_user_prompt(user_id, persona, movie_title, weighted_score, top_driver):
    return (
        f"User '{user_id}' is a passionate '{persona}' film enthusiast. "
        f"Our ML pipeline recommended '{movie_title}' with a cluster-weighted score of {weighted_score:.1f}/5.0. "
        f"The key genre feature that flagged this movie is '{top_driver}'. "
        f"Write exactly 2 sentences explaining why this movie is a great match, "
        f"explicitly tying the '{top_driver}' elements to their taste profile."
    )

def stream_ai_explanation(user_id, persona, movie_title, weighted_score, top_driver):
    print(f"--- PULSE CHECK: Requesting AI for {movie_title} ---")
    try: 
        sys_instruction = build_system_prompt()
        user_content = build_user_prompt(user_id, persona, movie_title, weighted_score, top_driver)
    
        responses = client.models.generate_content_stream(
            model="gemini-2.5-flash-lite", 
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                max_output_tokens=200,
                temperature=0.7
            ),
            contents=user_content
        )

        for chunk in responses:
            if chunk.text:
                yield chunk.text
    except exceptions.ServiceUnavailable:
        yield "Google's servers are a bit crowded! Refreshing in a few seconds might help."
    except exceptions.ResourceExhausted:
        yield "We've hit the speed limit for now. Take a short intermission and try again shortly."
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        yield "MovieMind is currently experiencing some technical difficulties. Please check your terminal."

# --- UI Layout ---
st.title("MovieMind: AI Recommendation Engine")
st.markdown("---")

st.sidebar.title("MovieMind Navigation")
app_mode = st.sidebar.radio("Go To:", ["Home", "Existing User", "Create New Profile"])

# --- PERSISTENT STATE RESET ---
if "last_mode" not in st.session_state:
    st.session_state.last_mode = app_mode

if st.session_state.last_mode != app_mode:
    # Clear session values to prevent mode-leaking
    st.session_state.target_user_id = None
    st.session_state.current_persona = None
    st.session_state.comparison_list = []
    st.session_state.last_mode = app_mode
    st.rerun()

# --- INITIALIZATION LOGIC ---

# --- 1. HOME PAGE ---
if app_mode == "Home":
    st.header("Welcome to MovieMind AI")
    st.markdown("""
    ### Your Personal Cinematic Curator
    MovieMind combines **Machine Learning** with **Generative AI** to find your next favorite film.
    
    **How it works:**
    1. **The Engine:** We use K-Means clustering to find your 'film tribe' and Random Forest to predict what you'll love.
    2. **The Voice:** Google Gemini 2.5 Flash-Lite explains *exactly* why each pick fits your unique persona.
    
    **Get Started:**
    * Select **Existing User** in the sidebar to see how we've analyzed historical data.
    * Select **Create New Profile** to build your own taste profile from scratch!
    """)

# --- 2. EXISTING USER MODE --- 
elif app_mode == "Existing User":
    all_users = user_genre_profiles.index.tolist()
    selected_user = st.sidebar.selectbox("Select User ID", all_users)
    
    st.session_state.target_user_id = str(selected_user)
    st.session_state.target_cluster = user_genre_profiles.loc[selected_user, 'cluster']
    
    top_genres = user_genre_profiles.groupby('cluster').mean().iloc[st.session_state.target_cluster].nlargest(3).index.tolist()
    st.session_state.current_persona = ' & '.join(top_genres)
    st.session_state.comparison_list = top_genres
    pass

# --- 3. CREATE NEW PROFILE MODE ---
elif app_mode == "Create New Profile":
    st.subheader("Customize Your Cinematic Identity")
    user_name = st.text_input("Enter your name:", value="")
    available_genres = user_genre_profiles.columns.drop('cluster').tolist()
    fav_genres = st.multiselect("Select favorite genres:", options=available_genres)
    
    if st.button("Generate My Profile") and fav_genres:
        new_profile = pd.DataFrame(0, index=[0], columns=user_genre_profiles.columns.drop('cluster'))
        for g in fav_genres:
            new_profile[g] = 1
        
        st.session_state.target_cluster = kmeans.predict(new_profile)[0]
        st.session_state.target_user_id = user_name if user_name else "New Explorer"
        st.session_state.current_persona = ' & '.join(fav_genres)
        st.session_state.comparison_list = fav_genres
        st.success(f"Profile Generated! You are in Cluster {st.session_state.target_cluster}")
    elif not st.session_state.get('target_user_id'):
        st.info("Pick your genres and click 'Generate' to see your AI-curated list.")
        st.stop()
    pass

# --- MAIN ENGINE ---
if st.session_state.get('target_user_id'):
    target_user_id = st.session_state.target_user_id
    target_cluster = st.session_state.target_cluster
    current_persona = st.session_state.current_persona
    comparison_list = st.session_state.comparison_list

    cluster_users = user_genre_profiles[user_genre_profiles['cluster'] == target_cluster].index
    cluster_r = ratings[ratings['user_id'].isin(cluster_users)]
    movie_stats = cluster_r.groupby('item_id')['rating'].agg(['mean', 'count'])

    movie_stats['weighted_score'] = (
        (movie_stats['count'] * movie_stats['mean']) + (m_threshold * global_mean)
    ) / (movie_stats['count'] + m_threshold)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Profile Summary")
        st.info(f"**Persona:** {current_persona} Enthusiast")
        if app_mode == "Existing User":
            st.write("**Top History Likes:**")
            u_likes = ratings[(ratings['user_id'] == int(target_user_id)) & (ratings['rating'] >= 4.5)].head(3)
            for t in u_likes['item_id']:
                st.write(f"- {itemid_to_title.get(t)}")

    with col2:
        st.subheader("Top Machine Learning Recommendations")
        final_recs = []
        trusted_movies = movie_stats.sort_values(by='weighted_score', ascending=False).head(100)
        user_selected_clean = [g.strip().lower() for g in comparison_list]

        for m_id, row in trusted_movies.iterrows():
            m_title = itemid_to_title.get(m_id)
            if not m_title: continue
            
            m_genres_list = movies[movies['item_id'] == m_id]['genres'].iloc[0].split('|')
            movie_genre_clean = [g.strip().lower() for g in m_genres_list]

            if not any(genre in movie_genre_clean for genre in user_selected_clean):
                continue

            input_features = pd.DataFrame(0, index=[0], columns=clf.feature_names_in_)
            for col in m_genres_list:
                if col in input_features.columns:
                    input_features[col] = 1
            input_features['cluster'] = target_cluster

            if clf.predict(input_features)[0] == 1:
                matching_genres = [g for g in m_genres_list if g.lower() in user_selected_clean]
                top_driver = matching_genres[0] if matching_genres else "Cinematic Quality"

                final_recs.append({
                    "Movie Title": m_title,
                    "Predicted Rating": round(float(row['weighted_score']), 2),
                    "top_driver": top_driver,
                })
            
            if len(final_recs) >= 5:
                break

        if final_recs:
            st.table(pd.DataFrame(final_recs)[['Movie Title', 'Predicted Rating']])
            st.subheader("AI Explanations (Powered by Gemini)")
            
            for i, rec in enumerate(final_recs):
                if i > 0:
                    time.sleep(2)

                with st.expander(f"#{i+1} — {rec['Movie Title']} ({rec['Predicted Rating']}★)", expanded=(i == 0)):
                    st.write_stream(
                        stream_ai_explanation(
                            user_id=target_user_id,
                            persona=current_persona,
                            movie_title=rec["Movie Title"],
                            weighted_score=rec["Predicted Rating"],
                            top_driver=str(rec["top_driver"]),
                        )
                    )
        else:
            st.warning("No matches found for your selected genres in this cluster. Try adding more variety!")