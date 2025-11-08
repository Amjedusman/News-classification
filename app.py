import streamlit as st
import pickle
import requests
import env as en

# -------------------------
# Load Trained Model and Vectorizer
# -------------------------
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📰 Real-Time News Topic Classifier")
st.caption("Using AG News Trained Model")

# -------------------------
# Sample Headlines for Testing
# -------------------------
sample_headlines = [
    "Scientists discover new planet in habitable zone",
    "Stock market reaches all-time high amid economic recovery",
    "Championship game ends in dramatic overtime victory",
    "World leaders meet to discuss climate change policies",
    "Tech company announces breakthrough in quantum computing",
    "Olympic athlete breaks world record in 100m sprint",
    "Global trade agreement signed by major economies",
    "Space mission successfully lands on Mars",
    "Football team wins league championship",
    "New study reveals impact of artificial intelligence on jobs"
]

# -------------------------
# Headline Source Selection
# -------------------------
st.sidebar.header("⚙️ Headline Source")
source = st.sidebar.radio(
    "Choose headline source:",
    ["GNews API", "Manual Input", "Sample Headlines"],
    index=2  # Default to sample headlines
)

headlines = []

if source == "GNews API":
    # -------------------------
    # Fetch Real-Time Headlines using GNews API
    # -------------------------
    API_KEY = en.api_key
    url = en.url
    
    try:
        with st.spinner("Fetching headlines from GNews API..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            
            # Check if API returned an error
            if "errors" in data:
                st.error(f"API Error: {', '.join(data['errors'])}")
                st.info("Please check your GNews API account status. You may need to activate your account at https://gnews.io/dashboard")
                st.info("💡 Tip: Switch to 'Sample Headlines' or 'Manual Input' to test the model.")
            elif "articles" in data:
                headlines = [article["title"] for article in data.get("articles", [])]
                st.success(f"Successfully fetched {len(headlines)} headlines!")
            else:
                st.warning("Unexpected API response format.")
                
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch headlines: {str(e)}")
        st.info("Please check your internet connection and API configuration.")
        st.info("💡 Tip: Switch to 'Sample Headlines' or 'Manual Input' to test the model.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("💡 Tip: Switch to 'Sample Headlines' or 'Manual Input' to test the model.")

elif source == "Manual Input":
    st.subheader("✍️ Enter Your Own Headlines")
    st.write("Enter headlines (one per line) to classify:")
    
    manual_input = st.text_area(
        "Headlines:",
        height=200,
        placeholder="Enter headlines here, one per line...\n\nExample:\nScientists discover new planet\nStock market reaches all-time high\nFootball team wins championship"
    )
    
    if st.button("Classify Headlines"):
        if manual_input.strip():
            headlines = [line.strip() for line in manual_input.split("\n") if line.strip()]
            if headlines:
                st.success(f"Processing {len(headlines)} headline(s)...")
            else:
                st.warning("Please enter at least one headline.")
        else:
            st.warning("Please enter some headlines to classify.")

elif source == "Sample Headlines":
    st.subheader("📋 Sample Headlines")
    st.write("Using sample headlines for demonstration:")
    headlines = sample_headlines
    st.info(f"Loaded {len(headlines)} sample headlines.")

# -------------------------
# Classify and Display Results
# -------------------------
if headlines:
    st.divider()
    st.subheader("📊 Classification Results")
    
    for i, headline in enumerate(headlines, 1):
        X_tfidf = vectorizer.transform([headline])
        pred = model.predict(X_tfidf)[0]
        
        # Color coding for categories
        category_colors = {
            "World": "🌍",
            "Sports": "⚽",
            "Business": "💼",
            "Sci/Tech": "🔬"
        }
        emoji = category_colors.get(pred, "📰")
        
        st.markdown(f"### {emoji} {headline}")
        st.write(f"**Predicted Category:** `{pred}`")
        st.divider()
else:
    if source == "GNews API":
        st.warning("No headlines fetched from API. Please try another source or check your API configuration.")
