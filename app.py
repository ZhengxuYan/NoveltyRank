
import streamlit as st
import pandas as pd
from datasets import load_dataset
from datetime import datetime, timedelta, timezone

# Page config
st.set_page_config(
    page_title="Novelty Rank: AI Paper Discovery",
    page_icon="🚀",
    layout="wide"
)

# Title and Description
st.title("🚀 Novelty Rank: Discover the Most Novel AI Papers")
st.markdown("""
This app ranks recent arXiv preprints based on their **Novelty Score**, predicted by a Siamese SciBERT model trained to compare papers within their field.
""")

# Sidebar Filters
st.sidebar.header("Filters")

# Load Data
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data():
    try:
        dataset = load_dataset("JasonYan777/novelty-ranked-preprints", split="train")
        df = dataset.to_pandas()
        # Convert published to datetime
        df["published"] = pd.to_datetime(df["published"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data found. Please run the ranking script first.")
else:
    # 1. Time Frame Filter
    days_options = [3, 7, 14, 30]
    selected_days = st.sidebar.select_slider("Time Frame (Last X Days)", options=days_options, value=7)
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=selected_days)
    # Ensure timezone awareness compatibility
    if df["published"].dt.tz is None:
        df["published"] = df["published"].dt.tz_localize(timezone.utc)
    
    filtered_df = df[df["published"] >= cutoff_date]
    
    # 2. Category Filter
    # Extract all unique categories
    all_cats = set()
    for cats in df["categories"]:
        if cats:
            for c in cats.split(", "):
                all_cats.add(c)
    
    # Main categories of interest
    main_cats = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.RO", "cs.CR"]
    other_cats = sorted(list(all_cats - set(main_cats)))
    filter_cats = ["All"] + main_cats + other_cats
    
    selected_cat = st.sidebar.selectbox("Category", filter_cats)
    
    if selected_cat != "All":
        # Filter if the category is present in the list
        filtered_df = filtered_df[filtered_df["categories"].str.contains(selected_cat, regex=False)]
        
    # Sort by Novelty Score
    filtered_df = filtered_df.sort_values("novelty_score", ascending=False)
    
    # Display Stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Papers Found", len(filtered_df))
    
    # Display Papers
    st.subheader(f"Top Papers (Last {selected_days} Days)")
    
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        with st.expander(f"#{i+1} {row['title']} (Score: {row['novelty_score']:.2f})", expanded=(i < 3)):
            st.markdown(f"**Categories:** {row['categories']}")
            st.markdown(f"**Published:** {row['published'].strftime('%Y-%m-%d')}")
            st.markdown(f"**Abstract:** {row['abstract']}")
            st.markdown(f"[Read on arXiv]({row['url']})")
            
            # Optional: Show similarity stats if available
            if "max_similarity" in row and pd.notna(row["max_similarity"]):
                 st.caption(f"Max Similarity to Training Set: {row['max_similarity']:.4f} | Avg Similarity: {row['avg_similarity']:.4f}")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Hugging Face")
