import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Sentiment Analyzer", layout="wide", page_icon="💬")

# Load Hugging Face BERT model using PyTorch
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# Inject Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f8;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.title("📘 About")
    st.markdown("""
        A modern sentiment analysis app using Hugging Face's BERT model.

        ➤ Analyze single or batch text  
        ➤ Get emoji-based feedback and visuals  
        ➤ Download sentiment reports  

        Built with ❤️ using Python + Streamlit.
    """)
    st.caption("💻 Created by: Ishan Gupta")

# App Title
st.title("💬 Sentiment Analysis Tool")

# Input method
input_option = st.radio("Choose Input Method", ("Text Input", "Upload File"))

# Text Input
if input_option == "Text Input":
    user_input = st.text_area("📝 Enter your text here:", height=150)
    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip():
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']

            # Emoji map
            emoji_map = {
                "POSITIVE": "😊",
                "NEGATIVE": "😠",
                "NEUTRAL": "😐"
            }
            emoji = emoji_map.get(label.upper(), "😐")

            st.markdown("### 📊 Result")
            st.success(f"**Sentiment:** {label.capitalize()} {emoji}")
            st.write(f"**Confidence:** `{score*100:.2f}%`")
            st.progress(score)

            result_df = pd.DataFrame({
                "Input Text": [user_input],
                "Sentiment": [label.capitalize()],
                "Confidence": [f"{score*100:.2f}%"]
            })
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Result as CSV",
                data=csv,
                file_name="single_sentiment_result.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please enter some text.")

# File Upload
elif input_option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload a .txt or .csv file", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            raw_text = uploaded_file.read().decode("utf-8")
            st.text_area("📄 File Content", raw_text, height=250)
            if st.button("🔍 Analyze File"):
                result = classifier(raw_text)[0]
                label = result['label']
                score = result['score']
                emoji_map = {
                    "POSITIVE": "😊",
                    "NEGATIVE": "😠",
                    "NEUTRAL": "😐"
                }
                emoji = emoji_map.get(label.upper(), "😐")

                st.markdown("### 📊 Result")
                st.success(f"**Sentiment:** {label.capitalize()} {emoji}")
                st.write(f"**Confidence:** `{score*100:.2f}%`")
                st.progress(score)

                result_df = pd.DataFrame({
                    "Input Text": [raw_text],
                    "Sentiment": [label.capitalize()],
                    "Confidence": [f"{score*100:.2f}%"]
                })
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Result as CSV",
                    data=csv,
                    file_name="textfile_sentiment_result.csv",
                    mime="text/csv"
                )

        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                st.write("### 📄 Data Preview")
                st.dataframe(df.head())

                if st.button("🔍 Analyze CSV"):
                    st.info("⏳ Analyzing text with BERT model...")

                    sentiments = []
                    for text in df['text']:
                        res = classifier(str(text))[0]
                        sentiments.append(res['label'].capitalize() + " " + (
                            "😊" if res['label'] == "POSITIVE" else "😠" if res['label'] == "NEGATIVE" else "😐"))

                    df['Sentiment'] = sentiments
                    st.success("✅ Analysis Complete")
                    st.dataframe(df[['text', 'Sentiment']])

                    # 📊 Sentiment Distribution Charts
                    st.write("### 📊 Sentiment Distribution")
                    sentiment_counts = df['Sentiment'].value_counts()

                    # Bar Chart
                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel', ax=ax1)
                    ax1.set_ylabel("Number of Texts")
                    ax1.set_title("Sentiment Distribution (Bar Chart)")
                    st.pyplot(fig1)

                    # Pie Chart
                    fig2, ax2 = plt.subplots()
                    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                            colors=sns.color_palette("pastel"), startangle=140)
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")
            else:
                st.error("❗ CSV must have a column named `text`.")
