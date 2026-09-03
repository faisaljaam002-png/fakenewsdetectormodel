# streamlit_app.py
# simple frontend that sends the article to the api and shows the result
# run the api first with: uvicorn api.main:app --reload

import streamlit as st
import requests

st.title("📰 Fake News Detector")
st.write("Paste in a news headline and article text to check if it's fake or real.")

title = st.text_input("Headline")
text = st.text_area("Article text", height=200)

if st.button("Check"):
    if text.strip() == "":
        st.write("Please paste in some article text first.")
    else:
        # send the text to our api and get back the prediction
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"title": title, "text": text}
        )

        if response.status_code == 200:
            result = response.json()
            label = result["label"]

            if label == "REAL":
                st.success("This looks REAL ✅")
            else:
                st.error("This looks FAKE 🚩")
        else:
            st.write("Something went wrong, make sure the api is running.")

st.caption("Note: this is a school/personal project, not a real fact checker. It just looks at word patterns.")
