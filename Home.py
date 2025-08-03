import streamlit as st

# Logo
st.image("logo.png", width=150)

# Title and description
st.title("Welcome to SkinDX")
st.markdown("""
**AI-Powered Skin Lesion Analysis**

This web app lets you upload a skin lesion photo, processes it with a deep learning model, 
and displays a diagnosis with highlighted analysis areas. The interface is clean, simple, 
and designed for easy use.
""")

# Navigation to Classify Image
if st.button("🔍 Classify Image"):
    st.switch_page("pages/Classify_Image.py")
