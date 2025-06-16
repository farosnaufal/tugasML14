
import streamlit as st
import pickle

# Load model dan vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Spam Message Detection")
st.write("Masukkan pesan SMS/email untuk mendeteksi apakah itu spam atau bukan.")

message = st.text_area("Masukkan Pesan")

if st.button("Deteksi"):
    if message:
        vector = vectorizer.transform([message])
        prediction = model.predict(vector)
        st.success("Hasil: SPAM" if prediction[0] == 1 else "Hasil: BUKAN SPAM")
    else:
        st.warning("Harap masukkan pesan terlebih dahulu.")
