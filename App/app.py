import streamlit as st
import altair as alt

import  pandas as pd
import numpy as np
import time


import joblib

pipe_lr=joblib.load(open("models/emotion_classifier_pipe_lr_03_march_2024.pkl","rb"))

def predict_emotions(docx):
    results=pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("NLP Project")
    menu=["Emotion Classifier","About"]

    choice=st.sidebar.selectbox("Menu",menu)

    if choice=="Emotion Classifier":
        st.subheader("Emotion In Text")

        with st.form(key='emotion-clf-form'):
            raw_text=st.text_area("Type Here")
            submit_text=st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2=st.columns(2)

            prediction=predict_emotions(raw_text)
            probability=get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon=emotions_emoji_dict[prediction]
                st.write("{}: {}".format(prediction,emoji_icon))
                st.write("Confidence: {}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean=proba_df.T.reset_index()
                proba_df_clean.columns=["emotions","probability"]

                fig=alt.Chart(proba_df_clean).mark_bar().encode(x="emotions",y="probability", color="emotions")
                st.altair_chart(fig,use_container_width=True)

    else:
        st.subheader("About")
        st.write("This is an NLP powered WebApp that can predict emotions from text recognition with 70 percent accuracy. Many python libraries like Numpy, Pandas, Seaborn, Scikit-learn, Scipy, Joblib, eli5, lime, neattext, altair, streamlit were used. Streamlit was mainly used for the front-end development. Linear regression model from the scikit-learn library was used to train a dataset containing speeches and their respective emotions. Joblib was used for storing and using the trained model in the website.")
        st.caption('Created by: Hafsa Noorain')



if __name__=="__main__":
    main()