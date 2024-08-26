import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# print("all loaded")
model=load_model('next_word_lstm.h5')
with open('tokenzer.pickle','rb') as handle:
  tokenizer=pickle.load(handle)
  
def predict_next_word(model,tokenize,text,max_seq_length):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if len(token_list)>=max_seq_length:
    token_list=token_list[-(max_seq_length-1):]
  token_list=pad_sequences([token_list],maxlen=max_seq_length-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predict_next_index=np.argmax(predicted,axis=1)
  for word, index in tokenizer.word_index.items():
    if index==predict_next_index:
      return word
  return None

# Streamlit app
st.title("Next Word Prediction")
input_text=st.text_input("Enter text")

max_seq_length=model.input_shape[1]+1
output=predict_next_word(model,tokenizer,input_text,max_seq_length)
st.write(f"output: {input_text} {output}")