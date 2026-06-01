import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('next_word_lstm.h5')


with open('tokenzer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)


def prediction(text:str, n_word:int)->str:
    """
    Predict the next 'n_word' words based on the input 'text'.
    
    Args:
    text (str): The input text to seed the prediction.
    n_word (int): The number of words to predict and append.
    
    Returns:
    str: The input text appended with the predicted words.
    """
    
    for _ in range(n_word):
        # Convert the input text to sequences (tokenized)
        token_text = tokenizer.texts_to_sequences([text])[0]
        
        # Pad the sequence to the max sequence length expected by the model
        padded_token_input = pad_sequences([token_text], maxlen=input_seq_len, padding="pre")
        
        # Predict the probabilities for the next word
        output_prob = model.predict(padded_token_input, verbose=0)
        
        # Find the word with the highest probability
        pos = np.argmax(output_prob)
        
        # Map the predicted index back to the corresponding word
        for word, index in tokenizer.word_index.items():
            if index == pos:
                # Append the predicted word to the input text
                text = text + " " + word
                break
    
    return text
    

st.title("Next word prediction with LSTM")
input_text=st.text_input("Enter the sequence of words",'to be or not to be')
input_no=st.text_input("Enter no of words to generate",'2')
if st.button("predict next word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=prediction(model,tokenizer,input_text,max_sequence_len)
    st.write(f'{next_word}')