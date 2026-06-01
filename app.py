import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('shekspeare2.h5')


with open('tokenzer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)

input_seq_len = 30

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



def prediction2(
    text: str,n_word: int,k: int = 7,temperature: float = 0.9
) -> str:
    """
    Generate n_word next words using temperature-scaled top-k sampling.

    Args:
        text: text.
        n_word: number of words to generate.
        k: Sample only from top-k predictions.
        temperature:
            <1.0 = more conservative
            1.0 = unchanged
            >1.0 = more random
    """

    for _ in range(n_word):

        # Convert text to token ids
        token_text = tokenizer.texts_to_sequences([text])[0]

        # Pad to model input length
        padded_token_input = pad_sequences(
            [token_text],
            maxlen=input_seq_len,
            padding="pre",
            truncating="pre"
        )

        # Predict probabilities
        probs = model.predict(
            padded_token_input,
            verbose=0
        )[0]

        # Temperature scaling
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / probs.sum()

        # Top-k selection
        top_k_indices = np.argsort(probs)[-k:]

        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / top_k_probs.sum()

        # Sample
        predicted_index = np.random.choice(
            top_k_indices,
            p=top_k_probs
        )

        # Index -> word
        predicted_word = tokenizer.index_word.get(
            predicted_index,
            "<UNK>"
        )

        # Append word
        text += " " + predicted_word

    return text
    

st.title("Next word prediction with LSTM")
input_text=st.text_input("Enter the sequence of words",'to be or not to be')
input_no=int(st.text_input("Enter no of words to generate",'2'))
option=st.selectbox("how to generate text",
    ("top k sampled", "argmax"),)
print(option)

if st.button("predict next word"):
    max_sequence_len=model.input_shape[1]+1
    if option=="top k sampled":

        next_word=prediction2(input_text,input_no)          

    else:    
        next_word=prediction(input_text,input_no)
    st.write(f'{next_word}')