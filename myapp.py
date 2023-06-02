import streamlit as st
import cohere
import numpy as np

co = cohere.Client('wxLxQpiNjXruopgx0sV6EvFGJIFxQAyyl3kyzM3m')

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = 'Output:'


def classify_disease(input):
    if len(input) == 0:
        return None
    response = co.classify(
        model='65ddf5bc-7962-4a20-a71b-9f33b5724110-ft',
        inputs=[input])
    # print('The confidence levels of the labels are: {}'.format(response.classifications[0]))

    # st.session_state['output'] = response.classifications[0].prediction
    # return response.classifications[0].labels
    # st.balloons()

    st.session_state['output'] = response.classifications[0].labels


st.title('Symptom 2 Disease')
st.subheader('Classify disease from symptoms')
st.write('''This will output one of 24 diseases from input symptoms.''')
input = st.text_area('Enter your symptoms here.', height=100)
st.button('Classify Disease', on_click=classify_disease(input))
# st.write(st.session_state['output'])
if len(input) != 0:
    # output = classify_disease(input)
    output = st.session_state['output']
    sorted_output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1].confidence, reverse=True)}
    # print(output)
    col1, col2 = st.columns(2)
    count = 0
    for key in sorted_output:
        if count == 0:
            col1.write(f"**{key}**: {np.round(sorted_output[key].confidence * 100, 2)}%")
            col1.progress(sorted_output[key].confidence)
            count += 1
        else:
            col1.write(f"{key}: {np.round(sorted_output[key].confidence * 100, 2)}%")
            col1.progress(sorted_output[key].confidence)
            # print(key, '->', output[key].confidence)