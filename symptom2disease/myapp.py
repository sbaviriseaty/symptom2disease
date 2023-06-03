import streamlit as st
import cohere
import numpy as np

co = cohere.Client('wxLxQpiNjXruopgx0sV6EvFGJIFxQAyyl3kyzM3m')

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = 'Output:'

diseases = [
    'Psoriasis',
    'Varicose Veins',
    'Typhoid',
    'Chicken Pox',
    'Impetigo',
    'Dengue',
    'Fungal Infection',
    'Common Cold',
    'Pneumonia',
    'Dimorphic Hemorrhoids',
    'Arthritis',
    'Acne',
    'Bronchial Asthma',
    'Hypertension',
    'Migraine',
    'Cervical Spondylosis',
    'Jaundice',
    'Malaria',
    'Urinary Tract Infection',
    'Allergy',
    'Gastroesophageal Reflux Disease',
    'Drug Reaction',
    'Peptic Ulcer Disease',
    'Diabetes'
]


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
st.write('''This will output confidence scores for the top 5 out of 24 diseases: ''', ', '.join(diseases))
st.divider()
input = st.text_area('Enter your symptoms here.', height=100)
st.button('Classify Disease', on_click=classify_disease(input))
# st.write(st.session_state['output'])
st.divider()
if len(input) != 0:
    # output = classify_disease(input)
    output = st.session_state['output']
    sorted_output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1].confidence, reverse=True)}
    # print(output)
    col1, col2 = st.columns(2)
    count = 0
    disease = ""
    for key in sorted_output:
        if count == 0:
            col1.write(f"**{key}**: {np.round(sorted_output[key].confidence * 100, 2)}%")
            col1.progress(sorted_output[key].confidence)
            disease = key
        elif count < 5:
            col1.write(f"{key}: {np.round(sorted_output[key].confidence * 100, 2)}%")
            col1.progress(sorted_output[key].confidence)
            # print(key, '->', output[key].confidence)
        count += 1
    response = co.generate(
        prompt='Can you provide a description of ' + disease + '?',
        max_tokens=250,
    )
    col2.write(response.generations[0].text)
