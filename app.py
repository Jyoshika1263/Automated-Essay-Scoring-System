import streamlit as st
from keras.models import load_model
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

ALLOWED_SPECIAL_CHARACTERS = set(string.punctuation)

def has_special_characters(text):
    return any(char in ALLOWED_SPECIAL_CHARACTERS for char in text)

def has_invalid_characters(text):
    invalid_chars = set("~@#$%^&*_=+")
    return any(char in invalid_chars or char.isnumeric() for char in text)

def matching_criteria(essay_subject, essay_input):
    if not essay_subject or not essay_input:
        return False

    if has_invalid_characters(essay_input):
        return False

    # Check if any words from the essay match with the question or essay topic
    essay_words = set(word.lower() for sentence in essay_input for word in sentence if word.isalpha())
    subject_words = set(word.lower() for sentence in essay_subject for word in sentence if word.isalpha())

    # Use cosine similarity to check for word similarity
    cosine_sim = cosine_similarity([essay_words], [subject_words])
    return cosine_sim[0][0] > 0.8  # Adjust the threshold as needed

def sent2word(x):
    stop_words = set(stopwords.words('english'))
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = []
    words = x.split()
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if len(i) > 0:
            final_words.append(sent2word(i))
    return final_words

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])
    vec = np.divide(vec, noOfWords) if noOfWords != 0 else vec
    return vec

def getVecs(essays, model, num_features):
    c = 0
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c += 1
    return essay_vecs

def convertToVec(text):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        predicted_score = round(preds[0][0])
        return predicted_score

def get_qualitative_description(score):
    if 5 <= score <= 7:
        return "Average"
    elif 8 <= score <= 9:
        return "Good"
    elif score == 10:
        return "Excellent"
    else:
        return "Poor"

def get_feedback(score):
    if score <= 5:
        return "Your essay seems to need improvement. Consider adding more details and organizing your thoughts."
    elif 6 <= score <= 7:
        return "Your essay is okay, but there's room for improvement. Focus on providing more context and clarity."
    elif 8 <= score <= 9:
        return "Well done! Your essay is good, but you can still enhance it by refining your language and structure."
    elif score == 10:
        return "Excellent job! Your essay is well-written and effectively communicates your ideas."

def matching_criteria(essay_subject, essay_input):
    if not essay_subject or not essay_input:
        return False

    if has_invalid_characters(essay_input):
        return False

    # Check if any words from the essay match with the question or essay topic
    essay_words = set(word.lower() for sentence in essay_input for word in sentence if word.isalpha())
    subject_words = set(word.lower() for sentence in essay_subject for word in sentence if word.isalpha())

    return bool(essay_words.intersection(subject_words))

def main():
    st.title("Automated Essay Scoring System")
    st.sidebar.text('Happy learning')
    st.sidebar.image('teacher-1916.gif', use_column_width=True)

    # Dialog box for entering a question
    essay_subject = st.text_input("Enter your question or essay topic:")

    # Dialog box for entering an essay
    essay_input = st.text_area("Enter your essay:")

    # If the entered subject and essay are not empty
    if essay_subject and essay_input:
        # Check if the essay words match with the question or essay topic
        if matching_criteria(essay_subject, essay_input):
            st.write("Subject of the essay matches the expected format. Proceed to enter your essay.")

            # Buttons for scoring and feedback
            if st.button("Get Score"):
                with st.spinner("Scoring..."):
                    score = convertToVec(essay_input)
                    description = get_qualitative_description(score)
                st.success(f"The predicted score is: {score} - *{description}*")

            if st.button("Get Feedback"):
                with st.spinner("Generating Feedback..."):
                    score = convertToVec(essay_input)
                    feedback = get_feedback(score)
                st.info("**Feedback:**")
                st.write(feedback)

        else:
            st.warning("Subject of the essay does not match the expected format or contains invalid characters.")
    else:
        st.warning("Please enter a question or essay topic and an essay before proceeding.")


if __name__ == '__main__':
    main()
