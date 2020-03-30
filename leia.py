import numpy as np
import re
import time
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from sklearn.metrics.pairwise import cosine_similarity



# Load the data
emotion_lines = open('dialogues_emotion.txt').read().split('\n')
conv_lines = open('dialogues_text.txt').read().split('\n')
act_lines = open('dialogues_act.txt').read().split('\n')

emotions =[]
for i in range(len(emotion_lines)):
    emotions.append(emotion_lines[i].split(" "))

act = []
for i in range(len(act_lines)):
    act.append(act_lines[i].split(" "))

corpus = []
for i in range(len(conv_lines)):
    corpus.append(conv_lines[i].split("__eou__"))

new_questions = []
new_answers = []
afinn = Afinn(language='en')

len(corpus) - 1
for j in range(len(corpus) - 1):
    for i in range(len(corpus[j]) - 2):
        if (emotions[j][i] == '4'):
            if afinn.score(corpus[j][i + 1]) >= 0:
                new_questions.append(corpus[j][i])
                new_answers.append(corpus[j][i + 1])

# Preprocessing
lemmer = nltk.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def generate_response(questions, answers, user_input):
    # print(len(questions))
    # print(len(answers))
    questions.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(questions)

    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    """
    question_response = [questions[idx], questions[vals.argsort()[0][-3]],questions[vals.argsort()[0][-4]],
                    questions[vals.argsort()[0][-5]],questions[vals.argsort()[0][-6]]]
    robo_response = [answers[idx], answers[vals.argsort()[0][-3]],answers[vals.argsort()[0][-4]],
                    answers[vals.argsort()[0][-5]],answers[vals.argsort()[0][-6]]]
    """
    if (req_tfidf == 0):
        robo_response = "I am sorry! I don't understand you"
    else:
        # print(idx, "idex")
        robo_response = answers[idx]

    answers.append(robo_response)
    # return question_response, robo_response
    return robo_response

quit = False

GREETING_INPUTS = ("hello", "hi", "Hi", "greetings", "sup", "what's up","hey")
GREETING_RESPONSES = "Hello Cutie"

while quit == False:
    print("User: ")
    user_input = raw_input()
    print("Leia: ")
    if(user_input in GREETING_INPUTS):
        print(GREETING_RESPONSES)
    elif(user_input!='bye'):
        print(generate_response(new_questions,new_answers,user_input))
    else:
        print("bye")
        quit = True
