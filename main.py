#Import libraries
import nltk
nltk.download('punkt')
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import requests
import telebot


stemmer = LancasterStemmer()

with open("model/intents.json") as file:
    data = json.load(file)
with open("model/data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)

# Converting message into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#Loading existing model from disk
model = tflearn.DNN(net)
model.load("model/model.tflearn")


searchurl = "https://api.themoviedb.org/3/search/movie?api_key=d2ad7dae00af2ece0a59f4dd88bc4fcb&language=en-US&query="

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string 
    for ele in s:
        str1 += ele 
    # return string 
    return str1

#api telegram
api = '5382004886:AAGqHM1Wqbg1NVtDPf61TqicHRV39Th5vuk'
bot = telebot.TeleBot(api)

#first message
@bot.message_handler(commands=['start'])
def action_start(message):
    nama = message.from_user.first_name
    bot.reply_to(message,'hallo {} selamat datang di chatbot sistem pemberi rekomendasi'.format(nama))

@bot.message_handler(content_types=['text'])
def get_chatbot_response(message):
    chat = message.text
    #message = requests.json.get('msg')
    results = model.predict([bag_of_words(chat,words)])[0]
    result_index = np.argmax(results)
    tag = labels[result_index]
    #resp = {}
    #resp["type"] = "bot"
    if results[result_index] > 0.4:
        if tag == "recommendations":
            print(chat)
            if 'kaya' in chat:
                movie_name = chat.split('kaya')[1].strip()
                
            elif 'mirip' in chat:
                movie_name = chat.split('mirip')[1].strip()
                
            elif 'seperti' in chat:
                movie_name = chat.split('seperti')[1].strip()
                
            elif 'nonton' in chat:
                movie_name = chat.split('nonton')[1].strip()
                
            else:
                print(movie_name)
                print("[INFO] Error movie / Input frow user")
                return "Judul tidak ditemukan"

            r = requests.get(url = searchurl+movie_name)
            movieData = r.json()
            if(len(movieData['results']))==0:
                print("[INFO] Error movie not found")
                
                return bot.reply_to(message,"masukan judul film yang benar kak :)")
            id = movieData['results'][0]['id']
            r = requests.get(url = f'https://api.themoviedb.org/3/movie/{id}/recommendations?api_key=d2ad7dae00af2ece0a59f4dd88bc4fcb&language=en-US&page=1')
            recommendData = r.json()
            s = "ini kak rekomendasi filmnya :-\n"
            n = len(recommendData['results'])
            if n> 5:
                n = 5
            for i in range(n):
                s+= f"{i+1}. {recommendData['results'][i]['original_title']}\n"
            bot.reply_to(message,{s})
        else:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            ss = random.choice(responses)
            bot.reply_to(message,{ss})
    else:
        bot.reply_to(message, "aku tidak mengerti kak, silakan coba lagi.")


print("[INFO] Bot Sedang Berjalan ...")
bot.infinity_polling(timeout=10, long_polling_timeout = 5)
# if __name__ == "__main__":
#         app.run(host='0.0.0.0',debug=False,port=8080)