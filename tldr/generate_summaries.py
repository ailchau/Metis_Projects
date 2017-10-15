import os
import re
import numpy as np
import pandas as pd
import random
import pickle
import tensorflow as tf # tensorflow version 1.1

# ---------------------------

# load necessary files
int_to_vocab = pickle.load(open("./model_files/int_to_vocab.pkl","rb"))
vocab_to_int = pickle.load(open("./model_files/vocab_to_int.pkl","rb"))
test_articles_dict = pickle.load(open("other_topic_test_dict.pkl","rb"))

# easier to be fed in
test_articles = [article["article"] for article in test_articles_dict]
test_summaries_length = [len(article["highlights"].split()) for article in test_articles_dict]

# ---------------------------

# Generate summaries

def text_to_seq(text):
    '''Prepare the text for the model'''
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


batch_size = 64
generate_words = []

checkpoint = "./best_model.ckpt"


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    
    for index,article in enumerate(test_articles):
        text = text_to_seq(article)
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                          summary_length: [test_summaries_length[index]], 
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0] 

        # Remove the padding from the tweet
        pad = vocab_to_int["<PAD>"]
        generate_words.append(" ".join([int_to_vocab[i] for i in answer_logits if i != pad]))
        
        # quick update
        if index%50==0:
            print("Generating. Currently at {}".format(index))
        
    print("Number of generated summaries: {}".format(len(generate_words)))

# add generated text to dict
for index,article_dict in enumerate(test_articles_dict):
    article_dict["generated"] = generate_words[index]


# ---------------------------

# Write output to file

with open("generated_crime_other_summaries.txt", "w") as output:
    for item in test_articles_dict:
        output.write("{}\n\n".format(item))

print("Output file written.")
