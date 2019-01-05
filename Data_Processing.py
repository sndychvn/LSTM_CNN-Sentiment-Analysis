import csv
import random as rd
import re
import numpy as np

def split_data(filename):                                  #seperates the mixed file into positive and negative
    positive_file = open("positive_"+filename,"w+")        #creates empty files to store the
    negative_file = open("negative_"+filename, "w+")       #positive and the negative sentences accordingly

    check = 1

    with open(filename, 'r') as f:
        reader = csv.reader(f)                             #returns a reader object which will iterate over lines in the given csvfile
        reader.next()

        for row in reader:                                 #iterating over each row in the reader object or file
            check +=1
            sentiment = row[1]
            sentence = row[3]

            if (sentiment=="0"):                           #seperates the sentences to files and writes them to file based on sentiment
                negative_file.write(sentence+"\n")
            else:
                positive_file.write(sentence+"\n")

            if(check%10000==0):
                print(check)

    positive_file.close()
    negative_file.close()



def load_data(posfile, negfile, max_limit, randomize=True):     #get the datset
    pos_file = list(open(posfile, "r").readlines())             #opening the file from posfile and removing all the leading spaces and storing in pos_file
    pos_file = [s.strip() for s in pos_file]

    neg_file = list(open(negfile, "r").readlines())             #opening the file from negfile and removing all the leading spaces and storing in neg file
    neg_file = [s.strip() for s in neg_file]

    if (randomize):                                             #shuffles the data positions randomly
        rd.shuffle(pos_file)
        rd.shuffle(neg_file)

    pos_file = pos_file[:max_limit]
    neg_file = neg_file[:max_limit]

    x_text = pos_file + neg_file                                #splits data by words
    x_text = [clean_data(s) for s in x_text]


    pos_labels = [[0,1] for _ in pos_file]                      #generates label if a list "pos_file" has 200 values, a pos_lables of 200 [0,1] is generated
    neg_labels = [[1,0] for _ in neg_file]
    y = np.concatenate([pos_labels, neg_labels],0)
    return [x_text,y]


def clean_data(string):

    string = re.sub(r":\)", "emojihappy1", string)              # emojis cleanup
    string = re.sub(r":P", "emojihappy2", string)
    string = re.sub(r":p", "emojihappy3", string)
    string = re.sub(r":>", "emojihappy4", string)
    string = re.sub(r":3", "emojihappy5", string)
    string = re.sub(r":D", "emojihappy6", string)
    string = re.sub(r" XD ", "emojihappy7", string)
    string = re.sub(r" <3 ", "emojihappy8", string)
    string = re.sub(r":\(", "emojisad9", string)
    string = re.sub(r":<", "emojisad10", string)
    string = re.sub(r":<", "emojisad11", string)
    string = re.sub(r">:\(", "emojisad12", string)
    string = re.sub(r"(@)\w+", "mentiontoken", string)          # hashtags, etc cleanup "(@)\w+"
    string = re.sub(r"http(s)*:(\S)*", "linktoken", string)     # websites cleanup
    string = re.sub(r"\\x(\S)*", "", string)                    # unwanted unicode cleanup \x...
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)      # General Cleanup and Symbols
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):                     #generates a batch iterator for a dataset
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):                                             #shuffles the batch at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))       #an index of the size of data would be created and would be shuffled
            shuffled_data = data[shuffle_indices]                               #this same shuffled index would be prefixed to the data, where shuffled_data would contain the shuffled index elements
        else:
            shuffled_data = data

        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    seperate_dataset("small.txt")



