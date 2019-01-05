import csv
import random as rd

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
    pos_file = list(open(posfile, "r").readlines())             #opening the file from posfile and removing all the unwanted spaces and storing in pos_file
    pos_file = [s.strip() for s in pos_file]

    neg_file = list(open(negfile, "r").readlines())             #opening the file from negfile and removing all the unwanted spaces and storing in neg file
    neg_file = [s.strip() for s in neg_file]

    if (randomize):
        rd.shuffle(pos_file)
        rd.shuffle(neg_file)

    pos_file = pos_file[:max_limit]
    neg_file = neg_file[:max_limit]

    x_text = pos_file + neg_file                                #splits data by words
    x_text = [clean_data(s) for s in x_text]


    pos_file =



