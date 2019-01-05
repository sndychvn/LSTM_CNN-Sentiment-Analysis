import csv


#seperates the mixed file into positive and negative
def split_data(filename):
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

            if (sentiment == "0"):
                negative_file.write(sentence+"\n")





