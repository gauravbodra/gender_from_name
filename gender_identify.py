import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
import nltk
import csv
nltk.download('names')

#I have used default database
'''
#if you have names list in txt files respectively use these lines to load your own database
with open('female.csv', 'r') as female:
    reader_fe = csv.reader(female)
    female_list = list(reader_fe)

with open('male.csv', 'r') as male:
    reader_fe = csv.reader(male)
    male_list = list(reader)
'''
# Extract features from the input word
def gender_features(word, num_letters=2):
    
    return {'feature': str(word[-num_letters:]).lower()}

if __name__=='__main__':
    #in case of You have names in file remove from comment -These lines 
    '''
    labeled_names = ([(name, 'male') for name in male_list] +
            [(name, 'female') for name in female_list])
    '''
    
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
            [(name, 'female') for name in names.words('female.txt')])

    random.seed(7)
    random.shuffle(labeled_names)
    #Enter Your Name here to check
    input_names = ["Hardik"]

    
    for i in range(1, 6):
        print("\n")
        print ("Number of letters:", i)
        featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names]
        train_set, test_set = featuresets[500:], featuresets[:500]
        classifier = NaiveBayesClassifier.train(train_set)

        # Print accuracy
        print ("Accuracy -->", str(100 * nltk_accuracy(classifier, test_set)) + str('%'))

        # Predict outputs for new inputs
        for name in input_names:
            print (name, "-->", classifier.classify(gender_features(name, i)))

