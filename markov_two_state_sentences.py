import numpy as np
import json
import re

class MrMarkov:

    '''
    Idea nabbed from markovify source
    '''    
    BEGIN = "__BEGIN__"
    END = "__END__"


    def __init__(self, filename=None, corpus=None):

        if corpus:
            self.corpus = corpus

        else:
            self.corpus = {}

            if not filename:
                raise Exception("filename must be provided to make corpus!")
            
            self.filename = filename

    def normalize(self):
        """
        a function that normalizes the counts of words 
        """
        for key in self.corpus.keys():
            sum_count = 0
            words = []
            counts = []
            for k, v in self.corpus[key].items():
                sum_count += v
                words.append(k)
                counts.append(v)
            prob = [float(count)/sum_count for count in counts]

            self.corpus[key] = [words, prob]

    def add_word(self, word_one, word_two):
        """
        a function to handle words
        """
        
        # if word_one is in the corpus
        if word_one in self.corpus.keys():
            # if the word_two is already in word_one's corpus
            if word_two in self.corpus[word_one].keys():
                # increment the count by 1
                self.corpus[word_one][word_two] += 1
            # if word_two is not in word_one's corpus
            else:
                # add word_two to the corpus with an initial value of 1
                self.corpus[word_one][word_two] = 1
        # if word_one is not already in the corpus
        else:
            # add it and initialize its dictionary with word_two
            self.corpus[word_one] = {word_two : 1}
        
    def markovifile(self):
        with open(self.filename, 'r', encoding='utf-8') as file:

            for line in file:
                line = line.encode('ascii', errors='ignore').strip().decode('ascii')
                line_array = line.split()

                if len(line_array) == 0:
                    continue

                new_line_array = []

                new_line_array.append(self.BEGIN)

                for i, word in enumerate(line_array):
                    new_line_array.append(word)
                    if ("." in word or "?" in word or "!" in word \
                        or ".\"" in word or "?\"" in word or "!\"" in word) \
                        and i != len(line_array)-1 \
                        and "..." not in word:
                        new_line_array.append(self.END)
                        new_line_array.append(self.BEGIN)

                new_line_array.append(self.END)

                for i in range(len(new_line_array)-2):

                    first_word = new_line_array[i]
                    second_word = new_line_array[i+1]
                    third_word = new_line_array[i+2]

                    key_word = "{} {}".format(first_word, second_word)                    

                    self.add_word(key_word, third_word)

            file.close()

            self.normalize()
    
    def generate_text(self, num_sentences, starting_word):
        """
        docstring
        """
        text = []

        starting_word = "{} {}".format(self.BEGIN, starting_word)

        if starting_word not in self.corpus.keys():
            starting_word = np.random.choice(list(self.corpus.keys()))
            print("Word not found in corpus, using random start: {}".format(starting_word))

        starting_words = starting_word.split()
        text.append(starting_words[0])
        text.append(starting_words[1])

        counter = 0
        index = 1

        while counter < num_sentences:

            key = " ".join([text[index-1], text[index]])

            if key not in self.corpus.keys():
                matching_keys = list(filter(lambda c_key: text[index] in c_key, self.corpus.keys()))
                key = np.random.choice(matching_keys, size=1)[0]

            words, prob = self.corpus[key]

            new_word = np.random.choice(words, p=prob, size=1)[0]
            if new_word == self.END:
                counter += 1
            text.append(new_word)
            index += 1
        
        text = list(filter(lambda word: word not in [self.BEGIN, self.END], text))

        return " ".join(text)

def main():
    markov = MrMarkov(filename="stories.txt")

    markov.markovifile()

    with open('test.json', 'w') as fp:
        json.dump(markov.corpus, fp)

    size = input("input number of sentences: ")
    starting_word = input("input starting word pls: ")

    out = markov.generate_text(int(size), starting_word)
    print(out)

if __name__ == "__main__":
    main()