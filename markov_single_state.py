import numpy as np
import json

class MrMarkov:

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

            for li, line in enumerate(file):
                line = line.encode('ascii', errors='ignore').strip().decode('ascii')

                line_array = line.split()

                if len(line_array) == 0:
                    continue

                if len(line_array) == 1:

                    new_word = line_array[0]
                    
                    self.add_word("\n", new_word)
                    self.add_word(new_word, "\n")
                    continue

                for i in range(len(line_array)):

                    if i == 0:
                        self.add_word("\n", line_array[i])
                        self.add_word(line_array[i], line_array[i+1])
                        continue

                    if i == len(line_array) - 1:
                        self.add_word(line_array[i], "\n")
                        continue

                    first_word = line_array[i]
                    second_word = line_array[i+1]

                    self.add_word(first_word, second_word)

            file.close()

            self.normalize()
    
    def generate_text(self, len_text, starting_word):
        """
        """
        text = []

        if starting_word not in self.corpus.keys():
            rand_word = np.random.choice(list(self.corpus.keys()))

            ans = input("starting_word needs to be a word in the corpus, do you want to try " + rand_word + "? y/n: ")

            if ans == 'y' or 'Y':
                return self.generate_text(len_text, rand_word)
            else:
                print("Goodbye loser") 
                exit  

        text.append(starting_word)

        for i in range(len_text):
            words, prob = self.corpus[text[i]]
            next_word = np.random.choice(words, p=prob, size=1)[0]
            text.append(next_word)
        
        return " ".join(text)

def main():
    markov = MrMarkov(filename="batman.txt")

    markov.markovifile()

    with open('test.json', 'w') as fp:
        json.dump(markov.corpus, fp)

    size = input("input text size: ")
    starting_word = input("input starting word pls: ")

    out = markov.generate_text(int(size), starting_word)
    print(out)

if __name__ == "__main__":
    main()