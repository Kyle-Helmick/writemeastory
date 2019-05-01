import numpy as np
import json
import nltk
import re

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

        for word_one in self.corpus.keys():

            word_prob_list_pairs = []
            pos_tallies = []
            pos_tallies_sum = 0

            for pos in self.corpus[word_one].keys():
                # self.corpus[word_one][pos][0] is a dictionary of second words with associated probs
                # self.corpus[word_one][pos][1] is the tally of the part of speech
                
                word_two_list = []
                word_two_tallies = []
                word_two_tallies_sum = 0

                for word_two, tally in self.corpus[word_one][pos][0].items():

                    word_two_list.append(word_two)
                    word_two_tallies.append(tally)
                    word_two_tallies_sum += tally

                word_two_probs = [tally/word_two_tallies_sum for tally in word_two_tallies]

                word_prob_list_pairs.append([word_two_list, word_two_probs])
                pos_tallies.append(self.corpus[word_one][pos][1])
                pos_tallies_sum += self.corpus[word_one][pos][1]
            
            pos_probs = [tally/pos_tallies_sum for tally in pos_tallies]

            self.corpus[word_one] = [word_prob_list_pairs, pos_probs]

    def add_word(self, first_word, second_word):
        """
        a function to handle words
        """
        
        (word_one, _) = first_word
        (word_two, pos_two) = second_word

        # if corpus -> word one
        if word_one in self.corpus.keys():
            # if word one -> part of speech
            if pos_two in self.corpus[word_one].keys():
                # tally count for word one -> part of speech
                self.corpus[word_one][pos_two][1] += 1
                # if word one -> part of speech[0] -> word two
                if word_two in self.corpus[word_one][pos_two][0]:
                    self.corpus[word_one][pos_two][0][word_two] += 1
                # if word one -> part of speech[0] !-> word two
                else:
                    self.corpus[word_one][pos_two][0][word_two] = 1
            # if word one !-> part of speech
            else:
                self.corpus[word_one][pos_two] = [{word_two: 1}, 1]
        # if corpus !-> word one
        else:
            self.corpus[word_one] = {pos_two: [{word_two: 1}, 1]}
        
    def markovifile(self):
        with open(self.filename, 'r', encoding='utf-8') as file:

            for li, line in enumerate(file):
                line = line.encode('ascii', errors='ignore').strip().decode('ascii')

                tokens = nltk.word_tokenize(line)
                tagged = nltk.pos_tag(tokens)

                if len(tagged) == 0 or len(tagged) == 1:
                    continue

                for i in range(len(tagged)-1):

                    first_word = tagged[i]
                    second_word = tagged[i+1]

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
            pos_lists, pos_probs = self.corpus[text[i]]
            index_list = range(len(pos_lists))
            index = np.random.choice(a=index_list, p=pos_probs, size=1)[0]
            word_list, word_probs = pos_lists[index]
            word = np.random.choice(a=word_list, p=word_probs, size=1)[0]
            text.append(word)

        spaced_text = " ".join(text)

        spaced_text = re.sub(r" \,", ",", spaced_text)
        spaced_text = re.sub(r" \.", ".", spaced_text)
        spaced_text = re.sub(r" \?", "?", spaced_text)
        spaced_text = re.sub(r" !", "!", spaced_text)
        spaced_text = re.sub(r"\'\' ", "\"", spaced_text)
        spaced_text = re.sub(r" \`\`", "\"", spaced_text)
        spaced_text = re.sub(r" \' ", "'", spaced_text)
        spaced_text = re.sub(r" \'", "'", spaced_text)
        spaced_text = re.sub(r"\'\'", "\"", spaced_text)

        return spaced_text
        

def main():
    markov = MrMarkov(filename="batman.txt")

    markov.markovifile()

    # print(markov.corpus)

    with open('test.json', 'w') as fp:
        json.dump(markov.corpus, fp)

    size = input("input text size: ")
    starting_word = input("input starting word pls: ")

    out = markov.generate_text(int(size), starting_word)
    print(out)

if __name__ == "__main__":
    main()