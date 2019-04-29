import numpy as np
import json
import nltk

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
        for word_one in self.corpus.keys():
            pos_sum = 0
            poss = []
            probs_pos = []

            for pos in self.corpus[word_one].keys():
                tally_sum = 0
                words_two = []
                probs_two = []

                for word_two, tally in self.corpus[word_one][pos][0].items():
                    tally_sum += tally
                    words_two.append(word_two)
                    probs_two.append(tally)

                two_probs = [float(prob)/tally_sum for prob in probs_two]
                
                pos_sum += self.corpus[word_one][pos][1]
                poss.append(pos)
                probs_pos.append(self.corpus[word_one][pos][1])

                self.corpus[word_one][pos][0] = [words_two, two_probs]

            
            pos_probs = [float(prob)/pos_sum for prob in probs_pos]

            self.corpus[word_one] = [poss, pos_probs]

    def add_word(self, first_word, second_word):
        """
        a function to handle words
        """
        
        (word_one, pos_one) = first_word
        (word_two, _) = second_word

        # if corpus -> word one
        if word_one in self.corpus.keys():
            # if word one -> part of speech
            if pos_one in self.corpus[word_one].keys():
                # tally count for word one -> part of speech
                self.corpus[word_one][pos_one][1] += 1
                # if word one -> part of speech[0] -> word two
                if word_two in self.corpus[word_one][pos_one][0]:
                    self.corpus[word_one][pos_one][0][word_two] += 1
                # if word one -> part of speech[0] !-> word two
                else:
                    self.corpus[word_one][pos_one][0][word_two] = 1
            # if word one !-> part of speech
            else:
                self.corpus[word_one][pos_one] = [{word_two: 1}, 1]
        # if corpus !-> word one
        else:
            self.corpus[word_one] = {pos_one: [{word_two: 1}, 1]}
        
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

                if li == 3:
                    break;

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

    print(markov.corpus['softly'])

    # with open('test.json', 'w') as fp:
    #     json.dump(markov.corpus, fp)

    # size = input("input text size: ")
    # starting_word = input("input starting word pls: ")

    # out = markov.generate_text(int(size), starting_word)
    # print(out)

if __name__ == "__main__":
    main()