import numpy as np
import json
import nltk
import re

class MrMarkov:

    '''
    Idea nabbed from markovify source
    '''    
    BEGIN = "__BEGIN__"
    END = "__END__"
    EMPTY = "__EMPTY__"


    def __init__(self, filename=None, corpus_filename=None):

        if corpus_filename:
            
            self.corpus_filename = corpus_filename
            print("[MrMarkov] Detected loaded corpus from {}, not training.".format(self.corpus_filename))

            self.theme = self.corpus_filename.split('.')[0]

            with open(corpus_filename) as json_corpus:
                self.corpus = json.load(json_corpus)

        else:
            self.corpus = None

            if not filename:
                raise Exception("[AngryMrMarkov] Filename must be provided to make corpus!")
            
            self.filename = filename

            self.theme = self.filename.split('.')[0]


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

        if self.corpus:
            return

        self.corpus = {}

        print("[MrMarkov] Opening file: {}".format(self.filename))
        with open(self.filename, 'r', encoding='utf-8') as file:

            print("[MrMarkov] Processing lines... (this may take a while)")
            for _, line in enumerate(file):
                line = line.encode('ascii', errors='ignore').strip().decode('ascii')

                tokens = nltk.word_tokenize(line)
                tagged = nltk.pos_tag(tokens)

                if len(tagged) == 0:
                    continue

                new_line_array = []

                new_line_array.append((self.BEGIN, self.EMPTY))

                for i, pair in enumerate(tagged):
                    _, pos = pair
                    new_line_array.append(pair)
                    if pos == '.' and i != len(tagged)-1:
                        new_line_array.append((self.END, self.EMPTY))
                        new_line_array.append((self.BEGIN, self.EMPTY))

                new_line_array.append((self.END, self.EMPTY))

                for i in range(len(new_line_array)-2):

                    first_word = new_line_array[i][0]
                    second_word = new_line_array[i+1][0]
                    third_word = new_line_array[i+2]

                    key_word = "{} {}".format(first_word, second_word)                    

                    self.add_word((key_word, self.EMPTY), third_word)

            print("[MrMarkov] Done processing.")
            file.close()
            print("[MrMarkov] File closed.")
    
        print("[MrMarkov] Normalizing data... (this may take a while)")
        self.normalize()
        print("[MrMarkov] Done normalizing data!")

        print("[MrMarkov] Saving corpus as json...")
        with open('{}.json'.format(self.theme), 'w') as fp:
            json.dump(self.corpus, fp)
        print("[MrMarkov] Done saving \"{}.json\"".format(self.theme))
        
    
    def generate_text(self, num_sentences, starting_word):
        """
        docstring
        """
        text = []

        starting_word = "{} {}".format(self.BEGIN, starting_word)

        if starting_word not in self.corpus.keys():
            starting_word = np.random.choice(list(self.corpus.keys()))
            print("[MrMarkov] Word not found in corpus, using random start: {}".format(starting_word))

        starting_words = starting_word.split()
        text.append(starting_words[0])
        text.append(starting_words[1])

        counter = 0
        text_index = 1

        while counter < num_sentences:

            key = " ".join([text[text_index-1], text[text_index]])

            if key not in self.corpus.keys():
                matching_keys = list(filter(lambda c_key: text[text_index] in c_key, self.corpus.keys()))
                key = np.random.choice(matching_keys, size=1)[0]

            pos_lists, pos_probs = self.corpus[key]

            index_list = range(len(pos_lists))
            index = np.random.choice(a=index_list, p=pos_probs, size=1)[0]

            word_list, word_probs = pos_lists[index]
            word = np.random.choice(a=word_list, p=word_probs, size=1)[0]

            if word == self.END:
                counter += 1

            text.append(word)
            text_index += 1

        text = list(filter(lambda word: word not in [self.BEGIN, self.END], text))
        
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
    #markov = MrMarkov(filename="stories.txt")
    markov = MrMarkov(filename="batman.txt")

    markov.markovifile()

    size = input("input number of sentences: ")
    starting_word = input("input starting word pls: ")

    out = markov.generate_text(int(size), starting_word)
    print(out)

if __name__ == "__main__":
    main()