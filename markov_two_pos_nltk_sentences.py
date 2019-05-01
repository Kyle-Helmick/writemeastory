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

        self.word_corpus = {}
        self.pos_corpus = {}
        self.theme = filename.split('.')[0]

        if corpus_filename:
            with open(corpus_filename, 'r') as json_corpus:
                corpus = json.load(json_corpus)
                self.pos_corpus = corpus['pos_corpus']
                self.word_corpus = corpus['word_corpus']

        else:
            if not filename:
                raise Exception("filename must be provided to make corpus!")
            
            self.filename = filename

    def normalize(self):
        """
        a function that normalizes the counts of words 
        """

        for word in self.word_corpus.keys():
            pos_list = []
            pos_tally_list = []
            pos_tally_total = 0

            for pos, tally in self.word_corpus[word].items():
                pos_list.append(pos)
                pos_tally_list.append(tally)
                pos_tally_total += tally

            pos_prob_list = [tally/pos_tally_total for tally in pos_tally_list]

            self.word_corpus[word] = [pos_list, pos_prob_list]

        for pos in self.pos_corpus.keys():
            word_list = []
            word_tally_list = []
            word_tally_total = 0

            for word, tally in self.pos_corpus[pos].items():
                word_list.append(word)
                word_tally_list.append(tally)
                word_tally_total += tally

            word_prob_list = [tally/word_tally_total for tally in word_tally_list]

            self.pos_corpus[pos] = [word_list, word_prob_list]

    def add_word(self, first_word, second_word):
        """
        a function to handle words
        """
        
        (word_one, _) = first_word
        (word_two, pos_two) = second_word

        # Make the relation of word to part of speech
        if word_one in self.word_corpus.keys():
            if pos_two in self.word_corpus[word_one].keys():
                self.word_corpus[word_one][pos_two] += 1
            else:
                self.word_corpus[word_one][pos_two] = 1
        else:
            self.word_corpus[word_one] = {pos_two: 1}
        
        # Make the relation of part of speech to word
        if pos_two in self.pos_corpus.keys():
            if word_two in self.pos_corpus[pos_two].keys():
                self.pos_corpus[pos_two][word_two] += 1
            else:
                self.pos_corpus[pos_two][word_two] = 1
        else:
            self.pos_corpus[pos_two] = {word_two: 1}

        
    def markovifile(self):
        with open(self.filename, 'r', encoding='utf-8') as file:

            for li, line in enumerate(file):
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

            file.close()

            self.normalize()
    
    def generate_text(self, num_sentences, starting_word):
        """
        """
        text = []

        starting_word = "{} {}".format(self.BEGIN, starting_word)

        if starting_word not in self.word_corpus.keys():
            starting_word = np.random.choice(list(self.word_corpus.keys()))
            print("[MrMarkov] Word not found in corpus, using random start: {}".format(starting_word))

        starting_words = starting_word.split()
        text.append(starting_words[0])
        text.append(starting_words[1])

        counter = 0
        text_index = 1

        while counter < num_sentences:

            word_key = " ".join([text[text_index-1], text[text_index]])

            if word_key not in self.word_corpus.keys():
                matching_keys = list(filter(lambda c_key: text[text_index] in c_key, self.word_corpus.keys()))
                word_key = np.random.choice(matching_keys, size=1)[0]

            pos_list, pos_probs = self.word_corpus[word_key]
            pos = np.random.choice(a=pos_list, p=pos_probs, size=1)[0]
            word_list, word_probs = self.pos_corpus[pos]
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
    markov = MrMarkov(filename="cyberslaves.txt")

    markov.markovifile()

    size = input("input text size: ")
    starting_word = input("input starting word pls: ")

    out = markov.generate_text(int(size), starting_word)
    print(out)

if __name__ == "__main__":
    main()