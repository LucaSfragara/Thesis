import nltk
from nltk import CFG
from nltk.parse.chart import ChartParser
from data.grammars import GRAMMAR_CFG3b

class CFGParser:
    def __init__(self, grammar):
        """
        Initializes the parser with a CFG string in NLTK format.
        """
        self.grammar = grammar
        self.parser = ChartParser(self.grammar)

    def is_valid(self, sentence):
        """
        Returns True if the sentence is in the language defined by the CFG.
        """
        tokens = list(sentence)
        tokens_to_words = { 0: "a", 1: "b", 2: "c" }
        tokens = [tokens_to_words[int(token)] for token in tokens]
        try:
            trees = list(self.parser.parse(tokens))
            return len(trees) > 0
        except ValueError:
            return False

    def parse(self, sentence):
        """
        Returns a list of parse trees if the sentence is valid; otherwise, empty list.
        """
        tokens = sentence.split()
        try:
            return list(self.parser.parse(tokens))
        except ValueError:
            return []
        
if __name__ == "__main__":
    

    parser = CFGParser(GRAMMAR_CFG3b)
    #print(parser.is_valid("""312312132132123323213132112332321233213123213132313211232131221123312321232121123312313221213212331312321213212332321123323121313213123221123323132121313122112332312123213213231312123213232131123213123132321321313221313232313212112331231322112321312321313123132213121321233122132131231321313123132213213132"""))
    