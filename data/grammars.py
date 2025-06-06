from lib2to3.pgen2 import grammar
from nltk import CFG

grammar_cfg3b  = """
22 -> 21 20
22 -> 20 19
19 -> 16 17 18
19 -> 17 18 16
20 -> 17 16 18
20 -> 16 17
21 -> 18 16
21 -> 16 18 17
16 -> 15 13
16 -> 13 15 14
17 -> 14 13 15
17 -> 15 13 14
18 -> 15 14 13
18 -> 14 13
13 -> 11 12
13 -> 12 11
14 -> 11 10 12
14 -> 10 11 12
15 -> 12 11 10
15 -> 11 12 10
10 -> 7 9 8
10 -> 9 8 7
11 -> 8 7 9
11 -> 7 8 9
12 -> 8 9 7
12 -> 9 7 8
7 -> "c" "a"
7 -> "a" "b" "c"
8 -> "c" "b"
8 -> "c" "a" "b"
9 -> "c" "b" "a"
9 -> "b" "a" 
"""

grammar_simple =  """
22 -> 14 15
22 -> 15 14
14 -> 11 10 12
14 -> 10 11 12
15 -> 12 11 10
15 -> 11 12 10
10 -> 7 9 8
10 -> 9 8 7
11 -> 8 7 9
11 -> 7 8 9
12 -> 8 9 7
12 -> 9 7 8
7 -> "c" "a"
7 -> "a" "b" "c"
8 -> "c" "b"
8 -> "c" "a" "b"
9 -> "c" "b" "a"
9 -> "b" "a" 
"""


GRAMMAR_CFG3b = CFG.fromstring(grammar_cfg3b)
GRAMMAR_CFG3b_string = grammar_cfg3b

GRAMMAR_SIMPLE = CFG.fromstring(grammar_simple)
GRAMMAR_SIMPLE_string = grammar_simple