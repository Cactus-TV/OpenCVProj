import pandas as pd
import numpy as np
from symspellpy import SymSpell, Verbosity
import os
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.trans_ = {}
        self.ids_ = []

    def add_word(self, word, id):
        node = self
        last_id = None

        for w in word:
            node = node.trans_.setdefault(w, InvertedIndex())
            if node.ids_ and node.ids_[-1] == id:
                continue
            node.ids_.append(id)

    def find(self, prefix):
        node = self

        for p in prefix:
            if p not in node.trans_:
                return None
            node = node.trans_[p]

        return node


kInvalidId = ~0
inverted_index = InvertedIndex()


class Sugg:
    def __init__(self, text, line):
        self.text_ = text
        self.line_ = line

forward_index = []

def build_index(in_stream):
    global forward_index
    line_num = 0
    for line in in_stream:
        line = line.rstrip()
        line_num += 1
        id = len(forward_index)
        forward_index.append(Sugg(line, line_num))
        words = line.split(' ')
        for word in words:
            if word:
                inverted_index.add_word(word, id)


def intersect_sorted_lists(lists, top):
    res = []
    lnum = len(lists)
    if lnum == 0:
        return 0
    if lnum == 1:
        res = lists[0][:]
        res = res[:min(top, len(res))]
        return res
    lptrs = [(iter(lst), iter(lst).__next__()) for lst in lists if lst]
    if len(lptrs) != lnum:
        return False
    li, found = 0, 0
    cur_id = lptrs[0][1]
    while len(res) < top:
        try:
            while lptrs[li][1] < cur_id:
                lptrs[li] = (lptrs[li][0], lptrs[li][0].__next__())
            check_id = lptrs[li][1]
            if check_id == cur_id:
                found += 1
                if found == lnum:
                    res.append(cur_id)
                    lptrs[li] = (lptrs[li][0], lptrs[li][0].__next__())
                    if lptrs[li][1] is not None:
                        cur_id = lptrs[li][1]
                        found = 1
                    else:
                        break
            else:
                cur_id = check_id
                found = 1
            li = (li + 1) % lnum
        except StopIteration:
            break
    return res


def search(q, top):
    res = []
    lists = []
    words = q.split(' ')
    for word in words:
        if not word:
            continue
        node = inverted_index.find(word)
        if node is None:
            return []
        lists.append(node.ids_)
    if not lists:
        return []
    intersection = intersect_sorted_lists(lists, top)
    if not intersection:
        return []
    for i in intersection:
        res.append(forward_index[i])
    return res


def extract_hashtags(row):
    return (' '.join(word[1:] for word in str(row).split() if word.startswith('#'))).replace('#', ' ')

def creating_word_rate(path):
    df = pd.read_csv(path)
    df["description"] = df["description"].apply(extract_hashtags)
    df = df.dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']
    arr = df['description'].values
    words = [word for element in arr for word in element.split()] 
    return dict(Counter(words))

sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)

def create_trees():
    global sym_spell
    dictionary_path = "dict.txt"  # путь к вашему словарю
    df_csv = "yappy_hackaton_2024_400k.csv" # путь к csv с тэгами к видео
    print("Start creating words rate")
    word_dict = creating_word_rate(df_csv)
    print("Finished")
    phrases = np.genfromtxt(dictionary_path, dtype=str, delimiter="\n")
    #строим дерево подсказок
    print("Start creating tree of suggestions")
    with open(dictionary_path, 'r') as in_file:
        build_index(in_file)
    print(f"Successfully added {len(forward_index)} suggestions")
    # добавляем словосочетания в словарь с частотой встречаемости
    print("Start creating dict for autocorrect")
    for phrase in phrases:
        sym_spell.create_dictionary_entry(phrase, word_dict[phrase] if phrase in word_dict else 0)
    print(f"Successfully created dict for autocorrections")


# test
create_trees()
word = ""
search_words = [""]
os.system('clear')
print("\nВведите поисковый запрос на рус/англ (после каждого символа нажимай enter):\n")
while True:
    word += input()
    search_words[-1] += word[-1]
    os.system('clear')
    res = search(word, 3)#кол-во подсказок
    print(*search_words, "\n")
    for i, sugg in enumerate(res):
        print(f"{i} \t{sugg.text_}\n")
    if word[-1] == " ":
        search_words[-1] = search_words[-1][:len(word)-1]
        suggestions = sym_spell.lookup(word[:len(word)-1], Verbosity.CLOSEST, max_edit_distance=5)
        word = ""
        if len(suggestions):
            if not word in suggestions:
                search_words[-1] = suggestions[0].term
                print(f"Возможно вы имели в виду: {search_words[-1]}")
        search_words.append("")