from pytorch_tutorial import Lang
from pytorch_tutorial import readLangs

if __name__ == '__main__':
    lang = Lang("english")
    lang.add_sentence("Hello my name is Pritish Yuvraj")
    lang.add_sentence("I am doing great Pritish")
    lang.print_word_stats()

    readLangs("eng", "french")
