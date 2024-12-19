def has_exclamation_mark(word):
    return word.endswith("!!")

def is_uppercase(word):
    return word.isupper()

def is_valid_word(word):
    return not is_uppercase(word) and not has_exclamation_mark(word)

def filter_words(words):
    filtered = filter(is_valid_word, words)
    return list(filtered)

user_input=input("Enter as many words as you want to check if they will get excluded. Seperate with Commas: ").split(",")
filtered_list=filter_words(user_input)
print(f"{filtered_list} is done!")

