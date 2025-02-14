word = 'bullshit'

guesses = []
user_input = ''

while user_input != '0':

    user_input = input('Enter a letter, or 0 to give up:')
    guesses.append(user_input)
    output = ''
    for letter in range(0, len(word)):
        if word[letter] in guesses:
            output = output + word[letter]
        else:
            output = output + '_'
    print(output)
    if output == word:
        print('You win!')
        break
    #Added this so that the game has a limited number of guesses

    elif (len(guesses)>10):
        break



print('Game over!')