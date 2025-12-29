import math
white_sock=10
black_sock=10

def way_1():
    #prob of drawing 2 of the same colour socks
    total_socks=white_sock+black_sock
    prob_white=white_sock/total_socks
    prob_black=black_sock/total_socks
    prob_same_colour=(prob_white* (white_sock-1)/(total_socks-1)) + (prob_black* (black_sock-1)/(total_socks-1))
    print("Probability of drawing 2 socks of the same colour is:", prob_same_colour)

def way_2():
    ##using binomial coefficient
    total_socks=white_sock+black_sock
    ways_to_choose_2_socks_of_same_colour=math.comb(white_sock,2) + math.comb(black_sock,2)
    total_ways_to_choose_2_socks=math.comb(total_socks,2)
    result=ways_to_choose_2_socks_of_same_colour/total_ways_to_choose_2_socks
    print(result)
    

if __name__ == "__main__":
    way_1()
    way_2()
