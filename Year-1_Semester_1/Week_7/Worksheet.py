#1 D
# user_input=input("Enter a name")
# for i in user_input:
#     print(i)

#1 E
# sum=0
# numbers_list=[1, 2, 3, 4, 5]
# for number in numbers_list:
#     sum+=number
# print(sum)

#1 F
# sum=0
# user_input=input("Enter a number ")
# for number in user_input:
#     sum+=1
# print(sum)


#1 G
# user_input=int(input("Enter a number "))
# for i in range(21):
#     print(f"{user_input}*{i}={i*user_input}")

#1 H
# def Calculate_Interest(amount, interest, number_of_yrs):
#     i=0
    
#     while(i<number_of_yrs):
#         amount+=interest*amount
#         i+=1
#     print(amount)


# Calculate_Interest(2,0.02,2)

#1 I
# i=1
# while(i<8):
#     sum=i**3
#     print(sum)
#     i+=1

#1 J
# def Print_Square():
#     for i in range(5):

#         for t in range(5):
#             print(i+t, end=" " )
#         print()
            

# Print_Square()

#1 k
# def Every_3rd_Letter():
#     result=""
#     user_word=input("Enter a word")
#     for i in range(len(user_word)):
#         if((i+1)%3==0):
#             result+=user_word[i]
#     print(result)

# Every_3rd_Letter()

#1 L


#2
# user_input=input("Enter your name ")
# for i in range(10):
#     print(f"Hello {user_input}")

#3
# import random;
# heads=0
# tails=0
# for i in range(100):
#     output=random.randint(0,1)
#     if(output==1):
#         heads+=1
#     else:
#         tails+=1
# print(f"heads: {heads}\ntails: {tails}")
# print(f"heads was thrown {(heads/i)*100}% of the time")

#4
##See( 1 G )

#5
# import random;
# number_to_guess=random.randint(1,9)
# correct=False
# while(correct != True):
#     user_input=int(input("Guess a number between 0 & 9 "))
#     if(user_input==number_to_guess):
#         print("Exactly Right!")
#         correct=True
#     elif(user_input<number_to_guess):
#         print("Too Low!")
#     else:
#         print("Too High!")

#6
        