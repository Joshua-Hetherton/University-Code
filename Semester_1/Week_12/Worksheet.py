#1
#error was with valid entry not being set to true when a valid number was entered
# def get_number_input():
#     valid_entry = False
#     while(not valid_entry):
#         input_number = input("Please enter a number: ")
#         if(input_number.isdigit()):
#             input_number = int(input_number)
#             valid_entry=True
#         else:
#             print("Sorry that is not a valid number - please try again")
#     return input_number


#2
#in input loop, it saved all inputs to index 1 instead of the rest of the list
# def say_hello_people():
#     names = ["","","","",""]
#     for i in range(5):
#         names[i] = input("Please enter your name: ")

#     for i in range(5):
#         print("Hello "+ names[i]+" how do you do?")
# say_hello_people()


#3
#should be divided by 3, not 4,
# def calculate_average():
#     num1 = get_number_input()
#     num2 = get_number_input()
#     num3 = get_number_input()
#     average_nums = (num1 + num2 + num3)/3.0
#     return average_nums


#4
##no clue what is wrong?
# def get_number_input():
#     valid_entry = False
#     while(not valid_entry):
#         input_number = input("Please enter a number: ")
#         if(input_number.isdigit()):
#             valid_entry = True
#         else:
#             print("Sorry that is not a valid number - please try again")
#     return input_number
# print(get_number_input())


#5
import math

def root_calculation(a, b, c):
    d = b ** 2 - 4 * a * c
    if d > 0:
        disc = math.sqrt(d)
        root1 = (-b + disc) / (2 * a)
        root2 = (b + disc) / (2 * a)
        print("Root 1 is: "+root1)
        print("Root 2 is: "+root2)
    elif d == 0:
        root = -b / (2 * a)
        print("The root is: " + root)
    else:
        print("This equation has no roots")
root_calculation(1,9,35)