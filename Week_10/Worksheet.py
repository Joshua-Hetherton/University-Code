import math

#1
def metre_conversion(number):
    
    feet=round(number*3.28, 2)
    inches=round(number*39.37, 2)
    yards=round(number*1.09, 2)

    output=[feet,inches,yards]
    return output

number=int(input("Enter a number"))
output=metre_conversion(number)
print(f"""{number} metres in feet is: {output[0]}
{number} metres in inches is: {output[1]}
{number} metres in yards is: {output[2]}\n""")


#2
# def calculate_total(number_of_coins):
#     one=number_of_coins[0]* 0.01
#     fives=number_of_coins[1]* 0.05
#     tens=number_of_coins[2]* 0.1
#     fifties=number_of_coins[3]* 0.5
#     pound=number_of_coins[4]
#     two_pounds=number_of_coins[5]* 2
    
#     total=one+fives+tens+fifties+pound+two_pounds
#     return total

# number_of_coins=list(map(int, input("Enter the Number of 1p, 5p, 10p, 50p, £1 and £2 coins: ").split()))
# print(f"Total is £{calculate_total(number_of_coins)}")


#3
# def get_number():
#     entered_number=float(input("Please Ennter a Number: "))
#     return entered_number
# #original code doesnt call the function

# #calling the function(fixing the code)
# print(get_number())


#4
# def happy_birthday(name, age):
#     print(f"Happy birthday {name}, well done for turning {age}!")

# happy_birthday("Josh", 18)


#5
# def diameter(radius):
#     return radius*2
# def circumference(radius):
#     return math.pi*2*radius
# def area(radius):
#     return math.pi*(radius**2)

# radius=int(input("Enter a number)"))
# print(f"""A circle of radius {radius} has:
# A diameter of {diameter(radius)}
# A circumferance of {circumference(radius)}
# An area of {area(radius)}""")


#6
# def celcius_to_fahrenheit(temperature):
#     return (1.8*temperature)+32

# get_temperature=int(input("Enter A temperature in C to convert to F: "))
# print(f"{get_temperature}C is the same as {celcius_to_fahrenheit(get_temperature)}F ")


#7
# def sum_list(numbers):
#     return sum(numbers)

# loop=False

# list_of_numbers=[]

# while loop!=True:
#     number=int(input("Enter a Number: "))
#     list_of_numbers.append(number)
#     end_loop=input("Do you want to end the Loop? Y/N ")   

#     if(end_loop=="Y"):
#         loop=True
# print(sum_list(list_of_numbers))


#8
# def find_max(numbers_list):
#     max=0
#     for number in numbers_list:
#         if(number>max):
#             max=number
#     return max

# numbers_list=[1,2,3,4,5,6,19999,7,8,9]

# print(find_max(numbers_list))


#9
# def is_prime(number):
#     if(number%2==0):
#         return False
#     else:
#         return True
    
# number=int(input("Input a number to test if it is Prime: "))
# print(is_prime(number))


#10
# def factorial(n):
#     return math.factorial(n)

# number=int(input("Input a number to find its factorial: "))
# print(factorial(number))


#11
# def count_vowels(word):
#     index=0
#     for char in word:
#         if(char=="a"):
#             index+=1
#         elif(char=="e"):
#             index+=1
#         elif(char=="i"):
#             index+=1
#         elif(char=="o"):
#             index+=1
#         elif(char=="u"):
#             index+=1
#     return index
# word=input("Enter a string to count the vowels: ")
# print(count_vowels(word.lower()))
