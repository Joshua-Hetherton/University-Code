#1
# filename="Week_11\example.txt"
# try:
#     in_file=open(filename,"r")
# except FileNotFoundError:
#     print("File Not found")



#2
# fruits=["apple","banana","cherry"]
# get_index=int(input("Enter an index to find the fruit: "))
# try:
#     print(fruits[get_index])
# except IndexError:
#     print("Error with the Index value given")



#3
# try:
#     age=int(input("Enter your Age: "))
# except ValueError:
#     print("Error with user value given")
    


#4
# number=input("Enter a number: ")
# try:
#     result="Hello World x"+ str(int(number))
#     print(result)



# except TypeError:
#     print("Problem with adding your number to the string")
# except ValueError:
#     print("Value Error")



#5
# user_input=input("Enter a file path: ")

# try:
#     in_file=open(user_input,"r")
#     for line in in_file:
#         print(line, end="")
#     in_file.close()
# except FileNotFoundError:
#     print(f"Error Opening the file, try again")

# print("File Operation completed")



#6
# user_input=input("Enter a file path: ")
# user_index=int(input("Enter a Index: "))
# lines=[]
# try:
#     in_file=open(user_input,"r")
#     lines=in_file.readlines()
#     print(lines[user_index])
#     in_file.close()
# except FileNotFoundError:
#     print(f"Error Opening the file, try again")
# except IndexError:
#     print(f"Index Error")



#7
# user_file=input("Enter a path: ")
# try:
#     in_file=open(user_file,"r")
#     lines=in_file.readlines()


# except FileNotFoundError:
#     print("File Not found!")

#I got no Clue!
#I got no Clue!
#I got no Clue!



#8
# list_input=input("Enter a list of numbers seperated by spaces: ")
# try:

#     numbers=list(map(int, list_input.split(" ")))
#     print(numbers)
# except ValueError:
#     print("There was a value Error")



#9
try:
    user_name=input("Enter your name")
    if(user_name.count()==0):
        raise ValueError
except ValueError:
    print("A value Error has occured")
except TypeError:
    print("Typer Error has occured")

