user_input=input("Enter a file path: ")
## Week_11\names.txt
lines=[]
try:
    in_file=open(user_input,"r")

    for line in in_file:
        lines.append(line)

    in_file.close()
except PermissionError:
    print("You Dont have permission to read this file")
except FileNotFoundError:
    print(f"Error Opening the file, try again")
else:
    for single_line in lines:
        print(single_line,end="")
finally:
    print("Well thanks for reading a text file!")
