filename="Week_8\_rainfall.txt"

#Opening the file
in_file=open(filename,"r")

#Q1
#print(in_file.read().upper())

#2
# print(in_file.read().lower())

#3
# print(in_file.read().replace(" ",""))

#4
for line in in_file:
    #split the line
    split_line=line.strip(" ")

    # print(split_line[0],end=" ")
    for number in split_line:
        print(number,end=" ")




in_file.close()