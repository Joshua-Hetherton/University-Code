import math

students=[]

##Reading the Text file
filename="Week_9\grades (1).csv"

in_file=open(filename,"r")

for line in in_file:
    split_line=line.split(",")
    students.append([split_line[0],int(split_line[1])])
in_file.close()

##All marks as a percentage
index=0
for student in students:
    grade_percentage=(students[index][1]/75)*100
    students[index].append(f"{round(grade_percentage,2)}")
    index+=1

#Highest mark and the name of the student(raw and percentage)
highest_mark=max(student[1] for student in students)
index_of_highest_mark=[student[1] for student in students].index(highest_mark)

best_student=students[index_of_highest_mark]

#Lowest mark and the name of the student(raw and percentage)
lowest_mark=min(student[1] for student in students)
index_of_lowest_mark=[student[1] for student in students].index(lowest_mark)

worst_student=students[index_of_lowest_mark]

#Range of all the marks(raw and %)
range=highest_mark-lowest_mark
student_range=[range,round((range/75)*100,2)]

#The average marks(raw and %)
average_mark=sum(student[1] for student in students)/len(students)
student_average=[average_mark,round((average_mark/75)*100,2)]

#Letter Grades Achieved(As,Bs,Cs,Ds,Es,Fs)
letter_grades = []
index=0
for student in students:
    grade = student[2]
    if(grade>="80"):
        letter_grades.append("A")
        students[index].append("A")

    elif(grade>="70"):
        letter_grades.append("B")
        students[index].append("B")
    
    elif(grade>="60"):
        letter_grades.append("C")   
        students[index].append("C")  
    
    elif(grade>="50"):
        letter_grades.append("D")
        students[index].append("D")
    
    elif(grade>="40"):
        letter_grades.append("E")
        students[index].append("E")
    
    elif(grade<"40"):
        letter_grades.append("F")
        students[index].append("F")
    index+=1

number_of_As=letter_grades.count("A")
number_of_Bs=letter_grades.count("B")
number_of_Cs=letter_grades.count("C")
number_of_Ds=letter_grades.count("D")
number_of_Es=letter_grades.count("E")


##Writing to the file
out_file=open("Week_9\Class_grades.txt","w")
for student in students:
    out_file.write(f"{student}\n")

#Writing the Highest and lowest mark achieved
out_file.write(f"{highest_mark}\n{lowest_mark}\n")

#Writing the range
out_file.write(f"{student_range}\n")

#Writing the average
out_file.write(f"{student_average}\n")

out_file.write(f"{letter_grades}")

out_file.close()
loop=True
while loop:
    teacher_interface=input("Enter Name of a Student to find their Raw mark, Percentage and Letter grade, or Enter Exit ").title()
    if(teacher_interface=="Exit"):
        loop=False
        break

    print(teacher_interface)
    search_for_student=[student[0] for student in students].index(teacher_interface)
    print(f"""Name: {students[search_for_student][0]},
Mark:{students[search_for_student][1]},
Grade: {students[search_for_student][2]}%,
Letter Grade: {students[search_for_student][3]}""")
