import math
#
#Reading the file

students_names=[]
student_grades=[]

filename="Week_9\grades (1).csv"
in_file=open(filename,"r")

for line in in_file:
   split_line=line.split(",")
   students_names.append(split_line[0])
   student_grades.append(int(split_line[1]))

in_file.close()

#
#All marks as a percentage
student_grades_percentage=[]
index=0
for student_grade in student_grades:
    grade_percentage=(student_grade/75)*100

    student_grades_percentage.append(grade_percentage)
    print(f"{students_names[index]}: {grade_percentage:.2f}%")
    index+=1
print()

#
#Highest mark and the name of the student(raw and percentage)
highest_mark=max(student_grades)
index_of_highest_mark=student_grades.index(highest_mark)


print(f"The Highest mark achieved was by {students_names[index_of_highest_mark]} with a raw score of {max(student_grades)} and a percentage of {student_grades_percentage[index_of_highest_mark]:.2f}%\n")

#
#Lowest mark and the name of the student(raw and percentage)
lowest_mark=min(student_grades)
index_of_lowest_mark=student_grades.index(lowest_mark)

print(f"The Lowest mark achieved was by {students_names[index_of_lowest_mark]} with a raw score of {min(student_grades)} and a percentage of {student_grades_percentage[index_of_lowest_mark]:.2f}%\n")


#
#Range of all the marks(raw and %)
student_range=max(student_grades)-min(student_grades)
print(f"The Range of all students marks is {student_range} and {max(student_grades_percentage)-min(student_grades_percentage):.2f}%\n")

#
#The average marks(raw and %)
average_mark=sum(student_grades)/len(student_grades)
print(f"The average mark is {math.floor(average_mark)} and the average percentage is {(average_mark/75)*100:.2f}%\n")

#
#Number of A's, B's, C's, D's and E's
letter_grades=[]

for grade in student_grades_percentage:
    
    if(grade>=80):
        letter_grades.append("A")
    
    elif(grade>=70):
        letter_grades.append("B")
    
    elif(grade>=60):
        letter_grades.append("C")        
    
    elif(grade>=50):
        letter_grades.append("D")
    
    elif(grade>=40):
        letter_grades.append("E")
    
    elif(grade<40):
        letter_grades.append("F")

#
#Counting the numbers of each grade achieved
number_of_As=letter_grades.count("A")
number_of_Bs=letter_grades.count("B")
number_of_Cs=letter_grades.count("C")
number_of_Ds=letter_grades.count("D")
number_of_Es=letter_grades.count("E")
print(f"""Number of A's:{number_of_As},
Number of B's:{number_of_As},
Number of C's:{number_of_As},
Number of D's:{number_of_As},
Number of E's:{number_of_As}\n""")

#
#Number of Fails
number_of_F=letter_grades.count("F")
print(f"Number of Fails:{number_of_F}\n")

#

#formatting to write to the file
all_Student_grades=[]
for student in students_names:
    all_Student_grades.append(f"{students_names[students_names.index(student)]},{student_grades[students_names.index(student)]},{student_grades_percentage[students_names.index(student)]:.2f}")

student_statistics=[]

#Outputting to the file
#Writing to the file(not appending)
out_file=open("Week_9\Class_grades.txt","w")
index=0
for student in students_names:
    out_file.write(f"{all_Student_grades[index]}\n")
    index+=1

out_file.write(f"{highest_mark},{students_names[index_of_highest_mark]},{student_grades_percentage[index_of_highest_mark]:.2f}\n")
out_file.write(f"{lowest_mark},{students_names[index_of_lowest_mark]},{student_grades_percentage[index_of_lowest_mark]:.2f}\n")

out_file.write(f"{max(student_grades)-min(student_grades)},{max(student_grades_percentage)-min(student_grades_percentage):.2f}\n")

out_file.write(f"{math.floor(average_mark)},{(average_mark/75)*100:.2f}\n")

out_file.write(f"{number_of_As},{number_of_Bs},{number_of_Cs},{number_of_Ds},{number_of_Es},{number_of_F}")

out_file.close()

#
#Teachers Interface
teacher_interface=input("""1.Grades
2.Highest Mark
3.Lowest Mark
4.Range
5.Average
6.Letter Grades
7.Exit
""")
match teacher_interface:
    case "1":
        print("test")
    case "2":
        print("test")

    case "3":
        print("test")

    case "4":
        print("test")
    
    case "5":
        print("test")

    case "6":
        print("test")
    
    case "7":
        print("test")
      
   