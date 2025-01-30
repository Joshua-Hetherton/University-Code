## Reading a File
# filename="Week_9\grades (1).csv"
# in_file=open(filename,"r")
# for line in in_file:
#     pass
# in_file.close()

## Writing to a file
# out_file=open("Week_9\Class_grades.txt","w")
# for student in students:
#     out_file.write("")
# out_file.close()

##Linked Lists
# A list containing another list
# Example:
#   split_line=line.split(",")
#   students.append([split_line[0],int(split_line[1])])


#Rounding and displaying it to 2dp
#student_range=[range,round((range/75)*100,2)]
# can also use:
#   :.2f
#   (average_mark/75)*100:.2f

## sum
#   gets the sum of all things

## len
#   returns the length of something

## min/max
# gets the minimum/maximum value

##Finding the Lowest/Highest mark in a list
#lowest_mark=min(student[1] for student in students)

##Linked lists
# students=[]
# student=["Name", 19, "SoftwareEng", 90]
# students.append(student)


# :.2f	    2 decimal places	        {:.2f}	    123.46
# :e	    Scientific notation	        {:e}	    1.234560e+02
# :.2e	    Sci. notation (2 decimals)	{:.2e}	    1.23e+02
# :g	    General (auto switch)	    {:g}	    123.456
# :%	    Percentage	                {:.1%}	    12345.6%
# :08.2f	Zero-padded width 8	        {:08.2f}	000123.46
# :>8.2f	Right-align width 8	        {:>8.2f}	123.46
# :<8.2f	Left-align width 8	        {:<8.2f}	123.46
# :^8.2f	Center-align width 8	    {:^8.2f}	123.46