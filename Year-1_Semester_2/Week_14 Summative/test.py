# Label(student_registration, text="Name ",bg="Green",font=("Arial",10),fg="Black").place(x=210,y=100)
# student_name=Entry(student_registration,width=30).place(x=145, y=223)

# Label(student_registration, text="Pronouns ",bg="Green",font=("Arial",10),fg="Black").place(x=210,y=150)
# student_pronouns=Entry(student_registration,width=30).place(x=145, y=273)

##You can do it as a dictionary or array, whichever you prefer,
#
# dictionary={
#     "Name":student_name,
#     "Pronouns":student_pronouns,
# }

# array1=[student_name,student_pronouns]

# or you skip that directly and put it straight into the writefile

# student_name="ELO"  ##Bc im lazy af, its the same as "student_name=Entry(student_registration,width=30).place(x=145, y=223)"
# student_pronouns="Mr Blue Sky"
# out_file=open("Semester_2\Week_14","w")
# out_file.write(f"{student_name} {student_pronouns}\n")

out_file=open("students", "w")
out_file.write("Hello")
