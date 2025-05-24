from tkinter import *
from tkinter import messagebox

## GUI setup
window=Tk()
window.geometry("500x600")

def show_frame(frame):
        frame.tkraise()

main_menu=Frame(window, bg="Red",width=500,height=600)
student_registration=Frame(window, bg="Green",width=500,height=600)
student_login=Frame(window,bg="Blue",width=500,height=600)
lecturer_login=Frame(window,bg="Purple",width=500,height=600)
student_data=Frame(window,bg="Yellow",width=500,height=600)
student_change_data=Frame(window,bg="Pink",width=500,height=600)
view_students=Frame(window,bg="Light Blue",width=500,height=600)


for frame in (main_menu,student_registration,student_login,lecturer_login,student_data,student_change_data,view_students):
    frame.place(x=0,y=0)

# Main Menu
Button(main_menu, text="Student Login", font=("Arial",10),bg="White", command=lambda : show_frame(student_login)).place(x=135,y=340)
Button(main_menu, text="Student Register", font=("Arial",10),bg="White", command=lambda : show_frame(student_registration)).place(x=135,y=370)

#Requirement 1 – Student Registration
# The data which needs to be collected is:
# •	Name
# •	Pronouns
# •	Date of Birth
# •	Home Address
# •	Term Time Address (if different)
# •	Emergency Contact Name
# •	Emergency Contact Number
# •	Course

Label(student_registration, text="Name ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=100)
Label(student_registration, text="Pronouns ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=150)
Label(student_registration, text="Date of Birth ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=200)
Label(student_registration, text="Home Address ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=250)
Label(student_registration, text="Term Time Address ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=300)
Label(student_registration, text="Emergency Contact Name ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=350)
Label(student_registration, text="Emergency Contact Number ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=400)
Label(student_registration, text="Course ", bg="Green", font=("Arial",10), fg="Black").place(x=210,y=450)

student_name_entry = Entry(student_registration, width=30)
student_name_entry.place(x=145, y=123)

student_pronouns_entry = Entry(student_registration, width=30)
student_pronouns_entry.place(x=145, y=173)

student_DOB_entry = Entry(student_registration, width=30)
student_DOB_entry.place(x=145, y=223)

student_home_address_entry = Entry(student_registration, width=30)
student_home_address_entry.place(x=145, y=273)

student_term_address_entry = Entry(student_registration, width=30)
student_term_address_entry.place(x=145, y=323)

student_emergency_name_entry = Entry(student_registration, width=30)
student_emergency_name_entry.place(x=145, y=373)

student_emergency_number_entry = Entry(student_registration, width=30)
student_emergency_number_entry.place(x=145, y=423)

student_course_entry = Entry(student_registration, width=30)
student_course_entry.place(x=145, y=473)

student_name_entry = Entry(student_registration, width=30)
student_name_entry.place(x=145, y=123)

Button(student_registration, text="Register", font=("Arial",10),bg="White", command=lambda :write_to_file()).place(x=135,y=540)


## Needs editing to just use an array cus this is too much--------------

def write_to_file():
    student_name = student_name_entry.get()
    student_pronouns = student_pronouns_entry.get()
    student_DOB = student_DOB_entry.get()
    student_home_address = student_home_address_entry.get()
    student_term_address = student_term_address_entry.get()
    student_emergency_name = student_emergency_name_entry.get()
    student_emergency_number = student_emergency_number_entry.get()
    student_course = student_course_entry.get()

    if not all(student_name,student_pronouns,student_DOB,student_home_address,student_term_address,student_emergency_name,student_emergency_number,student_course):
         messagebox.show("Error", "Missing Fields, Please Complete them before continuing")

    out_file=open("students", "w")
    out_file.write(f"{student_name},{student_pronouns},{student_DOB},{student_home_address},{student_term_address},{student_emergency_name},{student_emergency_number},{student_course}")
    out_file.close()
    print("Submitted")
#--------------------
#Requirement 2 – Student Login
#Requirement 3 – Lecturer Login
#Requirement 4 – Display Single Student Data
#Requirement 5 – Single Student Update Data
#Requirement 6 – Lecturer Student Display




# Displaying the GUI
show_frame(main_menu)
window.mainloop()