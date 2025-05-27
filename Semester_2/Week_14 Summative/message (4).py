from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Global users are defined in the read_student_data():
# Use this to check against the currently logged user

win = Tk()
win.title("University of Winchester")
win.geometry("600x500")
bg_colour="#6B275C"


student_filename="Semester_2\\Week_14 Summative\\students.txt"
lecturer_filename="Semester_2\\Week_14 Summative\\lecturers.txt"
logo_filename="Semester_2\\Week_14 Summative\\University_Logo.png"

def show_frame(frame):
    frame.tkraise()

def read_student_details():
    details_all_filled = True# Checks each entry to make sure its filled
    details = [name_input.get(),pronouns_input.get(),birthday_input.get(),address_input.get(),term_address_input.get(),emergency_contact_name_input.get(),emergency_contact_number_input.get(),course_input.get(),email_input.get(),password_input.get()]
    for specific_detail in details:
        if specific_detail == "":
            details_all_filled = False

    if details_all_filled == False:
        messagebox.showerror("Error", "Not all details filled")
        show_frame(student_registration_frame)

    elif details_all_filled == True:
        details_to_file(name_input.get(),email_input.get(),password_input.get(), pronouns_input.get(), birthday_input.get(), address_input.get(),term_address_input.get(), emergency_contact_name_input.get(), emergency_contact_number_input.get(), course_input.get())
        name_input.delete(0, END),email_input.delete(0, END),password_input.delete(0, END),pronouns_input.delete(0, END),birthday_input.delete(0, END),address_input.delete(0, END),term_address_input.delete(0, END),emergency_contact_name_input.delete(0, END),emergency_contact_number_input.delete(0, END), course_input.delete(0, END)
        messagebox.showinfo("Submitted","Registration Successful!")
        show_frame(main_menu_frame)

def details_to_file(name,email,password,pronoun,dob,home_address,term_address,contact_name,contact_number,course) :
    with open(student_filename, 'a') as out_file:
        out_file.write(f"{name},{email},{password},{pronoun},{dob},{home_address},{term_address},{contact_name},{contact_number},{course}\n")
    out_file.close()

def check_staff_details():
    read_staff_file = open(student_filename, 'r')
    for file_line in read_staff_file:
        seperated_lines = file_line.strip().split(',')
        staff_name = seperated_lines[0]
        staff_email = seperated_lines[1]
        staff_password = seperated_lines[2]

        print(f"Stored Name: {staff_name}  Stored Password: {staff_password} Stored email: {staff_email}")
        if staff_email.lower() == staff_email_login.get().lower() and staff_password == staff_password_login.get():
            staff_email_login.delete(0, END)
            staff_password_login.delete(0, END)
            show_frame(staff_logged_in_frame)
            welcome_label_staff = Label(staff_logged_in_frame, text =f"Welcome, {staff_name}", fg = "white", bg = "#6B275C", font = "arial").place(x = 215, y = 20)
            return
    else:
        messagebox.showerror("Error", "Incorrect Credentials")
        return
    

def update_student_info():
    updates_all_filled = True# Checks each entry to make sure its filled
    update_details =[name_input_update.get(),pronouns_input_update.get(),birthday_input_update.get(),address_input_update.get(),term_address_input_update.get(),emergency_contact_name_input_update.get(),emergency_contact_number_input_update.get(),course_input_update.get(),email_input_update.get(),password_input_update.get() ]
    for specific_detail in update_details:
        if specific_detail == "":
            updates_all_filled = False
    # Gives error if not all details are filled
    if updates_all_filled == False:
        
        messagebox.showerror("Error", "Not all details filled")
        show_frame(student_data_frame)


    elif updates_all_filled == True:
        


        update_read_file = open(student_filename, 'r')
        all_lines = update_read_file.readlines()
        print(all_lines)

        print(f"email: {user_email} password: {user_password}")

        for i in range(len(all_lines)):

            # Split the line into components
            seperated_lines = all_lines[i].strip().split(',')

            if seperated_lines[1] == user_email and seperated_lines[2] == user_password:
                # Update the line with the new details
                all_lines[i] = f"{name_input_update.get()},{email_input_update.get()},{password_input_update.get()},{pronouns_input_update.get()},{birthday_input_update.get()},{address_input_update.get()},{term_address_input_update.get()},{emergency_contact_name_input_update.get()},{emergency_contact_number_input_update.get()},{course_input_update.get()}\n"
                # Printing the updated line for debugging
                print(f"Updated line: {all_lines[i]}")
                break

        ## Writes the changed all_lines back to the file
        out_file = open(student_filename, "w")
        out_file.writelines(all_lines)
        out_file.close()
            
        

    # elif updates_all_filled == True:

    #     current_logged_student = email_student_login.get()
    #     update_read_file = open(student_filename, 'r')
    #     all_lines = update_read_file.readlines()
    #     update_write_file = open(student_filename, 'w')

    #     for line in all_lines:
    #         seperated_lines = line.strip().split(',')

    #         if seperated_lines[1] == current_logged_student:

    #             update_write_file.write(name_input_update.get(),pronouns_input_update.get(),birthday_input_update.get(),address_input_update.get(),term_address_input_update.get(),emergency_contact_name_input_update.get(),emergency_contact_number_input_update.get(),course_input_update.get(),email_input_update.get(),password_input_update.get())
            
    #         else:

    #             update_write_file.write(line)

def read_student_data():  
    global user_email, user_password

    read_file = open(student_filename, 'r')
    for line in read_file:
        seperated_lines = line.strip().split(',')
        current_name = seperated_lines[0]
        current_email = seperated_lines[1]
        current_password = seperated_lines[2]
        current_pronouns = seperated_lines[3]
        current_dob = seperated_lines[4]
        current_address = seperated_lines[5]
        current_term_address = seperated_lines[6]
        current_Econtact_name = seperated_lines[7]
        current_Econtact_number = seperated_lines[8]
        current_course = seperated_lines[9]
        # print(f"Stored Name: {current_name}  Stored Password: {current_password} Stored email: {current_email}")
        if current_email.lower() == email_student_login.get().lower() and current_password == password_student_login.get():

            user_email = current_email
            user_password = current_password

            email_student_login.delete(0, END)
            password_student_login.delete(0, END)

            show_frame(student_data_menu_frame)
            welcome_label = Label(student_data_menu_frame, text =f"Welcome, {current_name}", fg = "white", bg = "#6B275C", font = "arial").place(x = 215, y = 20)

            pronouns = Label(student_data_menu_frame, text = "Pronouns:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 85)
            current_student_pronouns = Label(student_data_menu_frame, text = current_pronouns, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 85)

            birthday = Label(student_data_menu_frame, text = "Date of Birth:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50, y = 120)
            current_student_birthday = Label(student_data_menu_frame, text = current_dob, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 120)

            address = Label(student_data_menu_frame, text = "Home address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50,y=155)
            current_student_address = Label(student_data_menu_frame, text = current_address, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 155)

            term_address = Label(student_data_menu_frame, text = "Term time Address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 190)
            current_student_term_address = Label(student_data_menu_frame, text = current_term_address, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 190)

            emergency_contact_name = Label(student_data_menu_frame, text = "Emergency Contact name:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 225)
            current_student_Econtact_name = Label(student_data_menu_frame, text = current_Econtact_name, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 225)

            emergency_contact_number = Label(student_data_menu_frame, text = "Emergency Contact number:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 260)
            current_student_Econtact_number = Label(student_data_menu_frame, text = current_Econtact_number, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 260)

            course = Label(student_data_menu_frame, text = "Course:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 295)
            current_student_pronouns = Label(student_data_menu_frame, text = current_course, font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 300, y = 295)
            return
    else:
        messagebox.showerror("Error", "Incorrect Credentials")
        return  


main_menu_frame = Frame(win, bg = "#6B275C", width = 600, height = 500)
student_login_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
staff_login_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
staff_logged_in_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
student_registration_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
student_data_menu_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
student_data_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
main_menu_frame.pack()

for frame in (main_menu_frame, staff_login_frame, student_login_frame, student_registration_frame,student_data_frame,student_data_menu_frame,staff_logged_in_frame):
    frame.place(x=0,y=0)

# main menu---------------------------------------------
button_studentreg = Button(main_menu_frame, text = "Student Registration", bg = "white", fg = "black", command=lambda: show_frame(student_registration_frame)).place(x = 235, y = 225)
button_studentlog = Button(main_menu_frame, text = "Student Login", bg = "white", fg = "black", command=lambda: show_frame(student_login_frame)).place(x = 250, y = 275)
button_stafflog = Button(main_menu_frame, text = "Staff Login", bg = "white", fg = "black", command=lambda: show_frame(staff_login_frame)).place(x = 257, y = 325)

img_Winchester_header = Image.open(logo_filename)
img_Winchester_header = ImageTk.PhotoImage(img_Winchester_header)

label = Label(main_menu_frame, image= img_Winchester_header)
label.place(x=0,y=0)

#student registration------------------------------------
name = Label(student_registration_frame, text = "Name:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 50)
name_input = Entry(student_registration_frame,width = 25)
name_input.place(x = 260, y = 52)

pronouns = Label(student_registration_frame, text = "Pronouns:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 85)
pronouns_input = Entry(student_registration_frame,width = 25)
pronouns_input.place(x = 260, y = 87)

birthday = Label(student_registration_frame, text = "Date of Birth:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50, y = 120)
birthday_input = Entry(student_registration_frame,width = 25)
birthday_input.place(x = 260, y = 122)

address = Label(student_registration_frame, text = "Home address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50,y=155)
address_input = Entry(student_registration_frame,width = 25)
address_input.place(x=260,y=157)

term_address = Label(student_registration_frame, text = "Term time Address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 190)
term_address_input = Entry(student_registration_frame,width = 25)
term_address_input.place(x = 260, y =192)

emergency_contact_name = Label(student_registration_frame, text = "Emergency Contact name:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 225)
emergency_contact_name_input = Entry(student_registration_frame,width = 25)
emergency_contact_name_input.place(x = 260, y = 227)

emergency_contact_number = Label(student_registration_frame, text = "Emergency Contact number:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 260)
emergency_contact_number_input = Entry(student_registration_frame,width = 25)
emergency_contact_number_input.place(x = 260, y = 262)

course = Label(student_registration_frame, text = "Course:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 295)
course_input = Entry(student_registration_frame,width = 25)
course_input.place(x = 260, y = 297)

email = Label(student_registration_frame, text = "Email:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 330)
email_input = Entry(student_registration_frame,width = 25)
email_input.place(x = 260, y = 332)

password = Label(student_registration_frame, text = "Password:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 365)
password_input = Entry(student_registration_frame,width = 25)
password_input.place(x = 260, y = 367)

button_return = Button(student_registration_frame, text = "Return to main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 50, y = 420)
button_submit = Button(student_registration_frame, text = "Submit Registration", bg = "white", fg = "black", command=lambda: read_student_details()).place(x = 230, y = 420)

#student Login---------------------------------------
img_Winchester_logo = Image.open(logo_filename)
img_Winchester_logo = ImageTk.PhotoImage(img_Winchester_logo)
winchester_log_label = Label(student_login_frame, image= img_Winchester_logo).place(x=190,y=0)

email_label = Label(student_login_frame, text = "Email:", font = "arial",bg = "#6B275C",fg = "white" , justify= "left").place(x = 150, y = 250)
password_label = Label(student_login_frame, text = "Password:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 150, y = 280)

email_student_login = Entry(student_login_frame,width = 25)
email_student_login.place(x = 250, y = 252)
password_student_login = Entry(student_login_frame,width = 25,show="*")
password_student_login.place(x = 250, y = 282)

button_return = Button(student_login_frame, text = "Main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 180, y = 350)
button_submit = Button(student_login_frame, text = "Log in", bg = "white", fg = "black",command=lambda: read_student_data()).place(x = 320, y = 350)

#Student data menu Frame------------------------------------
button_return = Button(student_data_menu_frame, text = "Main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 160, y = 420)
button_submit = Button(student_data_menu_frame, text = "Change Student Info", bg = "white", fg = "black", command=lambda: show_frame(student_data_frame)).place(x = 350, y = 420)

#Student update data frame
name = Label(student_data_frame, text = "Name:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 50)
name_input_update = Entry(student_data_frame,width = 25)
name_input_update.place(x = 260, y = 52)

pronouns = Label(student_data_frame, text = "Pronouns:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 85)
pronouns_input_update = Entry(student_data_frame,width = 25)
pronouns_input_update.place(x = 260, y = 87)

birthday = Label(student_data_frame, text = "Date of Birth:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50, y = 120)
birthday_input_update = Entry(student_data_frame,width = 25)
birthday_input_update.place(x = 260, y = 122)

address = Label(student_data_frame, text = "Home address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x=50,y=155)
address_input_update = Entry(student_data_frame,width = 25)
address_input_update.place(x=260,y=157)

term_address = Label(student_data_frame, text = "Term time Address:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 190)
term_address_input_update = Entry(student_data_frame,width = 25)
term_address_input_update.place(x = 260, y =192)

emergency_contact_name = Label(student_data_frame, text = "Emergency Contact name:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 225)
emergency_contact_name_input_update = Entry(student_data_frame,width = 25)
emergency_contact_name_input_update.place(x = 260, y = 227)

emergency_contact_number = Label(student_data_frame, text = "Emergency Contact number:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 260)
emergency_contact_number_input_update = Entry(student_data_frame,width = 25)
emergency_contact_number_input_update.place(x = 260, y = 262)

course = Label(student_data_frame, text = "Course:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 295)
course_input_update = Entry(student_data_frame,width = 25)
course_input_update.place(x = 260, y = 297)

email = Label(student_data_frame, text = "Email:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 330)
email_input_update = Entry(student_data_frame,width = 25)
email_input_update.place(x = 260, y = 332)

password = Label(student_data_frame, text = "Password:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 50, y = 365)
password_input_update = Entry(student_data_frame,width = 25)
password_input_update.place(x = 260, y = 367)

button_return = Button(student_data_frame, text = "Return to main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 160, y = 420)
button_submit = Button(student_data_frame, text = "Update Information", bg = "white", fg = "black", command=lambda: update_student_info()).place(x = 350, y = 420)

#Staff login-------------------------------------------
img_Winchester_logo2 = Image.open(logo_filename)
img_Winchester_logo2 = ImageTk.PhotoImage(img_Winchester_logo2)
winchester_log_label2 = Label(staff_login_frame, image= img_Winchester_logo2).place(x=190,y=0)

email_label = Label(staff_login_frame, text = "Email:", font = "arial",bg = "#6B275C",fg = "white" , justify= "left").place(x = 150, y = 250)
password_label = Label(staff_login_frame, text = "Password:", font = "arial", bg = "#6B275C",fg = "white" , justify= "left").place(x = 150, y = 280)

staff_email_login = Entry(staff_login_frame,width = 25)
staff_email_login.place(x = 250, y = 252)
staff_password_login = Entry(staff_login_frame,width = 25,show="*")
staff_password_login.place(x = 250, y = 282)

button_return = Button(staff_login_frame, text = "Main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 180, y = 350)
button_submit = Button(staff_login_frame, text = "Log in", bg = "white", fg = "black", command=lambda: check_staff_details()).place(x = 320, y = 350)

#staff Logged in frame---------------------------------

button_return = Button(staff_logged_in_frame, text = "Return to main menu", bg = "white", fg = "black", command=lambda: show_frame(main_menu_frame)).place(x = 150 , y = 420)
button_submit = Button(staff_logged_in_frame , text = "Update Information", bg = "white", fg = "black", command=lambda: read_student_data()).place(x = 350, y = 420)



show_frame(main_menu_frame)
win.mainloop()