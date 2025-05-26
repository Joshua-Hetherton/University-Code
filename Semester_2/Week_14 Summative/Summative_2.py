from tkinter import *
from tkinter import messagebox

#==================== Text File Information ====================
# The text files are formatted as follows:
# For Students:
# Name, Password, Email, Pronouns, Date of Birth, Home Address, Term Time Address, Emergency Contact Name, Emergency Contact Number, Course
# For Lecturers:
# Name, Password

# In the students.txt file, each field provided tests all cases, including empty fields and duplicate names (See Charlie fields)

# =================== Other Functions ===================
def get_field(fields, index):
    #Gets the field at the specified index, returns "N/A" if index is out of range
    #While this mostly likely wouldnt happen in the code, the file could be edited manually, causing error

    try:
        value = fields[index].strip()
        return value if value else "N/A"
    
    except IndexError:
        return "N/A"



# =================== GUI Setup ===================
window=Tk()
window.title("University of Winchester")
window.geometry("700x600")
bg_colour="#6b2b63"


#Loading the image
def get_logo(frame,x1,y1):
    img = PhotoImage(file="Semester_2\\Week_14 Summative\\University_Logo.png")
    img = img.subsample(1, 1)
    label = Label(frame, image=img)
    # Putting it onto the frame so that a white square doesnt appear
    label.image = img
    label.place(x=x1, y=y1)


def show_frame(frame):
        frame.tkraise()


# All Frames
main_menu = Frame(window, bg=bg_colour, width=700, height=600)
student_registration = Frame(window, bg=bg_colour, width=700, height=600)
universal_login = Frame(window, bg=bg_colour ,width=700, height=600)
student_data = Frame(window, bg=bg_colour, width=700, height=600)

view_students = Frame(window, bg=bg_colour, width=700 , height=600)


for frame in (main_menu, student_registration, universal_login, student_data, view_students):
    frame.place(x = 0, y = 0)




# =================== Main Menu ===================

get_logo(main_menu,290,30)
Button(main_menu, text = "Universal Login", font = ("Arial",10), bg = "White", command=lambda : show_frame(universal_login)).place(x=300,y=170)
Button(main_menu, text = "Student Register", font = ("Arial",10), bg = "White", command=lambda : show_frame(student_registration)).place(x=295,y=200)


# The Current Active User


# Specific user type buttons
# These are not displayed until a user has logged in (these are addded in the check_credentials function)
student_specific_button = Button(main_menu, text = "View Students", font = ("Arial",10), bg = "White", command=lambda : show_frame(view_students))
lecture_specific_button = Button(main_menu, text = "View Data", font = ("Arial",10), bg = "White", command=lambda: (load_student_data(), show_frame(student_data)))

# Logout Button
logout_button = Button(main_menu, text="Logout", font=("Arial", 10), bg="White", command=lambda: (student_specific_button.place_forget(), lecture_specific_button.place_forget(), show_frame(main_menu), messagebox.showinfo("Logged Out", "You have been logged out.")))



# =================== Student Registration ===================
# Name, Pronouns, Date of Birth, Home Address, Term Time Address, 
# Emergency Contact Name, Emergency Contact Number, Course


get_logo(student_registration,290,30)

# Displays Labels
label_names = ["Name", "Password", "Email", "Pronouns", "DOB", "Home Address", "Term Address", "Emergency Contact Name", "Emergency Contact No.", "Course"]


for i, label_text in enumerate(label_names):
     label=Label(student_registration,text=label_text,bg=bg_colour, font=("Arial",10), fg="#ffffff")
     label.place(x=170,y=170 + (i*30))

# Entries
entries=[]
for i in range(len(label_names)):
     entry=Entry(student_registration)
     entry.place(x=330, y=170 + (i * 30))
     entries.append(entry)



def submit():
    name = get_field(entries, 0).get().title()
    password = get_field(entries, 1).get()
    email=get_field(entries, 2).get()
    pronouns = get_field(entries, 3).get()
    dob = get_field(entries, 4).get()
    address= get_field(entries, 5).get().title()
    term_address= get_field(entries, 6).get().title()
    emergency_contact_name = get_field(entries, 7).get().title()
    emergency_contact_no = get_field(entries, 8).get()
    course = get_field(entries, 9).get().title()

    if not all([name,password,email, pronouns, dob, address, term_address, emergency_contact_name, emergency_contact_no, course]):
        messagebox.showerror("Error", "Missing Fields, Please Complete them before continuing!")
        return
    
    out_file = open("students.txt", "a")
    out_file.write(f"{name},{password},{email},{pronouns},{dob},{address},{term_address},{emergency_contact_name},{emergency_contact_no},{course}\n")
    out_file.close()
    print("Submitted")
    messagebox.showinfo("Success", "Registration Successful!")

# Back to Main Menu and Submit Button

# Submit Button
Button(student_registration, text="Submit", font=("Arial",10),bg="White", command=lambda : submit()).place(x=320,y=500)
# Main Menu button
Button(student_registration, text="Back to Main Menu", font=("Arial",10),bg="White", command=lambda : show_frame(main_menu)).place(x=290,y=530)




# =================== Universal Login ===================
def check_credentials():
    global current_user, current_password, current_user_type

    user_type=selected_option.get()
    name=login_name_entry.get()
    password=login_password_entry.get()

    # Used to access either lecturers or students text file
    filename = ""
    is_lecturer= None

    print(f" Name: {name}  Password: {password} Type: {user_type}")

    if not all([name, password]):
        messagebox.showerror("Error", "Please fill in both fields.")
        return
    #Checking User type to provide correct file
    if user_type == "Lecturer":
        filename="lecturers.txt"
        is_lecturer=True

    else:
        filename="students.txt"
        is_lecturer=False

    try:

        in_file=open(filename,"r")
        for line in in_file:
            split_line = line.strip().split(",") #.replace('\r', '').replace('\n', '')
            stored_name = get_field(split_line, 0)
            stored_password = get_field(split_line, 1)

            print(f"Stored Name: {stored_name}  Stored Password: {stored_password}")
            
            if stored_name == name and stored_password == password:
                print("Login Successful")
                messagebox.showinfo("Success", "Login Successful!")
                #Setting The Current User and Type
                current_user = name
                current_password = password
                current_user_type = user_type
                
                print(f"Current User: {current_user} Type: {current_user_type}")

                # Clearing the entries
                login_name_entry.delete(0, END)
                login_password_entry.delete(0, END)

                #Placing Logout button
                logout_button.place(x=600, y=10)


                if is_lecturer:
                    show_frame(main_menu)
                    student_specific_button.place(x=300, y=230)  # Shows the student-specific button
                
                else:
                    show_frame(main_menu)
                    lecture_specific_button.place(x=305, y=230) #Shows the lecturer-specific button
                return
        messagebox.showerror("Error", "Invalid credentials, please try again.")
             
        in_file.close()

    except FileNotFoundError:
        messagebox.showerror("Error", "No registered students found.")
        return

# GUI for Universal Login

def universal_login_gui():

    get_logo(universal_login,290,30)

    # https://www.geeksforgeeks.org/dropdown-menus-tkinter/ used as a reference for the dropdown menu
    # Dropdown Menu for selecting user type
    selected_option = StringVar()
    selected_option.set("Student")  # Default value
    OptionMenu(universal_login,selected_option, "Student", "Lecturer").place(x=320, y=170)

    #Name and Password Entries
    Label(universal_login, text="Name", font=("Arial",10), bg=bg_colour, fg="#ffffff").place(x=250,y=200)
    Label(universal_login, text="Password", font=("Arial",10), bg=bg_colour, fg="#ffffff").place(x=250,y=240)

    login_name_entry = Entry(universal_login, width=30)
    login_name_entry.place(x=320, y=203)

    login_password_entry = Entry(universal_login, width=30, show="*")
    login_password_entry.place(x=320, y=243)

    Button(universal_login, text="Login", font=("Arial",10), bg="White", command=lambda : check_credentials()).place(x=320,y=300)
    Button(universal_login, text="Back to Main Menu", font=("Arial",10), bg="White", command=lambda : show_frame(main_menu)).place(x=290,y=330)



# =================== Student Data ===================
# Displays the student data

def load_student_data():
    try:

        get_logo(student_data,290,30)


        print("Trying to load student data...")
        in_file=open("students.txt","r")
        lines= in_file.readlines()
        in_file.close()


        for line in lines:
            split_line = line.strip().split(",")#.replace('\r', '').replace('\n', '')
            stored_name = get_field(split_line, 0)
            stored_password = get_field(split_line, 1)
            print(f"Stored Name: {stored_name}  Current User: {current_user}")
            if stored_name == current_user and stored_password == current_password:
                print(f"Found data for {stored_name}")
                # Displaying the data in the labels

                for i, label_text in enumerate(label_names):
                    label=Label(student_data,text=f"{label_text}: ",bg=bg_colour, font=("Arial",10), fg="#ffffff")
                    label.place(x=170,y=170 + (i*30))

                    entry=Entry(student_data)
                    entry.place(x=330, y=170 + (i * 30))
                    entry.insert(0, get_field(split_line, i))
                    entries.append(entry)

        def save_changes():
            updated_data = [entry.get() for entry in entries]
            print("Data to save:", updated_data)  # Debug output
            if not all(updated_data):
                messagebox.showerror("Error", "Please complete all fields before saving!")
                return
        
            for i, line in enumerate(lines):
                if line.strip().split(",")[0] == current_user:
                    lines[i] = ",".join(updated_data) + "\n"
                    break

            out_file = open("students.txt", "w")
            out_file.writelines(lines)
            out_file.close()

            messagebox.showinfo("Success", "Data Saved")
        save_button = Button(student_data, text="Save Changes", font=("Arial", 10), bg="White", command=save_changes)
        save_button.place(x=300, y=500)
        ##Currently running into the problem where students can preview other students data
        # Add buttons inside the function so they are recreated every time
        Button(student_data, text="Back to Main Menu", font=("Arial", 10), bg="White", command=lambda: show_frame(main_menu)).place(x=290, y=530)



            

    except FileNotFoundError:
        messagebox.showerror("Error", "No registered students found.")
        return
    

    

# =================== View Students ===================


get_logo(view_students, 290, 30)


# Create Text widget for showing student data
text_widget = Text(view_students, wrap=WORD, width=60, height=20)
text_widget.place(x=50, y=200, width=600, height=300)  

# Creating Vertical Scrollbar
scrollbar = Scrollbar(view_students, command=text_widget.yview)
scrollbar.place(x=610, y=200, width=20, height=300)  
# Placed to the right of the Text widget

# Connecting scrollbar and text widget
text_widget.config(yscrollcommand=scrollbar.set)

# Reading from the students.txt file and displaying the data
in_file =open("students.txt","r")
for line in in_file:
        split_line = line.strip().replace('\r', '').replace('\n', '').split(",")
        # Start with the Name field on its own line:
        text_widget.insert(END, f"{get_field(split_line, 0)}\n")
        
        # For the rest, print label: value pairs, skipping index 1 if it’s unused
        for i in range(1, len(label_names)):
            label = label_names[i]
            value = get_field(split_line, i)
            text_widget.insert(END, f"{label}: {value}\n")
        
        text_widget.insert(END, "\n")  # Add an empty line after each student
in_file.close()
Button(view_students, text="Back to Main Menu", font=("Arial",10), bg="White", command=lambda : show_frame(main_menu)).place(x=290,y=530)


# =================== Displaying GUI ===================
show_frame(main_menu)
window.mainloop()
