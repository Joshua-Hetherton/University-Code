from tkinter import *
from tkinter import messagebox

# =================== GUI Setup ===================
window=Tk()
window.title("University of Winchester")
window.geometry("700x600")
bg_colour="#6b2b63"



#Loading the image
def Get_Logo(frame,x1,y1):
    img = PhotoImage(file="Semester_2\\Week_14 Summative\\University_Logo.png")
    img = img.subsample(1, 1)
    label = Label(frame, image=img)
    #Putting it onto the frame so that a white square doesnt appear
    label.image = img
    label.place(x=x1, y=y1)


def show_frame(frame):
        frame.tkraise()

# All Frames
main_menu=Frame(window, bg=bg_colour, width=700, height=600)
student_registration=Frame(window, bg=bg_colour, width=700, height=600)
student_login=Frame(window, bg=bg_colour ,width=700, height=600)
lecturer_login=Frame(window, bg=bg_colour, width=700, height=600)
student_data=Frame(window, bg=bg_colour, width=700, height=600)
student_change_data=Frame(window, bg=bg_colour, width=700, height=600)
view_students=Frame(window, bg=bg_colour, width=700 , height=600)


for frame in (main_menu,student_registration,student_login,lecturer_login,student_data,student_change_data,view_students):
    frame.place(x=0,y=0)



# Main Menu
Get_Logo(main_menu,290,30)
Button(main_menu, text="Student Login", font=("Arial",10),bg="White", command=lambda : show_frame(student_login)).place(x=300,y=170)
Button(main_menu, text="Student Register", font=("Arial",10),bg="White", command=lambda : show_frame(student_registration)).place(x=295,y=200)


# =================== Student Registration ===================
# •	Name
# •	Pronouns
# •	Date of Birth
# •	Home Address
# •	Term Time Address (if different)
# •	Emergency Contact Name
# •	Emergency Contact Number
# •	Course

#Placing Logo
Get_Logo(student_registration,290,30)

#Displays Labels
label_names=["Name","Password","Pronouns","DOB","Home Address","Term Address","Emergency Contact Name","Emergency Contact No.","Course"]

for i, label_text in enumerate(label_names):
     label=Label(student_registration,text=label_text,bg=bg_colour, font=("Arial",10), fg="#ffffff")
     label.place(x=170,y=170 + (i*30))

#Entries
entries=[]
for i in range(len(label_names)):
     entry=Entry(student_registration)
     entry.place(x=330, y=170 + (i * 30))
     entries.append(entry)



def submit():
    name = entries[0].get()
    password = entries[1].get()
    pronouns = entries[2].get()
    dob=entries[3].get()
    address=entries[4].get()
    term_address=entries[5].get()
    emergency_contact_name=entries[6].get()
    emergency_contact_no=entries[7].get()
    course=entries[8].get()

    if not all([name,password, pronouns, dob, address, term_address, emergency_contact_name, emergency_contact_no, course]):
        messagebox.showerror("Error", "Missing Fields, Please Complete them before continuing!")
    
    out_file=open("students.txt", "a")
    out_file.write(f"{name},{password},{pronouns},{dob},{address},{term_address},{emergency_contact_name},{emergency_contact_no},{course}\n")
    out_file.close()
    print("Submitted")
    messagebox.showinfo("Success", "Registration Successful!")

# Back to Main Menu and Submit Button
Button(student_registration, text="Back to Main Menu", font=("Arial",10),bg="White", command=lambda : show_frame(main_menu)).place(x=290,y=500)

# Submit Button
Button(student_registration, text="Submit", font=("Arial",10),bg="White", command=lambda : submit()).place(x=320,y=530)

# =================== Student Login ===================
def check_credentials():
    name = entries[0].get()
    password = entries[1].get()

    if not all([name, password]):
        messagebox.showerror("Error", "Please fill in both fields.")
        return

    try:
        with open("students.txt", "r") as file:
            for line in file:
                stored_name, stored_password, *_ = line.strip().split(',')
                if stored_name == name and stored_password == password:
                    messagebox.showinfo("Success", "Login Successful!")
                    show_frame(student_data)
                    return
            messagebox.showerror("Error", "Invalid credentials. Please try again.")
    except FileNotFoundError:
        messagebox.showerror("Error", "No registered students found.")


Get_Logo(student_login,290,30)
Label(student_login, text="Name", font=("Arial",10), bg=bg_colour, fg="#ffffff").place(x=250,y=200)
Label(student_login, text="Password", font=("Arial",10), bg=bg_colour, fg="#ffffff").place(x=250,y=240)

Entry(student_login, width=30).place(x=320, y=203)
Entry(student_login, width=30).place(x=320, y=243)

Button(student_login, text="Login", font=("Arial",10), bg="White", command=lambda : (messagebox.showinfo("Success", "Login Successful!"), show_frame(student_data))).place(x=320,y=300)


# =================== Displaying GUI ===================
show_frame(main_menu)
window.mainloop()
