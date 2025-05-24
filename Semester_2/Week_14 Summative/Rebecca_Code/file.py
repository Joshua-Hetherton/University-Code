from tkinter import *
from tkinter import messagebox

win = Tk()
win.title("University of Winchester")
# win.iconbitmap("uni.png")
win.configure(bg="#6b2b63")
win.geometry("750x600")

def show_frame(frame):
    frame.tkraise()

signup_frame = Frame(win, bg="#6b2b63", width = "750", height = "600")
login_frame = Frame(win, bg="#6b2b63", width = "750", height = "600")
welcome_frame = Frame(win, bg="#6b2b63", width = "750", height = "600")

for frame in (signup_frame, login_frame, welcome_frame):
    frame.place(x=0, y=0)

#Load the image and display the logo
# img = PhotoImage(file="uni.png")    
# img=img.subsample(2,2)
# label = Label(signup_frame, image = img).place()

#Creating A Database
import sqlite3
conn = sqlite3.connect("univeristy2.db")
c = conn.cursor()

#Table
c.execute("""CREATE TABLE IF NOT EXISTS students(
            first_name text,
            last_name text,
            address text,
            postcode text,
            phone integer,
            pronouns text,
            date_of_birth text,
            term_address text,
            emergency_contact_name text,
            emergency_contact_number integer,
            course text
                 )""")



    # Correct INSERT statement
# c.execute("""INSERT INTO students(
#                     first_name, last_name, address, postcode, phone, pronouns,
#                     date_of_birth, term_address, emergency_contact_name,
#                     emergency_contact_number, course
#                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#               (first_name_data, last_name_data, address_data, postcode_data, phone_data,
#                pronouns_data, date_of_birth_data, term_address_data,
#                emergency_contact_name_data, emergency_contact_number_data, course_data))

#     conn.commit()


    # #Clearing the text field
    # first_name.delete(0, END)
    # last_name.delete(0, END)
    # address.delete(0, END)
    # postcode.delete(0, END)
    # phone.delete(0, END)
    # pronouns.delete(0, END)
    # dob.delete(0, END)
    # term.delete(0, END)
    # contact_name.delete(0, END)
    # contact_number.delete(0, END)
    # course.delete(0, END)

    # print("Your details have been submitted")

#Fetching records
def query():
    c.execute("SELECT * FROM students")
    records = c.fetchall()
    print(records)
    print_records = ""
    for record in records:
        print_records += str(record) + "\n"
        query_lb.config(text=print_records)
query_lb = Label(win, text = "", bg="#6b2b63", fg="white", font = ("Ovo", 13))
query_lb.place(x=120, y=450)




#Deleting records
def delete():
    Confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete record?")
    c.execute("DELETE FROM students")
    conn.commit()
    query_lb.config(text = "All records are deleted", fg = "red")
    print("All records are deleted!")

#Sign up Frame
Label(signup_frame, text = "First Name", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=195)
Label(signup_frame, text = "Last Name", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=220)
Label(signup_frame, text = "Address", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=245)
Label(signup_frame, text = "Postcode", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=270)
Label(signup_frame, text = "Phone", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=295)
Label(signup_frame, text = "Pronouns", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=320)
Label(signup_frame, text = "Date of Birth", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=345)
Label(signup_frame, text = "Term Date", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=370)
Label(signup_frame, text = "Emergency Contact", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=395)
Label(signup_frame, text = "Emergency Contact No", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=420)
Label(signup_frame, text = "Course", font = ("Aptos Serif", 15), bg="#6b2b63", fg="#af98ac" ).place(x=180, y=445)




first_name_entry = Entry(signup_frame, width = 30)
first_name_entry.place(x=400, y=200)
last_name_entry = Entry(signup_frame, width = 30)
last_name_entry.place(x=400, y=225)
address_entry = Entry(signup_frame, width = 30)
address_entry.place(x=400, y=250)
postcode_entry= Entry(signup_frame, width = 30)
postcode_entry.place(x=400, y=275)
phone_entry = Entry(signup_frame, width = 30)
phone_entry.place(x=400, y=300)
pronouns_entry = Entry(signup_frame, width = 30)
pronouns_entry.place(x=400, y=325)
dob_entry = Entry(signup_frame, width = 30)
dob_entry.place(x=400, y=350)
term_entry = Entry(signup_frame, width = 30)
term_entry.place(x=400, y=375)
contact_name_entry = Entry(signup_frame, width = 30)
contact_name_entry.place(x=400, y=400)
contact_number_entry = Entry(signup_frame, width = 30)
contact_number_entry.place(x=400, y=425)
course_entry = Entry(signup_frame, width = 30)
course_entry.place(x=400, y=450)

# values={
#     "First Name":first_name,
#     "Last Name":last_name,
#     "Address": address,
#     "Postcode": postcode,
#     "Date of Birth": dob,
#     "Term Start": term,
#     "Emergency Contact Name": contact_name,
#     "Emergency Contact Number": contact_number,
#     "Course": course
# }


# values=[first_name, last_name, address, postcode, phone, pronouns, dob, term, contact_name, contact_number, course]


# filename = "Students.txt"
# out_file=open(filename,"w")
# out_file.write(f"{first_name} {last_name} {address}{postcode} {dob} {term} {contact_name} {contact_number} {course}\n")

#Buttons
Button(signup_frame, text = "Enter", bg = "#af98ac", fg = "#6b2b63", command=lambda:write_to_file()).place(x=250, y=500)
Button(signup_frame, text = "Show Account Details", bg = "#af98ac", fg = "#6b2b63", command = query).place(x=300, y=500)
Button(signup_frame, text = "Clear", bg = "#af98ac", fg = "#6b2b63", command = delete).place(x=440, y=500)
# Button(signup_frame, text="Student Login", bg="#af98ac", fg="#6b2b63", command=submit).place(x=120, y=350)
# Button(signup_frame, text="Admin Login", bg="#af98ac", fg="#6b2b63", command=submit).place(x=120, y=350)

def write_to_file():
    first_name = first_name_entry.get()
    last_name = last_name_entry.get()
    address = address_entry.get()
    postcode = postcode_entry.get()
    phone = phone_entry.get()
    pronouns = pronouns_entry.get()
    dob = dob_entry.get()
    term = term_entry.get()
    contact_name = contact_name_entry.get()
    contact_number = contact_number_entry.get()
    course = course_entry.get()

    out_file = open("students.txt", "w")
    out_file.write(f"{first_name}, {last_name}, {address}, {postcode}, {phone}, {pronouns}, {dob}, {term}, {contact_name}, {contact_number}, {course} ")
    out_file.close()
#Login Frame

# img = PhotoImage(file='uni.png')
# img = img.subsample(3,3)
# label = Label(signup_frame, image = img).place(x=300, y = 20)

Label(login_frame, text = 'Email', font = ('Aptos Serif', 15), bg ='#6b2b63', fg = '#af98ac').place(x=50, y = 200)
Label(login_frame, text = 'Password', font = ('Aptos Serif', 15), bg ='#6b2b63', fg = '#af98ac').place(x=50, y = 240)

Entry(login_frame, width=30).place(x=180, y = 203)
Entry(login_frame, width=30).place(x=180, y = 243)

Button(login_frame, text = 'Sign up now', bg = '#af98ac', fg = '#6b2b63').place(x = 120, y = 350)
Button(login_frame, text = "Login here", bg = "#af98ac", fg = "#6b2b63").place(x=230, y=350)


show_frame(signup_frame)
win.mainloop()
conn.close()