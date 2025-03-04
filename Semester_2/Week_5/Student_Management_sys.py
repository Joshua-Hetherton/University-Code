from tkinter import *
import sqlite3
from tkinter import messagebox
#Creating a Database
conn=sqlite3.connect("student_db.db")
c=conn.cursor()

#Creating a Table
c.execute("""CREATE TABLE IF NOT EXISTS students(
          name text,
          studentID integer,
          class text,
          email text,
          phone integer
          ) """)

#Allows for a submission of a single entry
def submit():
    getname=name_entry.get()
    getstudentid=studentid_entry.get()
    getclass=class_entry.get()
    getemail=email_entry.get()
    getphone=phone_entry.get()

    c.execute("""INSERT INTO students(name, studentID, class, email, phone) VALUES(?, ?, ?, ?, ?) """,
              (getname,getstudentid,getclass,getemail,getphone)
              )
    conn.commit()

    #Clearing fields
    name_entry.delete(0,END)
    studentid_entry.delete(0,END)
    class_entry.delete(0,END)
    email_entry.delete(0,END)
    phone_entry.delete(0,END)

    print("Records Submitted")

#Retrieves all records from the database
def retrieve():
     c.execute("SELECT * FROM students")
     records=c.fetchall()
     for i in range(len(records)):
        Singleentry=Label(retrieveentry_frame,text=f"{records[i]}")
        Singleentry.place(x=50,y=300)

#Deletes ALL records
def deleterecords():
     confirm=messagebox.askyesno("Confirm Deletion","Are you sure you want to delete all records?")
     if confirm:
          c.execute("DELETE FROM students")
          conn.commit()

          print("All Records Deleted")




#Screen Config
win=Tk()
win.title("Student Management System")
win.config(bg="Light Blue")
win.geometry("450x600")

def show_frame(frame):
        frame.tkraise()

#Creating Data Entry Frame and other frame
data_entry_frame=Frame(win, bg="Light Blue",width=450,height=600)
retrieveentry_frame=Frame(win, bg="Light Blue",width=450,height=600)

for frame in (data_entry_frame,retrieveentry_frame):
    frame.place(x=0,y=0)

#Data Entry Frame
Label(data_entry_frame, text="Enter Name",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=200)
Label(data_entry_frame, text="Enter StudentID",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=220)
Label(data_entry_frame, text="Enter Class",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=240)
Label(data_entry_frame, text="Enter Email",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=260)
Label(data_entry_frame, text="Enter Phone",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=280)

name_entry=Entry(data_entry_frame,width=30)
name_entry.place(x=180, y=203)
studentid_entry=Entry(data_entry_frame,width=30)
studentid_entry.place(x=180, y=223)
class_entry=Entry(data_entry_frame,width=30)
class_entry.place(x=180, y=243)
email_entry=Entry(data_entry_frame,width=30)
email_entry.place(x=180, y=263)
phone_entry=Entry(data_entry_frame,width=30)
phone_entry.place(x=180, y=283)

#Retrieve Entry Frame
Button(retrieveentry_frame, text="Test", font=("Arial",10),bg="White", command=lambda: retrieve()).place(x=175,y=190)

#Buttons
Button(data_entry_frame, text="Submit Information", font=("Arial",10),bg="White", command=lambda: submit()).place(x=175,y=310)
Button(data_entry_frame, text="Retrieve Information", font=("Arial",10),bg="White", command=lambda : show_frame(retrieveentry_frame)).place(x=175,y=340)
Button(data_entry_frame, text="Delete Records", font=("Arial",10),bg="White", command=lambda : deleterecords()).place(x=175,y=370)


show_frame(data_entry_frame)
win.mainloop()
conn.close()