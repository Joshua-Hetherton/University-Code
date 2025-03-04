from tkinter import *
import sqlite3

#Creating a Database
conn=sqlite3.connect("database.db")
c=conn.cursor()

#Creating a Table
c.execute("""CREATE TABLE IF NOT EXISTS student(
          email text,
          user text,
          password text
          ) """)

#Submit function
def submit():
      getemail=email.get()
      getusername=username.get()
      getpassword=password.get()

      c.execute("INSERT INTO student(email, user, password) VALUES (?, ?, ?)",
                (getemail,getusername,getpassword)
                )
      conn.commit()

      #Clearing the text fields
      email.delete(0,END)
      username.delete(0,END)
      password.delete(0,END)

      print("Records have been submitted")


win=Tk()
win.title("University of Winchester")
win.config(bg="Blue")
win.geometry("450x500")

def show_frame(frame):
        frame.tkraise()


#Frames
signup_frame=Frame(win, bg="Light Blue",width=450,height=500)
login_frame=Frame(win, bg="Light Blue",width=450,height=500)
welcome_frame=Frame(win, bg="Light Blue",width=450,height=500)

for frame in (signup_frame,login_frame,welcome_frame):
    frame.place(x=0,y=0)

#Load the image and logo
img=PhotoImage(file="Semester_2\Week_5\Friar_Tuck2.png")
img=img.subsample(2,2)
label=Label(signup_frame,image=img).place(x=175,y=20)
label=Label(login_frame,image=img).place(x=175,y=20)
label=Label(welcome_frame,image=img).place(x=175,y=20)

#Sign up Frame
Label(signup_frame, text="Enter Email",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=200)
Label(signup_frame, text="Username",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=240)
Label(signup_frame, text="Password",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=280)

email=Entry(signup_frame,width=30)
email.place(x=180, y=203)
username=Entry(signup_frame,width=30)
username.place(x=180, y=243)
password=Entry(signup_frame,width=30)
password.place(x=180, y=283)

Button(signup_frame, text="Submit Information", font=("Arial",10),bg="White", command=lambda: submit()).place(x=175,y=310)
Button(signup_frame, text="Sign Up Now!", font=("Arial",10),bg="White").place(x=175,y=340)
Button(signup_frame, text="Login Page!", font=("Arial",10),bg="White",command=lambda : show_frame(login_frame)).place(x=180,y=370)

#Login Frame
Label(login_frame, text="Email",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=200)
Label(login_frame, text="Password",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=240)

Entry(login_frame,width=30).place(x=180,y=203)
Entry(login_frame,width=30).place(x=180,y=243)

Button(login_frame, text="Sign Up Now!", font=("Arial",10),bg="White",command=lambda : show_frame(signup_frame)).place(x=175,y=310)
Button(login_frame, text="Login", font=("Arial",10),bg="White",command=lambda : show_frame(welcome_frame)).place(x=180,y=340)

#Welcome Frame
Label(welcome_frame, text="FRIAR TUCK SOCIETY!",bg="Light Blue",font=("Arial",15),fg="Black").place(x=50,y=200)
Button(welcome_frame, text="Back to Signup!", font=("Arial",10),bg="White",command=lambda : show_frame(signup_frame)).place(x=175,y=310)

show_frame(signup_frame)

win.mainloop()
conn.close()