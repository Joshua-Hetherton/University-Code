from tkinter import *

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

Entry(signup_frame,width=30).place(x=180,y=203)
Entry(signup_frame,width=30).place(x=180,y=243)
Entry(signup_frame,width=30).place(x=180,y=283)

Button(signup_frame, text="Sign Up Now!", font=("Arial",10),bg="White").place(x=175,y=310)
Button(signup_frame, text="Login Page!", font=("Arial",10),bg="White",command=lambda : show_frame(login_frame)).place(x=180,y=340)

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