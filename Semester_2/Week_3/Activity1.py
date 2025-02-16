from tkinter import *

window=Tk()

#   #Task 1
# window.geometry("400x300")
# label1=Label(window, text="Bob is cool",font=("Arial",10),bg="#34abeb")
# label2=Label(window, text="Josh is Cool",font=("Arial",10),bg="#34abeb")

# label1.pack(pady=20)
# label2.pack(padx=10,pady=20)

# button1=Button(window, text="Click me!", font=("Arial",10), bg="white")
# button1.pack(padx=10)

# button2=Button(window, text="Dont Click me!", font=("Arial",10), bg="white")
# button2.pack(padx=50)

#   #Task2
# window.geometry("400x300")
# label1=Label(window, text="Welcome",font=("Arial",10),bg="#34abeb",fg="Red")
# label1.place(x=150,y=50)

# login_button=Button(window, text="Login", font=("Arial",10), bg="white")
# login_button.place(x=100,y=150)

# signup_button=Button(window, text="Sign Up", font=("Arial",10), bg="white")
# signup_button.place(x=250,y=150)


#   #Task3

window.geometry("400x500")
app=Label(window,text="Calculator", font=("Arial",20),bg="White")
app.grid(row=0,column=0,padx=3)



for i in range(1,4):
    button=Button(window, text=f"{i}", font=("Arial",30), bg="white")
    button.grid(row=1,column=i,padx=0)


window.mainloop()