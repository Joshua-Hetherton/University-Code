from tkinter import *
from PIL import ImageTk,Image
win=Tk()

win.config(bg="#34abeb")
win.geometry("600x400")

##Label Widget using pack() method
# label=Label(win, text="Hello Programmer", font=("Arial", 10), fg="Black",bg="#34abeb",)
# label.pack()


##Label widget using grid() method
# label=Label(win, text="Hello Programmer", font=("Arial", 10),
#              fg="Black",bg="#34abeb")
# label.grid(row=2,column=2,padx=20,pady=20)

# label=Label(win, text="I am Josh", font=("Arial", 10),
#              fg="Black",bg="#34abeb")
# label.grid(row=5,column=5,padx=20,pady=20)


##Label using place() method
# label=Label(win, text="Hello Programmer", font=("Arial", 10),
#              fg="Black",bg="#34abeb")
# label.place(x=300,y=200)


# #Creating a button
# button=Button(win, text="Click me!", font=("Arial",10), bg="white")
# button.pack(pady=20)


#Using an Image for .png
# image=PhotoImage(file="Semester_2\Week_3\Friar_Tuck2.png")
# #image.subsample(1,1)

# image_label=Label(win,image=image)
# image_label.pack(pady=(50,20))

# # Using an image with .jpeg etc
img=Image.open("Semester_2\Week_3\Friar_Tuck2.png")
img=ImageTk.PhotoImage(img)

label=Label(win, image=img)
label.pack(padx=10,pady=10)


win.mainloop()