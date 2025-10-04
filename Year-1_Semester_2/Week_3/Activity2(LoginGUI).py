from tkinter import *

window=Tk()

window.geometry("400x500")
#logo
image=PhotoImage(file="Semester_2\Week_3\Friar_Tuck2.png")
#image.subsample(1,1)

image_label=Label(window,image=image)
image_label.pack(pady=(50,20))

#Input fields#
Fields=["First Name:", "Last Name:", "Username:", "Email ID:","Password:" ]
for i in range(len(Fields)-1):
    firstname_lb=Label(window,text=f"{Fields[i]}")
    firstname_lb.pack(padx=10,pady=10)

    firstname_input=Entry(window,width=30)
    firstname_input.pack()


#Submit Button


submit_button=Button(window, text="Submit Information", font=("Arial",20),bg="White")
submit_button.pack(pady=(10,20))

window.mainloop()