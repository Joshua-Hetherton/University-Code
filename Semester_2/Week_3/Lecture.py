from tkinter import *


win=Tk()
#Add widgets here

win.title("Bullshit")
# win.config(bg="") sets bg colour
win.config(bg="#34abeb")
win.geometry("600x400")

#Displays a PNG of Friar Tuck
image=PhotoImage(file="Semester_2\Week_3\Friar_Tuck2.png")

image_label=Label(win,image=image)
image_label.pack()

win.mainloop()