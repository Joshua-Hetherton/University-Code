from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
win = Tk()

def show_frame(frame):
    frame.tkraise()

main_menu_frame = Frame(win, bg = "#6B275C", width =600, height = 500)
main_menu_frame.pack()

win.geometry("600x500") # Set window size,
win.configure(bg = "#6B275C"),
win.title("Home Menu"),
button_studentreg = Button(main_menu_frame, text = "Student Registration", bg = "white", fg = "black")
button_studentreg.place(x = 230, y = 200)

button_studentlog = Button(main_menu_frame, text = "Student Login", bg = "white", fg = "black")
button_studentlog.place(x = 245, y = 250)

button_stafflog = Button(main_menu_frame, text = "Staff Login", bg = "white", fg = "black")
button_stafflog.place(x = 252, y = 300)

# img_Winchester_logo = Image.open("Assignment\Winchester_uni_header.png")
# img_Winchester_logo = ImageTk.PhotoImage(img_Winchester_logo)

# label = Label(win, image= img_Winchester_logo)
# label.pack()

show_frame(main_menu_frame)
win.mainloop()