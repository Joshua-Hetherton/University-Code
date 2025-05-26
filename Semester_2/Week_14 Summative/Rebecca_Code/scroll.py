from tkinter import *

window = Tk()  
window.title("University of Winchester")
window.geometry("700x600")
bg_colour = "#6b2b63"

# Create a frame called 'view_students' inside the main window
view_students = Frame(window, bg=bg_colour, width=700, height=600)
view_students.place(x=0, y=0)  # Put the frame in the window at position (0,0)

# Create a Text widget inside the frame. This is where you can show student data.
text_widget = Text(view_students, wrap=WORD, width=60, height=20)
text_widget.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)  
# side=LEFT makes the Text widget appear on the left side of the frame
# fill=BOTH and expand=True allow the Text box to grow and fill the space
# padx and pady add some padding (space) around the widget

# Create a vertical scrollbar that will control the Text widget
scrollbar = Scrollbar(view_students, command=text_widget.yview)
scrollbar.pack(side=RIGHT, fill=Y, pady=10)  
# side=RIGHT puts the scrollbar on the right edge of the frame
# fill=Y makes the scrollbar stretch vertically

# Link the scrollbar to the Text widget
text_widget.config(yscrollcommand=scrollbar.set)

# Insert example lines into the Text box so you can scroll through them
for i in range(50):
    text_widget.insert(END, f"Student {i+1}: Example student data here...\n")

window.mainloop()
