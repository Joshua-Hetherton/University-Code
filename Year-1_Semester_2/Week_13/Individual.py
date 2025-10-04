from tkinter import *
import sqlite3

#Creating a Database
# conn=sqlite3.connect("uni_library.db")
# c=conn.cursor()

# #Creating a Table
# c.execute("""CREATE TABLE IF NOT EXISTS students(
#           book_id integer primary key
#           title text,
#           author text
#           ) """)


window=Tk()
window.geometry("400x600")

def show_frame(frame):
        frame.tkraise()


#Creating the frames such as main men u, adding books and looking up books
main_menu=Frame(window, bg="Light Blue",width=450,height=600)
add_book=Frame(window, bg="Purple",width=450,height=600)
find_book=Frame(window,bg="Blue",width=450,height=600)

for frame in (main_menu,add_book,find_book):
    frame.place(x=0,y=0)


#Main Menu
Button(main_menu, text="Find Information", font=("Arial",10),bg="White", command=lambda : show_frame(find_book)).place(x=135,y=340)
Button(main_menu, text="Add Book", font=("Arial",10),bg="White", command=lambda : show_frame(add_book)).place(x=135,y=370)

#########
#Find Info


## Find Info -> Main Menu
Button(find_book, text="Back to Main Menu", font=("Arial",10),bg="White", command=lambda : show_frame(main_menu)).place(x=135,y=540)

#########
#Add Book

Label(add_book, text="Book Name: ",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=200)
name_entry=Entry(add_book,width=30)
name_entry.place(x=180, y=203)

Label(add_book, text="Author: ",bg="Light Blue",font=("Arial",10),fg="Black").place(x=50,y=200)
author_entry=Entry(add_book,width=30)
author_entry.place(x=180, y=203)

def submit():
    getname=name_entry.get()
    getauthor=author_entry.get()



    # c.execute("""INSERT INTO students(title,author,book_id) VALUES(?, ?, ?) """,
    #           (getname,getauthor)
    #           )
    # conn.commit()

##Add Book -> Main Menu
Button(add_book, text="Back to Main Menu", font=("Arial",10),bg="White", command=lambda : show_frame(main_menu)).place(x=135,y=540)


##Displaying tkinter
show_frame(main_menu)
window.mainloop()