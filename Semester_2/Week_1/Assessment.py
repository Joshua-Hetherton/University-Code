#the globally available list of books
books=[]

#Diplays all options for the program
def main_menu():

    loop= True
    while loop is True:

        user_input=input("""
Please enter what action you would like to do (Use numbers 1-6):
1. Add a book title to the list
2. Print all current books available
3. Find a book
4. Bulk Upload of Book Titles
5. Download Book Titles
6. Quit
""")
        
#Switch case to allow the correct function to run
        match(user_input.lower()):
            case "1":
                add_book()

            case "2":
                print_books()

            case "3":
                find_book()

            case "4":
                write_to_file()

            case "5":
                read_from_file()

            case "6":
                print("Ending Program")
                loop=False
            #in case users typed Quit/quit instead of "6"
            case "quit":
                print("Ending Program")
                loop=False


#Adds a book to the List, 1 at a time
def add_book():

    loop= True
    while loop is True:

        user_input=input("Enter a Book Title(1 at a time) or type quit: ")

        if(user_input.lower() == "quit"):
            loop=False

        else:
            books.append(user_input)

        
#Prints all books in the list
def print_books():

    print("All Books currently in the list: ")
    index_of_book= 0
    for book in books:

        print(f"Index {index_of_book}: {book} ")
        index_of_book+= 1


#finds a book from an index, loops if an invalid number is given
def find_book():

    loop= True
    while loop is True:

        user_input=int(input("Please Enter an Index number: "))

        #checking to see if the number given exceeeds the index of books
        if(user_input>len(books)-1):
            print("Invalid Number, please try again")

        else:
            loop=False
            print(f"Book found\nIndex {user_input}: {books[user_input]}")

#Requirement 4(Upload Program to .txt)
#Write to the txt
def write_to_file():
    
    out_file= open("MyBooks.txt", "w")
    for book in books:
        out_file.write(f"{book}\n")
    out_file.close()

    print("Writing to the file is complete")

#Requirement 5(Download from .csv to Program)
#Reading from the .csv
def read_from_file():

    filename= "Books.csv"
    in_file=open(filename,"r")

    print("All Books from the file that have been uploaded to the List of Books available")

    index_of_books=0
    for line in in_file:
        #using .replace to get rid of the extra spaces between the lines
        
        print(f"Index {index_of_books}: {line.replace("\n", " ")}")
        books.append(line.replace("\n", " "))
        index_of_books+=1
    in_file.close()

main_menu()