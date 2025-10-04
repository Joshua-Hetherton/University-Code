##Task 1
#  grades=(100,90,40,25.9,76)
# details=("Bob","Bobsgmail@gmail.com","123 Bob Avenue")
# details_and_grades=grades+details
# print(details_and_grades)

##Task 2
# dailymenu=("Fish","Steak","Chicken")*3
# print(dailymenu)

##Task 3
# shopping_cart=("Book 1","Book 2","Book 3", "Book 'How to tell you've got too many books'")
# if "Book 2" in shopping_cart:
#     print("Your Wishlisted item has been added")
# else:
#     print("You need to add your wishlisted book thats on sale and comes with a free iphone!")



#Activity 3

# Books_Read=(("Skyward Flight", "Brandon Sanderson", 12.99),
#             ("Stormlight Archive","Brandon Sanderson", 13.99),
#             ("Skyllduggery pleasent", "Derek Landy",12.99),
#             ("Divergent","Unknown",21.99),
#             ("Resurgent","Unkown2",17.99),
#             ("Darkest Minds","Cant Remember",19.99),
#             ("Seven Deaths of an Empire", "Who knows 3",18.99),
#             ("Inheritance","Bob Maker 2", 18.99),
#             ("Eragon","Bob Maker", 1000.99),
#             ("Court of Thorns and Roses", "Sarah J Mass",19.99),
#             ("Throne of Glass", "Sarah J Mass",19.99))

# print(f"The author of the second book is {Books_Read[1][1]} ")
# print(f"The author of the ninth book is {Books_Read[9][1]} ")


shopping_list=("Steak","Chicken","Brocolli","WensleyDale","Cheddar", "Smoke Cheddar", "Bread", "Peppers", "Milk", "Carrots")
print(shopping_list)
convert_to_list=list(shopping_list)
convert_to_list.append("Orange")
convert_to_list.append("Apple")
convert_to_list.remove("Steak")
back_to_tuple=tuple(convert_to_list)
print(back_to_tuple)


   




