# tuple= (1,2,3)*5
# print(tuple)

#Example showing Tuple MemberShip
# reg_users= ("Bob","Bill","Jeb","Valentina","Josh")
# username="Josh"
# if username in reg_users:
#     print("Access Granted")
# else:
#     print("Access Denied")

#Len()
# my_tuple= (1,2,3,4,5)
# print(len(my_tuple))

###Nested Tuple###
# movies=(
#     ("Interstellar","Christopher Nolan","Sci-Fi", 2014),
#     ("Oppenhiemer","Christopher Nolan", "Historical",2023),
# )
# print(movies[1][1])
# #Accessing specific bits of each part of the nested tuples
# print("All release dates are", movies[0][3],movies[1][3])

# #Dictionaries
# uni={
#     "course":"Programming In python",
#     "Lecturer":" Dr Neha",
#     "level": "BSC First Year"
# }
# print(uni)

# #Method 2
# #creating a dictionary in python using dict
# person=dict(name="Bob", age=22,grade="A")
# print(person)

## Nested Dictionary

students={
    "student1": {"name":"Bob", "Age":32},
    "student2": {"name":"Bill", "Age":600}
}
#Prints the current dict
print(students)

#Prints a specific value in the dict
print(students["student1"]["name"])

#.keys()
#prints all keys
print(students.keys())

#.values()
#prints all values
print(students.values())

