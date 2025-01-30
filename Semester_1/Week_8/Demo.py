# sep=""
# var_1=5
# var_2=4

# sum=var_1+var_2

# print(f"The sum of", var_1, "and", var_2, "is", sum,sep="")



#end=""
# var_1=5
# var_2=4

# sum=var_1+var_2

# print(f"The sum of", var_1, "and", var_2, "is", sum,end="")
# print("Hello")



# def Print_Square():
#     for i in range(5):

#         for t in range(5):
#             print(i+t, end=" " )
#         print()
            

# Print_Square()

#format
var_1=5.12
var_2=4

sum=var_1+var_2
#one way is:
# print("The sum of {} and {} is {}".format(var_1,var_2,sum))

#alternatively, you can use to specify what variable goes where:
print("The sum of {1:.2f} and {0:.2f} is {2:.2f}".format(var_1,var_2,sum))

##you can also "pad" the output:
print("The sum of {1:.2f} and {0:.2f} is {2:10.2f}".format(var_1,var_2,sum))

#you can pad with 0s:
print("The sum of {1:.2f} and {0:.2f} is {2:010.2f}".format(var_1,var_2,sum))

#my preffered way:
# print(f"The sum of {var_1:.2f} and {var_2:.2f} is {sum:.2f}")

