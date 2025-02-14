## Reading a File
# filename="Week_9\grades (1).csv"
# in_file=open(filename,"r")
# for line in in_file:
#     pass
# in_file.close()

## Writing to a file
# out_file=open("Week_9\Class_grades.txt","w")
# for student in students:
#     out_file.write("")
# out_file.close()

##Linked Lists
# A list containing another list
# Example:
#   split_line=line.split(",")
#   students.append([split_line[0],int(split_line[1])])


#Rounding and displaying it to 2dp
#student_range=[range,round((range/75)*100,2)]
# can also use:
#   :.2f
#   (average_mark/75)*100:.2f

## sum
#   gets the sum of all things

## len
#   returns the length of something

## min/max
# gets the minimum/maximum value

##Finding the Lowest/Highest mark in a list
#lowest_mark=min(student[1] for student in students)

##Linked lists
# students=[]
# student=["Name", 19, "SoftwareEng", 90]
# students.append(student)

# Specifier	Description	                            Example (num = 123.456)	    Output
# :.2f	    Fixed-point notation (2 decimal places)	{:.2f}	                    123.46
# :.0f	    Round to whole number	                {:.0f}	                    123
# :e	    Scientific notation	                    {:e}	                    1.234560e+02
# :.2e	    Sci. notation (2 decimals)	            {:.2e}	                    1.23e+02
# :g	    General (switches between f & e)	    {:g}	                    123.456
# :.2g	    General (2 significant figures)	        {:.2g}	                    1.2e+02
# :%	    Percentage (multiplies by 100)	        {:.1%}	                    12345.6%
# :x	    Hexadecimal (lowercase)	                {:x} (num=255)	            ff
# :X	    Hexadecimal (uppercase)	                {:X} (num=255)	            FF
# :b	    Binary representation	                {:b} (num=255)	            11111111
# :o	    Octal representation	                {:o} (num=255)	            377


# Specifier	    Description	                Example (word = "Hi")	    Output
# :>10	        Right-align (width 10)	    {:>10}	                    " Hi"
# :<10	        Left-align (width 10)	    {: <10}	                    "Hi "
# :^10	        Center-align (width 10)	    {:^10}	                    " Hi "
# :_<10	        Left-align, fill with _	    {:_<10}	                    "Hi________"
# :*>10	        Right-align, fill with *	{:*>10}	                    "********Hi"
# :*^10	        Center-align, fill with *	{:*^10}	                    "****Hi****"


# Specifier	        Description	                    Example (num = 42)	        Output
# :d	            Integer (default)	            {:d}	                    42
# :04d	            Zero-padding (width 4)	        {:04d}	                    0042
# :>4d	            Right-align (width 4)	        {:>4d}	                    " 42"
# :<4d	            Left-align (width 4)	        {: <4d}	                    "42 "
# :+d	            Show sign for positive numbers	{:+d}	                    +42
# : d	            Space before positive numbers	{: d}	"                   42"
# :,d	            Thousands separator	            "{:,d}".format(1000000)	    1,000,000

# Specifier	        Description	                    Example	                    Output
# :b	            Convert True/False to 1/0	    "{:b}".format(True)	        1
# :s	            Convert to string	            "{:s}".format("hello")	    hello
# !s	            Force string representation	    "{!s}".format(None)	        None
# !r	            Force representation (repr())	"{!r}".format(None)	        'None'