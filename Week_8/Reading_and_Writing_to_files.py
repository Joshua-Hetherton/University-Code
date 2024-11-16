filename="Week_8\TPS_Spend_Oct_2018.csv"
#panda allows for managable access for large amounts of data, and can be used with numpy

#Opening the file
#Define the mode its to open in (r(read_only),w(write_only),a(append))
#You cant insert things into the middle of a file, you can only append to the end
in_file=open(filename,"r")

daily_spend=[]

for line in in_file:
    #Splitting to get the amount spent
    split_line=line.split(",")
    #adding it to the list
    daily_spend.append(float(split_line[1]))

in_file.close()

sum_daily_spend=0
for spend in daily_spend:
    sum_daily_spend+=spend

print(f"Total is {sum_daily_spend:,.2f}")

#appending to the file
data_to_add="\n31/10/2018,23099083.57\n"

out_file=open(filename,"a")

#appending to the file
out_file.write(data_to_add)

#able to append from a print statement
# print(f"new data {data_to_add}",file=out_file)

out_file.close()

