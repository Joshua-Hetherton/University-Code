def get_number_input():
	valid_entry = False

	while(not valid_entry):
		input_number = input("Please enter a number: ")
		if(input_number.isdigit()):
			input_number = int(input_number)
			valid_entry = True
		else:
			print("Sorry that is not a valid number - please try again")
	
	return input_number
##Error was with the loop finishing at 4, not 5
def sum_to_number(stop_number):
	sum_nums = 0

	for i in range(stop_number+1):
		print(f"Before:{sum_nums}, {i}")
		sum_nums += i
		print(f"After:{sum_nums}, {i}\n")

	return sum_nums

def main():
	number_entered = get_number_input()
	sum_of_numbers = sum_to_number(number_entered)
	print("The sum of the first {0} numbers is: {1}".format(number_entered, sum_of_numbers))

main()
