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
##error is with product=0, and the loop in the range starting at 0
##make product=1
##make product *=i+1 (in the loop)
def multiply_to_number(stop_number):
	product = 1

	for i in range(stop_number):
		product *= i+1

	return product

def main():
	number_entered = get_number_input()
	product_of_numbers = multiply_to_number(number_entered)
	print("The product of the first {0} numbers is: {1}".format(number_entered, product_of_numbers))

main()