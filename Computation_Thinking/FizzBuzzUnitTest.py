# fizzbuzz.py

def isDivisible(dividend: int, divisor: int) -> bool:
    return (dividend % divisor) == 0


def fizzbuzz(number: int) -> str:
    if not isinstance(number, int):
        raise ValueError('The input is not a number: ' + number)

    divByThree = isDivisible(number, 3)
    divByFive = isDivisible(number, 5)

    if divByThree and divByFive:
        return "FizzBuzz"
    elif divByThree:
        return "Fizz"
    elif divByFive:
        return "Buzz"
    else:
        return str(number)


if __name__ == '__main__':
    number = input("Please enter a number: ")
    try:
        print(fizzbuzz(int(number)))
    except ValueError as e:
        print(e)