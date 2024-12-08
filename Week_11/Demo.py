try:
    loan_amount=int(input("Please enter loan amount: "))
    rate=7
    years=5
    interest=loan_amount* (rate/100)
    repayment=(loan_amount+interest)/years/12
    print(f"Your repayment on the loan {loan_amount:,} at a rate of {rate}% over {years} is: {repayment:0.2f}")

except ZeroDivisionError:
    print("Sorry: cant divide by zero!")
except ValueError:
    print("Sorry: Isn't a number")