#CLAUD's CODE

import tkinter as tk
from tkinter import ttk
import math
import datetime
from typing import List, Dict, Union

class CalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scientific Calculator")
        self.root.configure(bg='#2c3e50')  # Dark blue background
        
        # Calculator state
        self.memory: float = 0
        self.history: List[Dict[str, Union[str, float, datetime.datetime]]] = []
        self.current_number = ""
        self.first_number = None
        self.operation = None
        self.should_clear = False

        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TButton', 
                           padding=10, 
                           font=('Arial', 12),
                           background='#34495e')
        
        self.create_widgets()

    def create_widgets(self):
        # Display frame
        display_frame = tk.Frame(self.root, bg='#2c3e50')
        display_frame.pack(padx=10, pady=5, fill=tk.X)

        # Main display
        self.display_var = tk.StringVar(value="0")
        self.display = tk.Entry(display_frame,
                              textvariable=self.display_var,
                              font=('Arial', 24),
                              bd=10,
                              relief=tk.FLAT,
                              justify=tk.RIGHT,
                              bg='#ecf0f1')
        self.display.pack(fill=tk.X, padx=5, pady=5)

        # Memory display
        self.memory_var = tk.StringVar(value="Memory: 0")
        self.memory_label = tk.Label(display_frame,
                                   textvariable=self.memory_var,
                                   font=('Arial', 10),
                                   bg='#2c3e50',
                                   fg='white')
        self.memory_label.pack(anchor='e', padx=5)

        # Buttons frame
        buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        buttons_frame.pack(padx=10, pady=5)

        # Button layout
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3), ('C', 1, 4),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3), ('√', 2, 4),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3), ('^', 3, 4),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3), ('log', 4, 4)
        ]

        # Memory buttons
        memory_buttons = [
            ('MC', 0, 0), ('MR', 0, 1), ('M+', 0, 2), ('M-', 0, 3), ('MS', 0, 4)
        ]

        # Scientific buttons
        scientific_buttons = [
            ('sin', 5, 0), ('cos', 5, 1), ('tan', 5, 2), ('π', 5, 3), ('e', 5, 4)
        ]

        # Create all buttons with consistent styling
        all_buttons = buttons + memory_buttons + scientific_buttons
        for (text, row, col) in all_buttons:
            btn = tk.Button(buttons_frame,
                          text=text,
                          font=('Arial', 12),
                          width=5,
                          height=2,
                          bg='#34495e',
                          fg='white',
                          activebackground='#2980b9',
                          relief=tk.FLAT)
            btn.grid(row=row, column=col, padx=2, pady=2)
            
            # Bind button clicks
            if text in '0123456789.':
                btn.configure(command=lambda t=text: self.add_digit(t))
            elif text in '+-*/^':
                btn.configure(command=lambda t=text: self.set_operation(t))
            elif text == '=':
                btn.configure(command=self.calculate)
            elif text == 'C':
                btn.configure(command=self.clear)
            elif text in ['MC', 'MR', 'M+', 'M-', 'MS']:
                btn.configure(command=lambda t=text: self.handle_memory(t))
            elif text in ['sin', 'cos', 'tan', 'log', '√']:
                btn.configure(command=lambda t=text: self.handle_scientific(t))
            elif text == 'π':
                btn.configure(command=lambda: self.add_constant(math.pi))
            elif text == 'e':
                btn.configure(command=lambda: self.add_constant(math.e))

        # History frame
        history_frame = tk.Frame(self.root, bg='#2c3e50')
        history_frame.pack(padx=10, pady=5, fill=tk.X)

        # History display
        self.history_text = tk.Text(history_frame,
                                  height=4,
                                  font=('Arial', 10),
                                  bg='#34495e',
                                  fg='white',
                                  relief=tk.FLAT)
        self.history_text.pack(fill=tk.X)

    def add_digit(self, digit):
        if self.should_clear:
            self.display_var.set("")
            self.should_clear = False
        current = self.display_var.get()
        if current == "0" and digit != ".":
            self.display_var.set(digit)
        else:
            self.display_var.set(current + digit)

    def add_constant(self, value):
        self.display_var.set(str(value))
        self.should_clear = True

    def set_operation(self, op):
        self.first_number = float(self.display_var.get())
        self.operation = op
        self.should_clear = True

    def calculate(self):
        if self.operation and self.first_number is not None:
            second_number = float(self.display_var.get())
            result = None
            
            try:
                if self.operation == '+':
                    result = self.first_number + second_number
                elif self.operation == '-':
                    result = self.first_number - second_number
                elif self.operation == '*':
                    result = self.first_number * second_number
                elif self.operation == '/':
                    if second_number == 0:
                        self.display_var.set("Error")
                        return
                    result = self.first_number / second_number
                elif self.operation == '^':
                    result = math.pow(self.first_number, second_number)

                if result is not None:
                    self.display_var.set(str(result))
                    self.add_to_history(f"{self.first_number} {self.operation} {second_number} = {result}")
                
                self.first_number = None
                self.operation = None
                self.should_clear = True
                
            except Exception as e:
                self.display_var.set("Error")

    def handle_scientific(self, func):
        try:
            number = float(self.display_var.get())
            result = None
            
            if func == 'sin':
                result = math.sin(math.radians(number))
            elif func == 'cos':
                result = math.cos(math.radians(number))
            elif func == 'tan':
                result = math.tan(math.radians(number))
            elif func == 'log':
                if number <= 0:
                    self.display_var.set("Error")
                    return
                result = math.log10(number)
            elif func == '√':
                if number < 0:
                    self.display_var.set("Error")
                    return
                result = math.sqrt(number)
            
            if result is not None:
                self.display_var.set(str(result))
                self.add_to_history(f"{func}({number}) = {result}")
                self.should_clear = True
                
        except Exception as e:
            self.display_var.set("Error")

    def handle_memory(self, operation):
        try:
            if operation == 'MC':
                self.memory = 0
            elif operation == 'MR':
                self.display_var.set(str(self.memory))
                self.should_clear = True
            elif operation == 'M+':
                self.memory += float(self.display_var.get())
            elif operation == 'M-':
                self.memory -= float(self.display_var.get())
            elif operation == 'MS':
                self.memory = float(self.display_var.get())
            
            self.memory_var.set(f"Memory: {self.memory}")
            
        except Exception as e:
            self.display_var.set("Error")

    def clear(self):
        self.display_var.set("0")
        self.first_number = None
        self.operation = None
        self.should_clear = False

    def add_to_history(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.history_text.insert('1.0', f"[{timestamp}] {text}\n")
        self.history_text.see('1.0')

def main():
    root = tk.Tk()
    root.geometry("400x600")
    calc = CalculatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
else:





##

#CHAT GPT CALCULATOR
# import tkinter as tk
# from tkinter import messagebox

# # Initialize main window
# root = tk.Tk()
# root.title("Calculator")
# root.geometry("300x400")
# root.resizable(0, 0)  # Prevent window resizing

# # Entry field to display numbers and results
# entry = tk.Entry(root, width=20, font=('Arial', 18), borderwidth=2, relief="solid")
# entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# # Global variables for the calculator
# current_expression = ""

# # Button functions
# def button_click(value):
#     global current_expression
#     current_expression += str(value)
#     entry.delete(0, tk.END)
#     entry.insert(tk.END, current_expression)

# def clear():
#     global current_expression
#     current_expression = ""
#     entry.delete(0, tk.END)

# def calculate():
#     global current_expression
#     try:
#         result = eval(current_expression)  # Calculate result using eval
#         entry.delete(0, tk.END)
#         entry.insert(tk.END, str(result))
#         current_expression = str(result)  # Update expression with result
#     except Exception:
#         messagebox.showerror("Error", "Invalid Input")
#         clear()

# # Button Layout
# button_texts = [
#     '7', '8', '9', '/',
#     '4', '5', '6', '*',
#     '1', '2', '3', '-',
#     '0', '.', '=', '+'
# ]

# buttons = []
# for i, text in enumerate(button_texts):
#     # Create each button with appropriate label and command
#     button = tk.Button(root, text=text, width=5, height=2, font=('Arial', 14),
#                        command=lambda t=text: button_click(t) if t != '=' else calculate())
#     buttons.append(button)
#     # Place buttons in a grid layout
#     row, col = divmod(i, 4)
#     button.grid(row=row+1, column=col, padx=5, pady=5)

# # Clear button
# clear_button = tk.Button(root, text='C', width=5, height=2, font=('Arial', 14), command=clear)
# clear_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

# # Exit button
# exit_button = tk.Button(root, text='Exit', width=5, height=2, font=('Arial', 14), command=root.quit)
# exit_button.grid(row=5, column=2, columnspan=2, padx=5, pady=5)

# # Start the GUI event loop
# root.mainloop()









#GEMINI Calculator
# def calculator():
#     """A simple calculator with enhanced error handling and user experience."""

#     while True:
#         try:
#             num1 = float(input("Enter the first number: "))
#             operator = input("Enter the operator (+, -, *, /): ")
#             num2 = float(input("Enter the second number: "))

#             if operator not in ['+', '-', '*', '/']:
#                 raise ValueError("Invalid operator")

#             if operator == '/' and num2 == 0:
#                 raise ZeroDivisionError("Division by zero error")

#             result = eval(f"{num1}{operator}{num2}")

#             print(f"Result: {result}")

#         except (ValueError, ZeroDivisionError) as e:
#             print(f"Error: {str(e)}")

#         choice = input("Do you want to continue? (yes/no): ").lower()
#         if choice != 'yes':
#             break

# if __name__ == "__main__":
#     calculator()