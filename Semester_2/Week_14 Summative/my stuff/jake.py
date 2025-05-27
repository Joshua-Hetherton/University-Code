def read_student_data():
    read_file = open('Assignment\student_details.txt', 'r')
    for line in read_file:
        seperated_lines = line.strip().split(',')
        current_name = seperated_lines[0]
        current_email = seperated_lines[1]
        current_password = seperated_lines[2]
        print(f"Name: {current_name}, Email: {current_email}, Password: {current_password}")

    if current_email == email_student_login.get() and current_password == password_student_login.get():
        show_frame(student_data_frame)
    else:
        messagebox.showerror("Error", "Incorrect Credentials")