##CHAT GPT
import tkinter as tk
from tkinter import messagebox
import json

class ToDoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Accessible To-Do List")
        self.root.geometry("500x500")
        self.root.configure(bg="#f0f0f5")
        
        self.tasks = []
        self.filtered_tasks = []
        self.filter_status = "All"
        
        # Custom fonts and styles
        self.default_font = ("Helvetica", 12)
        self.header_font = ("Helvetica", 14, "bold")
        
        # Header
        header = tk.Label(root, text="To-Do List", font=self.header_font, bg="#f0f0f5", fg="#333")
        header.pack(pady=10)
        
        # Entry and Priority frame
        self.entry_frame = tk.Frame(root, bg="#f0f0f5")
        self.entry_frame.pack(pady=10)

        # Entry for adding tasks
        self.entry_task = tk.Entry(self.entry_frame, font=self.default_font, width=30)
        self.entry_task.grid(row=0, column=0, padx=5)
        self.entry_task.bind("<Return>", lambda event: self.add_task())
        
        # Priority dropdown menu
        self.priority_var = tk.StringVar(value="Low")
        self.dropdown_priority = tk.OptionMenu(self.entry_frame, self.priority_var, "High", "Medium", "Low")
        self.dropdown_priority.config(font=self.default_font)
        self.dropdown_priority.grid(row=0, column=1, padx=5)
        
        # Task controls frame
        self.controls_frame = tk.Frame(root, bg="#f0f0f5")
        self.controls_frame.pack(pady=10)

        self.button_add = tk.Button(self.controls_frame, text="Add Task", command=self.add_task, font=self.default_font, bg="#4CAF50", fg="white")
        self.button_add.grid(row=0, column=0, padx=5)
        
        self.button_delete = tk.Button(self.controls_frame, text="Delete Task", command=self.delete_task, font=self.default_font, bg="#f44336", fg="white")
        self.button_delete.grid(row=0, column=1, padx=5)
        
        self.button_mark_complete = tk.Button(self.controls_frame, text="Mark as Complete", command=self.mark_complete, font=self.default_font, bg="#2196F3", fg="white")
        self.button_mark_complete.grid(row=0, column=2, padx=5)
        
        # Filter and search frame
        self.search_filter_frame = tk.Frame(root, bg="#f0f0f5")
        self.search_filter_frame.pack(pady=10)

        # Filter dropdown menu
        self.filter_var = tk.StringVar(value="All")
        self.dropdown_filter = tk.OptionMenu(self.search_filter_frame, self.filter_var, "All", "Completed", "Pending", command=self.filter_tasks)
        self.dropdown_filter.config(font=self.default_font)
        self.dropdown_filter.grid(row=0, column=0, padx=5)

        # Search bar
        self.entry_search = tk.Entry(self.search_filter_frame, font=self.default_font, width=20)
        self.entry_search.grid(row=0, column=1, padx=5)
        self.button_search = tk.Button(self.search_filter_frame, text="Search", command=self.search_task, font=self.default_font, bg="#FFD700", fg="black")
        self.button_search.grid(row=0, column=2, padx=5)
        
        # Task Listbox
        self.frame_tasks = tk.Frame(root, bg="#f0f0f5")
        self.frame_tasks.pack(fill=tk.BOTH, expand=True)
        
        self.listbox_tasks = tk.Listbox(self.frame_tasks, selectmode=tk.SINGLE, font=self.default_font, width=50, height=15, bg="white", fg="#333", activestyle="dotbox")
        self.listbox_tasks.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
        self.listbox_tasks.bind("<Delete>", lambda event: self.delete_task())
        
        self.scrollbar_tasks = tk.Scrollbar(self.frame_tasks, command=self.listbox_tasks.yview)
        self.scrollbar_tasks.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_tasks.config(yscrollcommand=self.scrollbar_tasks.set)
        
    def add_task(self):
        task = self.entry_task.get()
        priority = self.priority_var.get()
        if task:
            self.tasks.append({"task": task, "priority": priority, "completed": False})
            self.update_task_list()
            self.entry_task.delete(0, tk.END)
            messagebox.showinfo("Task Added", "Your task has been added.")
        else:
            messagebox.showwarning("Input Error", "Please enter a task.")

    def delete_task(self):
        try:
            task_index = self.listbox_tasks.curselection()[0]
            del self.tasks[task_index]
            self.update_task_list()
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select a task to delete.")

    def mark_complete(self):
        try:
            task_index = self.listbox_tasks.curselection()[0]
            self.tasks[task_index]["completed"] = True
            self.update_task_list()
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select a task to mark as complete.")

    def filter_tasks(self, filter_type="All"):
        self.filter_status = self.filter_var.get()
        self.update_task_list()

    def search_task(self):
        keyword = self.entry_search.get()
        self.filtered_tasks = [task for task in self.tasks if keyword.lower() in task["task"].lower()]
        self.update_task_list(filtered=True)

    def update_task_list(self, filtered=False):
        self.listbox_tasks.delete(0, tk.END)
        task_list = self.filtered_tasks if filtered else self.tasks

        # Apply completion filter
        if self.filter_status == "Completed":
            task_list = [task for task in task_list if task["completed"]]
        elif self.filter_status == "Pending":
            task_list = [task for task in task_list if not task["completed"]]
        
        # Display tasks with priority and completion status
        for task in task_list:
            status = "(Done)" if task["completed"] else "(Pending)"
            display_text = f"{task['task']} - {task['priority']} {status}"
            self.listbox_tasks.insert(tk.END, display_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ToDoApp(root)
    root.mainloop()















##CLAUDE

# import tkinter as tk
# from tkinter import ttk, messagebox
# from datetime import datetime
# import json
# import os

# class DateEntry(ttk.Frame):
#     def __init__(self, parent):
#         super().__init__(parent)
        
#         # Create spinboxes for day, month, and year
#         self.day_var = tk.StringVar(value="1")
#         self.month_var = tk.StringVar(value="1")
#         self.year_var = tk.StringVar(value=str(datetime.now().year))
        
#         # Month spinbox
#         self.month_spin = ttk.Spinbox(self, from_=1, to=12, width=2,
#                                     textvariable=self.month_var)
#         self.month_spin.pack(side=tk.LEFT, padx=2)
        
#         # Day spinbox
#         self.day_spin = ttk.Spinbox(self, from_=1, to=31, width=2,
#                                   textvariable=self.day_var)
#         self.day_spin.pack(side=tk.LEFT, padx=2)
        
#         # Year spinbox
#         self.year_spin = ttk.Spinbox(self, from_=2024, to=2100, width=4,
#                                    textvariable=self.year_var)
#         self.year_spin.pack(side=tk.LEFT, padx=2)

#     def get_date(self):
#         try:
#             return datetime(
#                 int(self.year_var.get()),
#                 int(self.month_var.get()),
#                 int(self.day_var.get())
#             )
#         except ValueError:
#             return datetime.now()

# class TodoApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Todo List Manager")
#         self.root.geometry("800x600")
#         self.root.configure(bg="#f0f0f0")

#         # Data storage
#         self.filename = "tasks.json"
#         self.tasks = []
#         self.load_tasks()

#         # Styling
#         self.style = ttk.Style()
#         self.style.configure("Custom.TFrame", background="#f0f0f0")
#         self.style.configure("TaskFrame.TFrame", background="white", relief="raised")
#         self.style.configure("Custom.TButton", padding=5, font=("Arial", 10))
#         self.style.configure("Task.TCheckbutton", background="white")

#         # Main layout
#         self.setup_gui()

#     def setup_gui(self):
#         # Top frame for adding tasks
#         top_frame = ttk.Frame(self.root, style="Custom.TFrame", padding="10")
#         top_frame.pack(fill=tk.X, padx=10, pady=5)

#         # Task entry
#         self.task_var = tk.StringVar()
#         task_entry = ttk.Entry(top_frame, textvariable=self.task_var, font=("Arial", 12), width=40)
#         task_entry.pack(side=tk.LEFT, padx=(0, 10))

#         # Due date picker
#         date_frame = ttk.Frame(top_frame)
#         date_frame.pack(side=tk.LEFT, padx=5)
#         ttk.Label(date_frame, text="Due Date:", background="#f0f0f0").pack(side=tk.LEFT)
#         self.due_date = DateEntry(date_frame)
#         self.due_date.pack(side=tk.LEFT, padx=5)

#         # Priority selector
#         self.priority_var = tk.StringVar(value="Medium")
#         priority_frame = ttk.Frame(top_frame, style="Custom.TFrame")
#         priority_frame.pack(side=tk.LEFT, padx=10)
        
#         ttk.Label(priority_frame, text="Priority:", background="#f0f0f0").pack(side=tk.LEFT)
#         priorities = ["High", "Medium", "Low"]
#         priority_menu = ttk.OptionMenu(priority_frame, self.priority_var, "Medium", *priorities)
#         priority_menu.pack(side=tk.LEFT, padx=5)

#         # Add button
#         add_button = ttk.Button(top_frame, text="Add Task", command=self.add_task,
#                               style="Custom.TButton")
#         add_button.pack(side=tk.LEFT, padx=10)

#         # Filter frame
#         filter_frame = ttk.Frame(self.root, style="Custom.TFrame", padding="5")
#         filter_frame.pack(fill=tk.X, padx=10)

#         self.show_completed_var = tk.BooleanVar(value=True)
#         show_completed_check = ttk.Checkbutton(filter_frame, text="Show Completed",
#                                              variable=self.show_completed_var,
#                                              command=self.refresh_tasks)
#         show_completed_check.pack(side=tk.LEFT)

#         # Tasks canvas with scrollbar
#         canvas_frame = ttk.Frame(self.root, style="Custom.TFrame")
#         canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

#         self.canvas = tk.Canvas(canvas_frame, bg="#f0f0f0")
#         scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
#         self.scrollable_frame = ttk.Frame(self.canvas, style="Custom.TFrame")

#         self.scrollable_frame.bind(
#             "<Configure>",
#             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
#         )

#         self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
#         self.canvas.configure(yscrollcommand=scrollbar.set)

#         scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
#         self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#         # Bind mouse wheel
#         self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

#         # Load initial tasks
#         self.refresh_tasks()

#     def _on_mousewheel(self, event):
#         self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

#     def load_tasks(self):
#         if os.path.exists(self.filename):
#             try:
#                 with open(self.filename, 'r') as file:
#                     self.tasks = json.load(file)
#             except:
#                 self.tasks = []
#         else:
#             self.tasks = []

#     def save_tasks(self):
#         with open(self.filename, 'w') as file:
#             json.dump(self.tasks, file)

#     def add_task(self):
#         title = self.task_var.get().strip()
#         if not title:
#             messagebox.showwarning("Warning", "Please enter a task title!")
#             return

#         task = {
#             'id': len(self.tasks) + 1,
#             'title': title,
#             'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'due_date': self.due_date.get_date().strftime('%Y-%m-%d'),
#             'priority': self.priority_var.get(),
#             'completed': False
#         }

#         self.tasks.append(task)
#         self.save_tasks()
#         self.task_var.set("")  # Clear entry
#         self.refresh_tasks()

#     def toggle_task(self, task_id):
#         for task in self.tasks:
#             if task['id'] == task_id:
#                 task['completed'] = not task['completed']
#                 self.save_tasks()
#                 self.refresh_tasks()
#                 break

#     def delete_task(self, task_id):
#         self.tasks = [task for task in self.tasks if task['id'] != task_id]
#         self.save_tasks()
#         self.refresh_tasks()

#     def get_priority_color(self, priority):
#         return {
#             "High": "#ff7f7f",
#             "Medium": "#ffb366",
#             "Low": "#90EE90"
#         }.get(priority, "#ffffff")

#     def refresh_tasks(self):
#         # Clear existing tasks
#         for widget in self.scrollable_frame.winfo_children():
#             widget.destroy()

#         # Filter tasks
#         visible_tasks = [t for t in self.tasks 
#                         if self.show_completed_var.get() or not t['completed']]

#         # Sort tasks by priority and due date
#         priority_order = {"High": 0, "Medium": 1, "Low": 2}
#         visible_tasks.sort(key=lambda x: (x['completed'], 
#                                         priority_order[x['priority']], 
#                                         x['due_date']))

#         # Create task widgets
#         for task in visible_tasks:
#             task_frame = ttk.Frame(self.scrollable_frame, style="TaskFrame.TFrame")
#             task_frame.pack(fill=tk.X, padx=5, pady=2)

#             # Configure grid columns
#             task_frame.grid_columnconfigure(1, weight=1)

#             # Checkbox
#             completed_var = tk.BooleanVar(value=task['completed'])
#             check = ttk.Checkbutton(task_frame, variable=completed_var, 
#                                   command=lambda t=task: self.toggle_task(t['id']),
#                                   style="Task.TCheckbutton")
#             check.grid(row=0, column=0, padx=5, pady=5)

#             # Priority indicator
#             priority_label = tk.Label(task_frame, text="", width=2, 
#                                     bg=self.get_priority_color(task['priority']))
#             priority_label.grid(row=0, column=1, sticky="ns")

#             # Task title
#             title_label = ttk.Label(task_frame, text=task['title'],
#                                   font=("Arial", 10, "overstrike" if task['completed'] else ""))
#             title_label.grid(row=0, column=2, padx=10, sticky="w")

#             # Due date
#             due_label = ttk.Label(task_frame, text=f"Due: {task['due_date']}")
#             due_label.grid(row=0, column=3, padx=5)

#             # Delete button
#             delete_btn = ttk.Button(task_frame, text="×",
#                                   command=lambda t=task: self.delete_task(t['id']),
#                                   style="Custom.TButton")
#             delete_btn.grid(row=0, column=4, padx=5)

# def main():
#     root = tk.Tk()
#     app = TodoApp(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()






##GEMINI

# import tkinter as tk

# def add_task():
#     task = task_entry.get()
#     if task:
#         tasks_list.insert(tk.END, task)
#         task_entry.delete(0, tk.END)

# def delete_task():
#     selected_task = tasks_list.curselection()
#     if selected_task:
#         tasks_list.delete(selected_task[0])

# root = tk.Tk()
# root.title("To-Do List")

# task_label = tk.Label(root, text="Enter Task:")
# task_label.pack()

# task_entry = tk.Entry(root)
# task_entry.pack()

# add_button = tk.Button(root, text="Add Task", command=add_task)
# add_button.pack()

# delete_button = tk.Button(root, text="Delete Task", command=delete_task)
# delete_button.pack()

# tasks_list = tk.Listbox(root)
# tasks_list.pack()

# root.mainloop()