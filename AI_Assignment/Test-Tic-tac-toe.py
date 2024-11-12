import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.player = 'X'
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_buttons()
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_game)
        self.reset_button.grid(row=3, column=0, columnspan=3)

    def create_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.root, text="", font=('normal', 40), width=5, height=2,
                                               command=lambda i=i, j=j: self.on_button_click(i, j))
                self.buttons[i][j].grid(row=i, column=j)

    def on_button_click(self, i, j):
        if self.buttons[i][j]['text'] == "" and self.check_winner() is False:
            self.buttons[i][j]['text'] = self.player
            if self.check_winner():
                messagebox.showinfo("Tic-Tac-Toe", f"Player {self.player} wins!")
            elif self.is_draw():
                messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            else:
                self.player = 'O' if self.player == 'X' else 'X'

    def check_winner(self):
        for i in range(3):
            if self.buttons[i][0]['text'] == self.buttons[i][1]['text'] == self.buttons[i][2]['text'] != "":
                return True
            if self.buttons[0][i]['text'] == self.buttons[1][i]['text'] == self.buttons[2][i]['text'] != "":
                return True
        if self.buttons[0][0]['text'] == self.buttons[1][1]['text'] == self.buttons[2][2]['text'] != "":
            return True
        if self.buttons[0][2]['text'] == self.buttons[1][1]['text'] == self.buttons[2][0]['text'] != "":
            return True
        return False

    def is_draw(self):
        for row in self.buttons:
            for button in row:
                if button['text'] == "":
                    return False
        return True

    def reset_game(self):
        self.player = 'X'
        for row in self.buttons:
            for button in row:
                button['text'] = ""

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()

class TitleScreen:
    def __init__(self, root, start_game_callback):
        self.root = root
        self.start_game_callback = start_game_callback
        self.frame = tk.Frame(self.root, bg='lightblue')
        self.frame.pack(expand=True, fill='both')
        
        self.title_label = tk.Label(self.frame, text="Tic-Tac-Toe", font=('Helvetica', 40, 'bold'), bg='lightblue')
        self.title_label.pack(pady=40)
        
        self.start_button = tk.Button(self.frame, text="Start Game", font=('Helvetica', 20), command=self.start_game, bg='white', fg='black', activebackground='green', activeforeground='white')
        self.start_button.pack(pady=20)
        
        self.quit_button = tk.Button(self.frame, text="Quit", font=('Helvetica', 20), command=self.root.quit, bg='white', fg='black', activebackground='red', activeforeground='white')
        self.quit_button.pack(pady=20)
        
    def start_game(self):
        self.frame.destroy()
        self.start_game_callback()

def start_game():
    game = TicTacToe(root)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x400")
    title_screen = TitleScreen(root, start_game)
    root.mainloop()