import socket
import time

def client_program():
    client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost',5000))

    def send(cmd):
        client.send(cmd.encode('utf-8'))
        data=client.recv(1024).decode('utf-8')
        print(f"FROM SERVER: {data}")
        return data
    quit=False
    while not quit:
        print("Enter command: LOCK, UNLOCK, READ, WRITE <value>, QUIT")
        command=input("-> ")
        if command.startswith("WRITE"):
            value=command[6:].strip()
            if value:
                send(f"WRITE {value}")
            else:
                print("No value provided for WRITE command.")
        elif command in ["LOCK", "UNLOCK", "READ", "QUIT"]:
            response = send(command)
            if command == "QUIT":
                quit = True
        else:
            print("Invalid command. Please try again.")

if __name__ == "__main__":
    client_program()