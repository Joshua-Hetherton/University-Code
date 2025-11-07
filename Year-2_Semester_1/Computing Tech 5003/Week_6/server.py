import socket
import threading


#Tracks which thread owns the lock
lock_owner= None

#Shared data
shared_obj="Initial Value"

"""Synchronization primitive
Not owned by any particular thread when it is locked
"""
thread_lock = threading.Lock()


def lock(command):
    global lock_owner
    if command[0]=="LOCK":
        if lock_owner is None:
            lock_owner = threading.current_thread().name
            return "LOCKED"
        else:
            return "LOCK FAILED"
        
def unlock(command):
    global lock_owner
    if command[0]=="UNLOCK":
        if lock_owner == threading.current_thread().name:
            lock_owner = None
            return "UNLOCKED"
        else:
            return "UNLOCK FAILED"
        

def read(command):
    global shared_obj
    if command[0]=="READ":
        return f"VALUE {shared_obj}"
    
def write(command):
    global shared_obj
    if command[0]=="WRITE":
        if len(command) > 1:
            shared_obj = command[1]
            return "WRITE SUCCESSFUL"
        else:
            return "WRITE FAILED: No value provided"

def quit(command):
    if command[0]=="QUIT":
        return "QUIT SUCCESSFUL"
    elif not threading.current_thread().name == lock_owner:
        return "Cannot QUIT: Lock not owned by you"

def handle_client(client_socket):
    global lock_owner
    global shared_obj
    print(f"[NEW CONNECTION] {client_socket.getpeername()} connected.")
    
    while True:
        #Checks for constant connection
        data=client_socket.recv(1024).decode('utf-8')
        if not data:
            break

        
        command = data.strip().split(" ", 1)
        response=""

        with thread_lock:
            if command[0] == "LOCK":
                response = lock(command)

            elif command[0] == "UNLOCK":
                response = unlock(command)

            else:
                response = "UNKNOWN COMMAND"
            if lock_owner == threading.current_thread().name:
                if command[0] == "READ":
                    response = read(command)

                elif command[0] == "WRITE":
                    response = write(command)

                elif command[0] == "QUIT":
                    response = quit(command)
                    client_socket.send(response.encode('utf-8'))
                    break
        client_socket.send(response.encode('utf-8'))
    client_socket.close()

#Starts Localhost server
def start_server(host='localhost', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[LISTENING] Server is listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        #Tells when a new connection is made
        print(f"[NEW CONNECTION] {addr} connected.")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()


if __name__ == "__main__":
    start_server()