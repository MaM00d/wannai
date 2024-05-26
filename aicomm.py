import socket

def process_message(message):
    # Example function to process the message
    return message.upper()

def send_message(client_socket, message):
    client_socket.sendall(message.encode())
    print(f"Sent to client: {message}")

def receive_message(client_socket):
    data = client_socket.recv(1024)
    if not data:
        return None
    message = data.decode()
    print(f"Received from client: {message}")
    return message

def start_server(host='localhost', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}")

        client_socket, client_address = server_socket.accept()
        print(f"Connected by {client_address}")

        while True:
            message = receive_message(client_socket)
            if message is None:
                print("no message")
                break
            result = process_message(message)
            send_message(client_socket, result)

        # print(f"Connection with {client_address} closed")

if __name__ == "__main__":
    start_server()
