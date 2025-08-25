import socket

print("Starting client")
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = int(input("Enter port to connect to (9999 or 9998): "))
socket_client.connect(('localhost', port))
print(f"Connected to server on port {port}")

try:
    message = input("Enter message to send: ")
    socket_client.sendall(message.encode())
    print("Message sent!")

    response = socket_client.recv(1024)
    print(f"Response from server: {response.decode()}")
    
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    socket_client.close()
    print("Connection closed.")

