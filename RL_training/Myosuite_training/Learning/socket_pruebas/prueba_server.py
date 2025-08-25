import socket


print("Starting server")
socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_server.bind(('localhost', 9999))
print("Server started on port 9999")
socket_server.listen(1)

"""
socket_server2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_server2.bind(('localhost', 9998))
print("Server 2 started on port 9998")
socket_server2.listen(1)

# Cómo hacer que los dos estén activos a la vez? usando threading o asyncio?
"""
def process_msg(msg):
    variables = msg.split(", ")  # Assuming the message is a comma-separated string
    i =  0
    for var in variables:
        variables[i] = float(var)
        i += 1
    print(f"Processed variables: {variables}")
    return variables


while True:
    client_socket, addr = socket_server.accept() #Esta llamada bloquea hasta que un cliente se conecta
    print(f"Connection from {addr} has been established!")

    #client_socket2, addr2 = socket_server2.accept()
    #print(f"Connection from {addr2} has been established on server 2!")

    try:
        #while True:
            #data = client_socket.recv(1024) #Se guarda en data aunque no lo leas directamente
            #print(f"Received data: {data.decode()}")
        message = "1.0"
        client_socket.sendall(message.encode())
            #if not data:
                #print("No more data received, closing connection.")
                
                #break
    finally:
        client_socket.close()
        print("Connection closed.")
    
    client_socket, addr = socket_server.accept() #Esta llamada bloquea hasta que un cliente se conecta
    print(f"Connection from {addr} has been established!")
    try:
        
        data = client_socket.recv(1024) #Se guarda en data aunque no lo leas directamente
        print(f"Received observation before parsing: {data.decode()}")
        observation = process_msg(data.decode())
        print(f"Received observation after parsing: {observation}")
        
            #if not data:
                #print("No more data received, closing connection.")  
                #break
    finally:
        client_socket.close()
        print("Connection closed.")

