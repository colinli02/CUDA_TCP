#include <iostream>
#include <thread>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <sstream>
#include <atomic>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BACKLOG 10

extern "C" cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
extern "C" cudaError_t matmulWithCuda(int* C, const int* A, const int* B, unsigned int N);

// Atomic counter for the number of active connections
std::atomic<int> activeConnections(0);

// Function to handle each client request
void handleClient(SOCKET clientSocket) {
    char buffer[1024] = { 0 };
    std::string command;

    activeConnections++; // Increment active connections count when a client connects
    std::cout << "New connection established. Active connections: " << activeConnections.load() << std::endl;

    while (true) {
        int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
        if (bytesReceived <= 0) {
            break; // Connection closed or error
        }

        command = std::string(buffer, bytesReceived); // Capture the exact received data
        memset(buffer, 0, sizeof(buffer)); // Clear the buffer after reading

        if (command == "add") {
            const int arraySize = 5;
            const int a[arraySize] = { 1, 2, 3, 4, 5 };
            const int b[arraySize] = { 10, 20, 30, 40, 50 };
            int c[arraySize] = { 0 };

            std::cout << "Performing CUDA add operation\n" << std::flush;
            cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
            if (cudaStatus != cudaSuccess) {
                send(clientSocket, "CUDA operation failed", 21, 0);
            }
            else {
                std::string result = "Result: {" + std::to_string(c[0]) + "," + std::to_string(c[1]) + ","
                    + std::to_string(c[2]) + "," + std::to_string(c[3]) + "," + std::to_string(c[4]) + "}";
                send(clientSocket, result.c_str(), result.size(), 0);
                std::cout << "Result sent to client: " << result << std::endl;
            }
        }
        else if (command == "matmul") {
            const int N = 3;
            int A[N][N] = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
            };
            int B[N][N] = {
                {9, 8, 7},
                {6, 5, 4},
                {3, 2, 1}
            };
            int C[N][N] = { 0 };

            std::cout << "Performing CUDA matrix multiplication\n" << std::flush;
            cudaError_t cudaStatus = matmulWithCuda((int*)C, (int*)A, (int*)B, N);
            if (cudaStatus != cudaSuccess) {
                send(clientSocket, "CUDA matrix multiplication failed", 33, 0);
            }
            else {
                std::ostringstream result;
                result << "Result: ";
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        result << C[i][j] << " ";
                    }
                }
                send(clientSocket, result.str().c_str(), result.str().size(), 0);
                std::cout << "Result sent to client: " << result.str() << std::endl;
            }
        }
        else if (command == "exit") {
            std::cout << "Client requested exit\n";
            send(clientSocket, "Server exiting...", 18, 0);
            break;
        }
        else {
            std::string errorMsg = "Unknown command: " + command;
            send(clientSocket, errorMsg.c_str(), errorMsg.size(), 0);
            std::cout << errorMsg << std::endl;
        }

        // Print active connections after command handling
        std::cout << "Active connections after command: " << activeConnections.load() << std::endl;
    }

    // Close the client socket and decrement active connections
    closesocket(clientSocket); // Close the client socket
    activeConnections--; // Decrement active connections count
    std::cout << "Connection closed. Active connections: " << activeConnections.load() << std::endl;
}




// Main server function
void startServer() {
    WSADATA wsaData;
    SOCKET serverSocket, clientSocket;
    struct sockaddr_in serverAddr;
    int addrlen = sizeof(serverAddr);
    int opt = 1;

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }

    // Create a socket
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed\n";
        WSACleanup();
        return;
    }

    // Set socket options
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0) {
        std::cerr << "Setsockopt failed\n";
        closesocket(serverSocket);
        WSACleanup();
        return;
    }

    // Set up the server address
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    // Bind the socket
    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Bind failed\n";
        closesocket(serverSocket);
        WSACleanup();
        return;
    }

    // Listen for incoming connections
    if (listen(serverSocket, BACKLOG) < 0) {
        std::cerr << "Listen failed\n";
        closesocket(serverSocket);
        WSACleanup();
        return;
    }

    std::cout << "Server is listening for connections...\n";

    // Accept connections and spawn new threads for each client
    while (true) {
        clientSocket = accept(serverSocket, (struct sockaddr*)&serverAddr, &addrlen);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Accept failed\n";
            continue;
        }

        std::cout << "New client connected. Active connections: " << activeConnections.load() << std::endl;

        // Spawn a new thread for each new client
        std::thread clientThread(handleClient, clientSocket);
        clientThread.detach(); // Detach the thread so it runs independently
    }

    closesocket(serverSocket);
    WSACleanup();
}

// Client function
void startClient() {
    WSADATA wsaData;
    SOCKET sock;
    struct sockaddr_in serverAddr;
    char buffer[1024] = { 0 };

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }

    // Create a socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed (client)\n";
        WSACleanup();
        return;
    }

    // Set up the server address
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    if (InetPton(AF_INET, "127.0.0.1", &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address/Address not supported (client)\n";
        closesocket(sock);
        WSACleanup();
        return;
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed (client)\n";
        closesocket(sock);
        WSACleanup();
        return;
    }

    std::cout << "Connected to server\n";

    std::string input;
    while (true) {
        std::cout << "Enter command (add, matmul, exit): ";
        std::getline(std::cin, input);

        // Send command to server
        send(sock, input.c_str(), input.size(), 0);
        if (input == "exit") {
            break;
        }

        // Receive response from server
        memset(buffer, 0, sizeof(buffer)); // Clear buffer before each receive
        int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
        if (bytesReceived > 0) {
            std::cout << "Response from server: " << buffer << std::endl;
        }
        else {
            std::cerr << "Failed to receive response from server\n";
        }
    }

    closesocket(sock);
    WSACleanup();
}


int main() {
    // Start TCP/IP server and client
    std::thread serverThread(startServer);
    std::thread clientThread(startClient);

    serverThread.join();
    clientThread.join();

    return 0;
}
