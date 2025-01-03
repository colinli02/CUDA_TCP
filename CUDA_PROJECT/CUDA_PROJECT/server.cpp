#include <iostream>
#include <thread>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include <sstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BACKLOG 10

extern "C" cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
extern "C" cudaError_t matmulWithCuda(int* C, const int* A, const int* B, unsigned int N);

void startServer() 
{
    WSADATA wsaData;
    SOCKET server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = { 0 };

    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }
    std::cout << "WSAStartup succeeded\n";

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Socket creation failed\n";
        WSACleanup();
        return;
    }
    std::cout << "Socket created\n";

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0) {
        std::cerr << "Setsockopt failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }
    std::cout << "Setsockopt succeeded\n";

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }
    std::cout << "Bind succeeded\n";

    if (listen(server_fd, BACKLOG) < 0) {
        std::cerr << "Listen failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }
    std::cout << "Listening...\n";

    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen)) == INVALID_SOCKET) {
        std::cerr << "Accept failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }
    std::cout << "Connection accepted\n";

    // Main loop for handling commands
    while (true) {
        recv(new_socket, buffer, static_cast<int>(sizeof(buffer)), 0);
        std::string command(buffer);

        if (command == "exit") {
            std::cout << "Exiting...\n";
            send(new_socket, "Server exiting...", 18, 0);
            break;
        }
        else if (command == "add") {
            const int arraySize = 5;
            const int a[arraySize] = { 1, 2, 3, 4, 5 };
            const int b[arraySize] = { 10, 20, 30, 40, 50 };
            int c[arraySize] = { 0 };

            std::cout << "Performing CUDA add operation\n";
            cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addWithCuda failed!");
                send(new_socket, "CUDA operation failed", 21, 0);
            }
            else {
                std::cout << "CUDA operation succeeded\n";
                std::string result = "Result: {" + std::to_string(c[0]) + "," + std::to_string(c[1]) + ","
                    + std::to_string(c[2]) + "," + std::to_string(c[3]) + "," + std::to_string(c[4]) + "}";
                send(new_socket, result.c_str(), static_cast<int>(result.size()), 0);
                std::cout << "Result sent to client: " << result << std::endl;
            }
        }
        else if (command == "matmul") {
            // Perform Matrix Multiplication
            const int N = 3;  // 3x3 matrix
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

            std::cout << "Performing CUDA matrix multiplication\n";
            cudaError_t cudaStatus = matmulWithCuda((int*)C, (int*)A, (int*)B, N);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "matmulWithCuda failed!" << std::endl;
                send(new_socket, "CUDA matrix multiplication failed", 33, 0);
            }
            else {
                std::cout << "Matrix multiplication succeeded" << std::endl;
                std::ostringstream result;
                result << "Result: ";
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        result << C[i][j] << " ";
                    }
                }
                send(new_socket, result.str().c_str(), static_cast<int>(result.str().size()), 0);
                std::cout << "Result sent to client: " << result.str() << std::endl;
            }
        }
        else {
            std::string errorMsg = "Unknown command: " + command;
            send(new_socket, errorMsg.c_str(), static_cast<int>(errorMsg.size()), 0);
            std::cout << errorMsg << std::endl;
        }
    }

    closesocket(new_socket);
    closesocket(server_fd);
    WSACleanup();
}

void startClient() {
    WSADATA wsaData;
    SOCKET sock = INVALID_SOCKET;
    struct sockaddr_in serv_addr;
    char buffer[1024] = { 0 };

    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }
    std::cout << "WSAStartup succeeded (client)\n";

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Socket creation failed (client)\n";
        WSACleanup();
        return;
    }
    std::cout << "Socket created (client)\n";

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (InetPton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/Address not supported (client)\n";
        closesocket(sock);
        WSACleanup();
        return;
    }
    std::cout << "Address converted (client)\n";

    std::this_thread::sleep_for(std::chrono::seconds(1)); // Give server time to start

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed (client)\n";
        closesocket(sock);
        WSACleanup();
        return;
    }
    std::cout << "Connected to server\n";

    std::string input;
    while (true) {
        std::cout << "Enter command (add, exit, matmul): ";
        std::getline(std::cin, input);

        send(sock, input.c_str(), static_cast<int>(input.size()), 0);
        if (input == "exit") {
            break;
        }

        recv(sock, buffer, static_cast<int>(sizeof(buffer)), 0);
        std::cout << "Response from server: " << buffer << std::endl;
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
