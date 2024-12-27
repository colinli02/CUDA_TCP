#include <iostream>
#include <thread>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BACKLOG 10

void startServer() {
    WSADATA wsaData;
    SOCKET server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    const char *hello = "Hello from server";

    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Socket creation failed\n";
        WSACleanup();
        return;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0) {
        std::cerr << "Setsockopt failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    if (listen(server_fd, BACKLOG) < 0) {
        std::cerr << "Listen failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen)) == INVALID_SOCKET) {
        std::cerr << "Accept failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    recv(new_socket, buffer, static_cast<int>(sizeof(buffer)), 0);
    std::cout << "Server received: " << buffer << std::endl;
    send(new_socket, hello, static_cast<int>(strlen(hello)), 0);
    std::cout << "Hello message sent from server\n";

    closesocket(new_socket);
    closesocket(server_fd);
    WSACleanup();
}

void startClient() {
    WSADATA wsaData;
    SOCKET sock = INVALID_SOCKET;
    struct sockaddr_in serv_addr;
    const char *message = "Hello from client";
    char buffer[1024] = {0};

    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Socket creation failed\n";
        WSACleanup();
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (InetPton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/Address not supported\n";
        closesocket(sock);
        WSACleanup();
        return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1)); // Give server time to start

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed\n";
        closesocket(sock);
        WSACleanup();
        return;
    }

    send(sock, message, static_cast<int>(strlen(message)), 0);
    std::cout << "Hello message sent from client\n";
    recv(sock, buffer, static_cast<int>(sizeof(buffer)), 0);
    std::cout << "Client received: " << buffer << std::endl;

    closesocket(sock);
    WSACleanup();
}

int main() {
    std::thread serverThread(startServer);
    std::thread clientThread(startClient);

    serverThread.join();
    clientThread.join();

    return 0;
}
