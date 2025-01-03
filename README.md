# CUDA TCP Project Demo (Windows)

This project demonstrates how to simulate a CUDA-based computational workload over TCP/IP on a Windows platform. The server performs data processing using GPU and handles multiple client requests concurrently, enabling high-performance computations through CUDA.

## Overview

This is a **single-file** project that combines both server and client functionality into one `.cpp` file, utilizing **multithreading** to concurrently handle multiple client requests. The server processes the commands sent by the client, performs CUDA-based operations (array addition, matrix multiplication), and sends the results back to the client.

- **Server**: Handles multiple client connections, processes various commands (like array addition and matrix multiplication), and offloads the computational work to the GPU using CUDA.
- **Client**: Sends commands to the server (like `add`, `matmul`, or `exit`), receives the results, and displays them. 

The server communicates with clients over **TCP/IP**, and the server is designed to handle multiple requests at the same time by creating a new thread for each client.

## Key Features

- **Multithreaded TCP Server**: Handles each client connection in a separate thread using C++ `std::thread`
- **CUDA Integration**: Offloads simple vector addition and matrix multiplication to GPU using CUDA kernels
- **Client-Server Communication**: Uses Windows Sockets (Winsock2) for TCP-based command exchange
- **Command-Based Interface**: Supports `add`, `matmul`, and `exit` commands for triggering GPU computations
- **Active Connection Tracking**: Uses `std::atomic<int>` to track connected clients in real time

## Requirements

To build and run this project, ensure you have the following installed:

- **CUDA Toolkit**: Required to compile and run CUDA-based functions. This project uses **CUDA 12.6** (latest version at the time).
- **Windows Operating System**: This project is designed to run on Windows-based platforms.
- **Visual Studio (or other IDE for CUDA & C++)**: Visual Studio is commonly used for CUDA development, but any IDE or build environment that supports C++ and CUDA should work.

## Dependencies

- **CUDA Toolkit**: For GPU-accelerated operations such as array addition and matrix multiplication.
- **Winsock2**: Windows-specific library for socket communication (built into Windows).

## Commands

### Command Prefix:
- **None**: No prefix is required for the commands.

### Command Suffixes:
- **`add`**: Triggers the CUDA-based array addition operation.
- **`matmul`**: Triggers the CUDA-based matrix multiplication operation.
- **`exit`**: Terminates the client-server connection.

### Example Client Interaction:

```sh
Enter command (add, matmul, exit): add
Response from server: Result: {11, 22, 33, 44, 55}

Enter command (add, matmul, exit): matmul
Response from server: Result: 30 24 18 84 69 54 138 114 90

Enter command (add, matmul, exit): exit
Server exiting...
