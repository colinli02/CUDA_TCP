# CUDA TCP Project Demo (for Windows) 
Simulate CUDA cpp code that receives data for for processing in GPU 
TCP / IP as major protocol 

In typical TCP/IP, we build it with 2 projects (server.cpp & client.cpp).

Instead here, is just simulating TCP/IP so we build it in 1 .cpp file using threading

For "cloud like" behaviour, requests are processed server side (within the server function).

## Requirements
```
CUDA Toolkit
Windows
Visual Studio (or preferred IDE for CUDA & C++)
```

## Commands
```
add
matmul
exit
```


# Technical explanation of code components
   
'<cuda_runtime.h>'

'extern "C" '
Linked with C 

'cudaError_t'
return type

# Future Possible Implementations

Consider GPT like commmand processing through user input text instead of hard coded commands.
i.e. with classifier & data set to train.
