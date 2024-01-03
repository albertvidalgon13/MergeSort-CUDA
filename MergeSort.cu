#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SIZE
#define SIZE 32
#endif

#ifndef PINNED
#define PINNED 0
#endif

__global__ void MergeSort (int N, int *vector) {
    
    for(int i=0; i<N; ++i){
        vector[i] = 4;
    }

}

void InitV(int N, int *V);

int main(int argc, char** argv)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    //float TiempoTotal, TiempoKernel;
    cudaEvent_t E0, E1, E2, E3;

    int *h_vector;
    int *d_vector;

    // char test;

    // ./mergesort N

    // Dimension de las matrices NxN y comprobacion resultado
    if (argc == 1)      { N = 1024; }
    else if (argc == 2) { N = atoi(argv[1]); }
    // else if (argc == 3) { test = *argv[2]; N = atoi(argv[1]); }
    else { printf("Usage: ./mergesort TAM(N) \n"); exit(0); }

    int count, gpu;
    // Buscar GPU de forma aleatoria
    cudaGetDeviceCount(&count);
    srand(time(NULL));
    gpu = (rand()>>3) % count;
    cudaSetDevice(1);

    // numero de Threads en cada dimension
    nThreads = SIZE;

    // numero de Blocks en cada dimension
    //nBlocks =N/nThreads;
    nBlocks = (N+nThreads-1)/nThreads;

    numBytes = N * sizeof(int);

    dim3 dimGrid(nBlocks, nBlocks, 1);
    dim3 dimBlock(nThreads, nThreads, 1);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    if (PINNED) {
        // Obtiene Memoria [pinned] en el host
        cudaMallocHost((int**)&h_vector, numBytes);
    }
    else {
        // Obtener Memoria en el host
        h_vector = (int*) malloc(numBytes);
    }

    // Inicializa el vector
    InitV(N, h_vector);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // Obtener Memoria en el device
    cudaMalloc((int**)&d_vector, numBytes);

    // Copiar datos desde el host en el device
    cudaMemcpy(d_vector, h_vector, numBytes, cudaMemcpyHostToDevice);

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);
    
    for(int i=0; i<N; ++i){
        printf("%i", h_vector[i]);
        printf(",");
    }
    printf("\n");

    // Ejecutar el kernel
    MergeSort<<<dimGrid, dimBlock>>>(N, d_vector);

    // Copiar datos desde el device en el Host
    cudaMemcpy(h_vector, d_vector, numBytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<N; ++i){
        printf("%i", h_vector[i]);
        printf(",");
    }

}

void InitV(int N, int *V) {
    int i;
        for (i=0; i<N; i++){
           V[i] = rand(); 
        }       
}
