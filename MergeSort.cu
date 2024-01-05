#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SIZE
#define SIZE 124
#endif

#ifndef PINNED
#define PINNED 0
#endif

// Inicializa un vector de N elementos
void InitVector(int N, int *V);

// Imprime por stdout el vector pasado por parámetro
void PrintVector(int* x, unsigned int size);

__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end);

__global__ void MergeSort (int *vector, int *vres, int N, int width, int slices);

int main(int argc, char** argv)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    //float TiempoTotal, TiempoKernel;
    cudaEvent_t E0, E1, E2, E3;
    float TiempoTotal, TiempoKernel;
    int *h_vector;
    int *d_vector;

    int *hresult_vector;
    int *dresult_vector;

    // Dimension del vector y comprobacion resultado
    if (argc == 1)      { N = 1024; }
    else if (argc == 2) { N = atoi(argv[1]); }
    // else if (argc == 3) { test = *argv[2]; N = atoi(argv[1]); }
    else { printf("Usage: ./mergesort TAM(N) \n"); exit(0); }

    int count, gpu;
    // Buscar GPU de forma aleatoria
    cudaGetDeviceCount(&count);
    srand(time(NULL));
    gpu = (rand()>>3) % count;
    cudaSetDevice(gpu);

    // numero de Threads en cada dimension
    nThreads = SIZE;

    // numero de Blocks en cada dimension
    //nBlocks =N/nThreads;
    nBlocks = (N+nThreads-1)/nThreads;

    numBytes = N * sizeof(int);

    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1, 1);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    if (PINNED) {
        // Obtiene Memoria [pinned] en el host
        cudaMallocHost((int**)&h_vector, numBytes);
        cudaMallocHost((int**)&hresult_vector, numBytes);
    }
    else {
        // Obtener Memoria en el host
        h_vector = (int*) malloc(numBytes);
        hresult_vector = (int*) malloc(numBytes);
    }

    // Inicializa el vector
    InitVector(N, h_vector);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // Obtener Memoria en el device
    cudaMalloc((int**)&d_vector, numBytes);
    cudaMalloc((int**)&dresult_vector, numBytes);

    // Copiar datos desde el host en el device
    cudaMemcpy(d_vector, h_vector, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dresult_vector, hresult_vector, numBytes, cudaMemcpyHostToDevice);

 /*   int test = N;
    while (test > 2) test /=4;
    if (test == 2) printf("HOLA");*/
    
    printf("El vector SIN ordenar:\n");
    PrintVector(h_vector,N);
    printf("\n");

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);


    int* A = d_vector;
    int* B = dresult_vector;

    for (int width = 1; width < (N << 1); width <<= 1){
        int slices = N / ((nThreads) * width) + 1;

        // Ejecutar el kernel
        MergeSort<<<dimGrid, dimBlock>>>(A, B, N, width, slices);

        // Switch the input / output arrays instead of copying them around
        A = A == d_vector ? dresult_vector : d_vector;
        B = B == d_vector ? dresult_vector : d_vector;

    }

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    // Copiar datos desde el device en el Host
    cudaMemcpy(h_vector, d_vector, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hresult_vector, dresult_vector, numBytes, cudaMemcpyDeviceToHost);

    // Liberar Memoria del device
    cudaFree(d_vector);
    

    cudaDeviceSynchronize();

    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);

    cudaEventElapsedTime(&TiempoTotal, E0, E3);
    cudaEventElapsedTime(&TiempoKernel, E1, E2);

    cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

    printf("El vector ordenado:\n");
    PrintVector(hresult_vector, N);

    printf("\n");

    printf("N Elementos: %d\n", N);
    printf("nThreads: %d\n", nThreads);
    printf("nBlocks: %d\n", nBlocks);
    printf("Tiempo Paralelo Global: %4.6f milseg\n", TiempoTotal);
    printf("Tiempo Paralelo Kernel: %4.6f milseg\n", TiempoKernel);
   // printf("Tiempo Secuencial: %4.6f milseg\n", t2-t1);
    printf("Rendimiento Paralelo Global: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TiempoTotal));
    printf("Rendimiento Paralelo Kernel: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TiempoKernel));
   // printf("Rendimiento Secuencial: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * (t2 - t1)));


}

// Inicializa un vector de N elementos
void InitVector(int N, int *V) {
    for (int i = 0; i < N; i++){
        V[i] = rand() % 1000; 
    }       
}

// Imprime por stdout el vector pasado por parámetro
void PrintVector(int* list, unsigned int size){
    printf("[");
    for(int i=0; i<size; ++i){
        printf("%i", list[i]);
        if (i != size-1) printf(",");
    }
    printf("]\n");
}

__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

__global__ void MergeSort (int *vector, int *vres, int N, int width, int slices) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = width*idx*slices;
    int middle,end;

    for (int slice = 0; slice < slices; slice++) {
        if (start >= N) break;
        middle = min(start + (width >> 1), N);
        end = min(start + width, N);
        gpu_bottomUpMerge(vector, vres, start, middle, end);
        start += width;
    }
   /* int test = N;
    while (test > 2) test /=4;
    if (test == 2) gpu_bottomUpMerge(vector, vres, 0, N/2, N);*/
}


