#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SIZE
#define SIZE 124
#endif

#ifndef PINNED
#define PINNED 0
#endif

// Initialises a vector of N elements
void InitVector(int N, int *V);

// Prints to stdout the vector passed by parameter
void PrintVector(int* x, unsigned int size);

//funcions for parallel mergesort 
__device__ void MergeGPU(int* source, int* dest, int start, int middle, int end);
__global__ void MergeSort (int *vector, int *vres, int N, int width, int slices);

//functions for sequential mergesort
void MergeSortSequencial(int* arr, int l, int r);
void MergeSequencial(int arr[], int l, int m, int r);

int main(int argc, char** argv)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    //float TotalTime, KernelTime;
    cudaEvent_t E0, E1, E2, E3, E4;
    float TotalTime, KernelTime, SeqTime;
    int *h_vector;
    int *d_vector;

    int *hresult_vector;
    int *dresult_vector;

    // vector size and parameters check
    if (argc == 1)      { N = 1024; }
    else if (argc == 2) { N = atoi(argv[1]); }
    else { printf("Usage: ./mergesort TAM(N) \n"); exit(0); }

    int count, gpu;
    // Random GPU search
    cudaGetDeviceCount(&count);
    srand(time(NULL));
    gpu = (rand()>>3) % count;
    cudaSetDevice(gpu);

    // thread num in each dimension
    nThreads = SIZE;

    // block num in each dimension
    //nBlocks =N/nThreads;
    nBlocks = (N+nThreads-1)/nThreads;

    numBytes = N * sizeof(int);

    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1, 1);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);
    cudaEventCreate(&E4);

    if (PINNED) {
        // obtains [pinned] memory in host
        cudaMallocHost((int**)&h_vector, numBytes);
        cudaMallocHost((int**)&hresult_vector, numBytes);
    }
    else {
        // obtain memory on host
        h_vector = (int*) malloc(numBytes);
        hresult_vector = (int*) malloc(numBytes);
    }

    // initialises the vector
    InitVector(N, h_vector);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // obtain device memory
    cudaMalloc((int**)&d_vector, numBytes);
    cudaMalloc((int**)&dresult_vector, numBytes);

    // Copy data from host to device
    cudaMemcpy(d_vector, h_vector, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dresult_vector, hresult_vector, numBytes, cudaMemcpyHostToDevice);
    
    printf("El vector SIN ordenar:\n");
    PrintVector(h_vector,N);
    printf("\n");

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);

    int* A = d_vector;
    int* B = dresult_vector;

    for (int width = 1; width < (N << 1); width <<= 1){
        int slices = N / ((nThreads) * width) + 1;

        // Kernel exec
        MergeSort<<<dimGrid, dimBlock>>>(A, B, N, width, slices);

        A = A == d_vector ? dresult_vector : d_vector;
        B = B == d_vector ? dresult_vector : d_vector;
    }

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    // Copy data from device to host
    cudaMemcpy(h_vector, d_vector, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hresult_vector, dresult_vector, numBytes, cudaMemcpyDeviceToHost);

    // Unlock Device Memory
    cudaFree(d_vector);
    
    cudaDeviceSynchronize();

    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);
    /*----------------------------*/
    /* Sequential code starts here*/

    int* arr = d_vector;
    MergeSortSequencial(arr, 0, N-1);

    cudaEventRecord(E4, 0);
    cudaEventSynchronize(E4);

    /* Sequential code ends here*/
    /*----------------------------*/

    cudaEventElapsedTime(&TotalTime, E0, E3);
    cudaEventElapsedTime(&KernelTime, E1, E2);
    cudaEventElapsedTime(&SeqTime, E3, E4);

    cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3); cudaEventDestroy(E4);

    printf("El vector ordenado:\n");
    PrintVector(hresult_vector, N);

    printf("\n");

    printf("N Elements: %d\n", N);
    printf("nThreads: %d\n", nThreads);
    printf("nBlocks: %d\n", nBlocks);
    printf("Global Parallel Time: %4.6f milseg\n", TotalTime);
    printf("Kernel Parallel Time: %4.6f milseg\n", KernelTime);
    printf("Sequential time: %4.6f milseg\n", KernelTime);
    printf("Global Parallel Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TotalTime));
    printf("Kernel Parallel Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * KernelTime));
    printf("Sequential Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * SeqTime));
}

// Inicialice N elements vector
void InitVector(int N, int *V) {
    for (int i = 0; i < N; i++){
        V[i] = rand() % 1000; 
    }       
}

// Print the vector (list)
void PrintVector(int* list, unsigned int size){
    printf("[");
    for(int i=0; i<size; ++i){
        printf("%i", list[i]);
        if (i != size-1) printf(",");
    }
    printf("]\n");
}

/* Parallel functions*/
__device__ void MergeGPU(int* source, int* dest, int start, int middle, int end) {
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

__global__ void MergeSort(int *vector, int *vres, int N, int width, int slices) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = width*idx*slices;
    int middle,end;

    for (int slice = 0; slice < slices; slice++) {
        if (start >= N) break;
        middle = min(start + (width >> 1), N);
        end = min(start + width, N);
        MergeGPU(vector, vres, start, middle, end);
        start += width;
    }
}

/*-------------------------*/
/* Sequential functions*/
void MergeSortSequencial(int* arr, int l, int r) {
    if (l < r) {
        // Encuentra el punto medio del arreglo
        int m = l + (r - l) / 2;

        // Ordena la primera y segunda mitad
        MergeSortSequencial(arr, l, m);
        MergeSortSequencial(arr, m + 1, r);

        // Combina las mitades ordenadas
        MergeSequencial(arr, l, m, r);
    }
}

void MergeSequencial(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Crear subarreglos temporales
    int L[n1], R[n2];

    // Copiar datos a los subarreglos temporales L[] y R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Combinar los subarreglos temporales de nuevo en arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copiar los elementos restantes de L[], si los hay
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copiar los elementos restantes de R[], si los hay
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}
/*-------------------------*/


