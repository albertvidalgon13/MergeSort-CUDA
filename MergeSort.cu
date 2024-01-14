#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/times.h>
#include <sys/resource.h>

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

float GetTime(void);

/*-------------------------*/
/* Sequential functions*/

void MergeSequential(int *array, int p, int q, int r) {
    // Variable declaration
    int i, j, k;
    int n_1 = (q - p) + 1;
    int n_2 = (r - q);
    int *L, *R;

    // Mem assign
    L = (int*)malloc(n_1 * sizeof(int));
    R = (int*)malloc(n_2 * sizeof(int));

    // data copy
    for (i = 0; i < n_1; i++)
    {
        L[i] = *(array + p + i);
    }

    for (j = 0; j < n_2; j++)
    {
        R[j] = *(array + q + j + 1);
    }

    i = 0;
    j = 0;

    // data fusion
    for (k = p; k < r + 1; k++)
    {
        if (i == n_1)
        {
            *(array + k) = *(R + j);
            j =  j+ 1;
        }
        else if(j == n_2)
        {
            *(array + k) = *(L + i);
            i = i + 1;
        }
        else
        {
            if (*(L + i) <= *(R + j))
            {
                *(array + k) = *(L + i);
                i = i + 1;
            }
            else
            {
                *(array + k) = *(R + j);
                j = j + 1;
            }
        }
    }
}


void MergeSortSequential(int* array, int p, int r) {
    if (p < r)
    {
        // divide problem
        int q = (p + r)/2;
        
        // recursive for trivial solution
        MergeSortSequential(array, p, q);
        MergeSortSequential(array, q + 1, r);
        
        // fusion of partial divisions
        MergeSequential(array, p, q, r);
    }
}


//-----------------------------

int main(int argc, char** argv)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    //float TotalTime, KernelTime;
    cudaEvent_t E0, E1, E2, E3;
    float TotalTime, KernelTime;
    int *h_vector;
    int *d_vector;
    float t1,t2;

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

        if (A == d_vector) A = dresult_vector;
        else A = d_vector;

        if (B == d_vector) B = dresult_vector;
        else B = d_vector;
    }

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    // Copy data from device to host
    cudaMemcpy(h_vector, d_vector, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hresult_vector, dresult_vector, numBytes, cudaMemcpyDeviceToHost);

    printf("El vector ordenado:\n");
    PrintVector(hresult_vector, N);

    // Unlock Device Memory
    cudaFree(d_vector);
    
    cudaDeviceSynchronize();

    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);
    
    cudaEventElapsedTime(&TotalTime, E0, E3);
    cudaEventElapsedTime(&KernelTime, E1, E2);

    cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

    /*----------------------------*/
    /* Sequential code starts here*/

    t1=GetTime();
    int* arr = h_vector;
    int r = sizeof(arr)/sizeof(arr[0]) - 1, p = 0;
    MergeSortSequential(arr, p, r);
    t2=GetTime();

    /* Sequential code ends here*/
    /*----------------------------*/

    printf("\n");

    printf("N Elements: %d\n", N);
    printf("nThreads: %d\n", nThreads);
    printf("nBlocks: %d\n", nBlocks);
    printf("Global Parallel Time: %4.6f milseg\n", TotalTime);
    printf("Kernel Parallel Time: %4.6f milseg\n", KernelTime);
    printf("Sequential time: %4.6f milseg\n", t2-t1);
    printf("Global Parallel Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TotalTime));
    printf("Kernel Parallel Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * KernelTime));
    printf("Sequential Performance: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * (t2-t1)));
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

float GetTime(void)        {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

