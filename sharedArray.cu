#include  <stdio.h>

#define WIDTH_ARRAY   10
#define HEIGHT_ARRAY  10

__global__ void shareArray (int *modArray, int *sum) {
    
    int i, j, n, index, countVal;
    __shared__ int sumColumn[WIDTH_ARRAY], 
        tmpModArray[WIDTH_ARRAY][HEIGHT_ARRAY];
    
    i = threadIdx.x;
    j = threadIdx.y;
    index = i + j * WIDTH_ARRAY;
    
    tmpModArray[i][j] = -1 * modArray[index];
    
    __syncthreads();
    
    if (threadIdx.y == 0) {
        countVal = 0;
        for (n = 0; n < HEIGHT_ARRAY; n++)
            countVal = countVal + tmpModArray[i][n];
        sumColumn[i] = countVal;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        countVal = 0;
        for (n = 0; n < WIDTH_ARRAY; n++)
            countVal = countVal + sumColumn[n];
        sum[0] = countVal;
    }
}

__host__ int main (void) {
    
    int index, i, j;
    int *modArray, *gpu_modArray, *sum, *gpu_sum;
    size_t sizeArray;
    
    sizeArray = WIDTH_ARRAY * HEIGHT_ARRAY * sizeof(int);
    modArray  = (int*) malloc( sizeArray );
    cudaMalloc( &gpu_modArray, sizeArray );
    sum       = (int*) malloc( sizeof(int) );
    cudaMalloc( &gpu_sum,      sizeof(int) );
    
    printf("original values\n");
    for ( i = 0; i < WIDTH_ARRAY; i++ ) {
        for ( j = 0; j < HEIGHT_ARRAY; j++ ) {
            index = i + j * WIDTH_ARRAY;
            modArray[index] = index + 1;
            printf("%d ", modArray[index]);
        }
        printf("\n");
    }
    
    cudaMemcpy( gpu_modArray, modArray, 
        sizeArray, cudaMemcpyHostToDevice );
    
    dim3    threads1(WIDTH_ARRAY, HEIGHT_ARRAY);
    shareArray <<< 1, threads1 >>> (gpu_modArray, gpu_sum);
    
    cudaMemcpy( sum, gpu_sum, 
        sizeof(int), cudaMemcpyDeviceToHost );
    
    printf("\nfinal sum\n %d\n\n", sum[0]);
    
    free( modArray );
    cudaFree( gpu_modArray );
    
    return 0;
    
}
