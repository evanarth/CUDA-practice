#include  <stdio.h>

__global__ void modifyArray (int *modArray) {
    
    int i = threadIdx.x;
    modArray[i] = modArray[i] + 100;
}

__host__ int main (void) {
    
    int lenArray = 10;
    int *modArray, *gpu_modArray;
    size_t sizeArray;
    
    sizeArray = lenArray * sizeof(int);
    modArray  = (int*) malloc( sizeArray );
    cudaMalloc( &gpu_modArray, sizeArray );
    
    printf("original values\n");
    for ( int i = 0; i < lenArray; i++ ) {
        modArray[i] = i + 1;
        printf("%d ", modArray[i]);
    }
    
    cudaMemcpy( gpu_modArray, modArray, 
        sizeArray, cudaMemcpyHostToDevice );
    
    modifyArray <<< 1, lenArray >>> (gpu_modArray);
    
    cudaMemcpy( modArray, gpu_modArray, 
        sizeArray, cudaMemcpyDeviceToHost );
    
    printf("\nfinal values\n");
    for ( int i = 0; i < lenArray; i++ )
        printf("%d ", modArray[i]);
    printf("\n");
    
    free( modArray );
    cudaFree( gpu_modArray );
    
    return 0;
    
}
