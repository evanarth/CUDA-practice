/* 
 * this program is a simple test of the atomicAdd function for serial-dependent
 * addition of results
 * 
 */

#include <iostream>
#define TOTAL_SIZE 100000
#define nTPB 256

#define NUM_ATOMS 20
#define NUM_THREADS 12
#define NUM_BLOCKS 10

#define LENGTH_LOOKUP  240

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void kernelCode(float *result) {
    int index = threadIdx.x+blockIdx.x*blockDim.x;
    if (index < TOTAL_SIZE)
    {
        atomicAdd(result, 1.0f);
    }
}

__global__ void atomicAddTestKernel( float *inputAtoms ) {
    
    float num2Add = (float) threadIdx.x+blockIdx.x*blockDim.x;
    for ( int i = 0; i < NUM_ATOMS; i++ )
        atomicAdd( &inputAtoms[i], num2Add );
    
}

__global__ void atomicAddLookupTableKernel ( int *lookup ) {
    
    int index = atomicAdd( &lookup[0], 1 );
//    printf("%d  %d\n", index, threadIdx.x);
    lookup[index+1] = threadIdx.x+1;
}


int main(){
    
    //--------------------------------------------------------------------------
    // stock code (one number)
    
    // allocate variable on the GPU
    float h_result, *d_result;
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaCheckErrors("cuda malloc fail");
    
    // copy local variable to GPU
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy 1 fail");
    
    // run atomicAdd kernel
    kernelCode<<<(TOTAL_SIZE+nTPB-1)/nTPB, nTPB>>>(d_result);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel fail");
    
    // copy back result to local memory
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy 2 fail");
    
    std::cout<< "result = " << h_result << std::endl;
    
    
    //--------------------------------------------------------------------------
    // practice code (array)
    
    // allocate variable on the GPU and CPU
    int i;
    float *inputAtoms, *d_inputAtoms;
    size_t sizeArray = sizeof(float) * NUM_ATOMS;
    
    inputAtoms = (float*) malloc ( sizeArray );
    cudaMalloc( &d_inputAtoms, sizeArray );
    cudaCheckErrors("cuda malloc fail");
    
    // copy local variable to GPU
    for ( i = 0; i < NUM_ATOMS; ++i ) {
        inputAtoms[i] = 1.0f;
        printf("%d ", (int)inputAtoms[i]);
    } printf("\n");
    cudaMemcpy( d_inputAtoms, inputAtoms, sizeArray, cudaMemcpyHostToDevice );
    cudaCheckErrors("cudaMemcpy 1 fail");
    
    // run atomicAdd kernel
    atomicAddTestKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_inputAtoms);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel fail");
    
    // copy back result to local memory
    cudaMemcpy( inputAtoms, d_inputAtoms, sizeArray, cudaMemcpyDeviceToHost );
    cudaCheckErrors("cudaMemcpy 2 fail");
    
    // report results and close
    for ( i = 0; i < NUM_ATOMS; ++i )
        printf("%d ", (int)inputAtoms[i]);
    int tmp = (NUM_BLOCKS)*(NUM_THREADS);
    printf("\nshould be %d\n", 
        1 - tmp + tmp * (tmp + 1) / 2);
    
    free( inputAtoms );
    cudaFree( d_inputAtoms );
    
    
    //--------------------------------------------------------------------------
    // practice lookup table prototype
    
    // allocate variable on the GPU and CPU
    int *lookup, *d_lookup;
    sizeArray = sizeof(float) * LENGTH_LOOKUP;
    
    lookup = (int*) malloc ( sizeArray );
    cudaMalloc( &d_lookup, sizeArray );
    cudaCheckErrors("cuda malloc fail");
    
    // copy local variable to GPU
    for ( i = 0; i < LENGTH_LOOKUP; ++i ) {
        lookup[i] = 0;
        printf("%d ", (int)inputAtoms[i]);
    } printf("\n");
    cudaMemcpy( d_lookup, lookup, sizeArray, cudaMemcpyHostToDevice );
    cudaCheckErrors("cudaMemcpy 1 fail");
    
    
    // run atomicAdd lookup table test kernel
    atomicAddLookupTableKernel<<<1, LENGTH_LOOKUP-1>>>(d_lookup);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel fail");
    
    
    // copy back result to local memory
    cudaMemcpy( lookup, d_lookup, sizeArray, cudaMemcpyDeviceToHost );
    cudaCheckErrors("cudaMemcpy 2 fail");
    
    // report results and close
    for ( i = 0; i < LENGTH_LOOKUP; ++i )
        printf("%d ", lookup[i]);
    printf("\n");
    
    free( lookup );
    cudaFree( d_lookup );
    
    
    return 0;
}


