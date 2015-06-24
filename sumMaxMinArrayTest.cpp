// from the following URL
// https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks

/* 
 * this program is a simple test of the binary tree recommended reduction
 * algorithm for CUDA
 * 
 */


#include <iostream>
#define TOTAL_SIZE 100000
#define nTPB 256

#define BLOCK_SIZE 64  // MUST BE A POWER OF 2!!
//#define NUM_THREADS 1 // this is now "BLOCK_SIZE"
#define NUM_BLOCKS 1

#define LENGTH_LOOKUP  240


__global__ void sumMaxMinKernel( float* inputAtoms, float* answers ) {
    
    // Reduction (min/max/avr/sum), valid only when blockDim.x is a power of two:
    int  thread2;
    float temp;
    __shared__ float min[BLOCK_SIZE], max[BLOCK_SIZE], avg[BLOCK_SIZE], sum[BLOCK_SIZE];
    
//    printf("inputting %f   %d\n", inputAtoms[threadIdx.x], threadIdx.x);
    
    // import info
    min[threadIdx.x] = inputAtoms[threadIdx.x];
    max[threadIdx.x] = inputAtoms[threadIdx.x];
    avg[threadIdx.x] = inputAtoms[threadIdx.x];
    sum[threadIdx.x] = inputAtoms[threadIdx.x];
    __syncthreads();
    
    int nTotalThreads = blockDim.x;	// Total number of active threads
    
    while (nTotalThreads > 1) {
        
        int halfPoint = (nTotalThreads >> 1);	// divide by two
        // only the first half of the threads will be active.
        
        if (threadIdx.x < halfPoint) {
            
//            printf("thread pair  %d  %d\n", threadIdx.x, threadIdx.x + halfPoint);
            
            thread2 = threadIdx.x + halfPoint;
            
            // Get the shared value stored by another thread
            
            // min reduction
            temp = min[thread2];
            if (temp < min[threadIdx.x]) 
                min[threadIdx.x] = temp; 
            
            // max reduction
            temp = max[thread2];
            if (temp > max[threadIdx.x]) 
                max[threadIdx.x] = temp;			
            
            // sum reduction
//            printf("sum %f   %f\n", sum[threadIdx.x], sum[thread2]);
            sum[threadIdx.x] += sum[thread2];
            
            // average reduction
            avg[threadIdx.x] += avg[thread2];
            avg[threadIdx.x] *= 0.5f;
        }
        __syncthreads();
        
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }
    
    // export results
    if (threadIdx.x == 0) {
        answers[0] = min[0];
        answers[1] = max[0];
        answers[2] = avg[0];
        answers[3] = sum[0];
    }
}

int main(){
    
    //--------------------------------------------------------------------------
    // practice code (array)
    
    // allocate variable on the GPU and CPU
    int i;
    float *inputAtoms, *d_inputAtoms, *answers, *d_answers;
    size_t sizeArray;
    
    sizeArray = sizeof(float) * BLOCK_SIZE;
    inputAtoms = (float*) malloc ( sizeArray );
    cudaMalloc( &d_inputAtoms, sizeArray );
    
    sizeArray = sizeof(float) * 4;
    answers = (float*) malloc ( sizeArray );
    cudaMalloc( &d_answers, sizeArray );
    
    // copy local variable to GPU
    for ( i = 0; i < BLOCK_SIZE; ++i ) {
        inputAtoms[i] = 1.0f + (float)i;
        printf("%f ", inputAtoms[i]);
    } printf("\n\n");
    cudaMemcpy( d_inputAtoms, inputAtoms, sizeof(float) * BLOCK_SIZE, cudaMemcpyHostToDevice );
    
    // run atomicAdd kernel
    sumMaxMinKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_inputAtoms, d_answers);
    cudaDeviceSynchronize();
    
    // copy back result to local memory
    cudaMemcpy( answers, d_answers, sizeof(float) * 4, cudaMemcpyDeviceToHost );
    
    // report results and close
    printf("min %f\n", answers[0]);
    printf("max %f\n", answers[1]);
    printf("average %f\n", answers[2]);
    printf("sum %f\n", answers[3]);
    
    free( inputAtoms );
    cudaFree( d_inputAtoms );
    free( answers );
    cudaFree( d_answers );
    
    return 0;
}








