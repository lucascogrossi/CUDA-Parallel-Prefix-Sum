#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    if (i < N) {
        inBuffer_s[threadIdx.x] = input[i];
    } else {
        inBuffer_s[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];
    }
    output[i] = inBuffer_s[threadIdx.x];
}
