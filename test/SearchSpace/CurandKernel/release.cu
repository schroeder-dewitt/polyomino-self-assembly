extern "C++"{
#include <curand_kernel.h>
}

#define m_curand_NR_THREADS_PER_BLOCK {{ curand_nr_threads_per_block }}
#define m_curand_DimBlockX {{ curand_dim_block_x }}

extern "C" __global__ void CurandInitKernel(curandState *state)
{
    int id = m_curand_NR_THREADS_PER_BLOCK * (blockIdx.y * m_curand_DimBlockX + blockIdx.x) + threadIdx.x;
    //Each thread gets same seed, a different sequence number, no offset 
    curand_init(1234, id, 0, &state[id]);
    //Solve via offset: http://forums.nvidia.com/index.php?showtopic=185740
    //curand_init((5364<<20)+id, 0, 0, &state[id]); //Might lead to collisions - or not...
}

