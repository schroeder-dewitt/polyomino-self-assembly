extern "C" {
	#include <stdio.h>
}

/* -----------------------------------------------------------------------------
                Declare Textures and Constant Memory
----------------------------------------------------------------------------- */

texture<unsigned char, 2> t_ucFitnessFunctionGrids0;

texture<float, 2> t_ucInteractionMatrix; 

__constant__ float c_fParams[32];
__constant__ float c_fFitnessParams[48];
__constant__ float c_ucFourPermutations[24][4];
__constant__ float c_fFitnessSumConst;
__constant__ float c_fFitnessListConst[512];
__constant__ float c_fGAParams[52];

/* -----------------------------------------------------------------------------
                Include All Header Files
----------------------------------------------------------------------------- */
#include "curand_kernel.h"

extern "C"
{

//start include globals.inc.cuh
typedef int xMutex;
typedef unsigned char ucTYPELENGTH; 
typedef unsigned short usSHARED1D;
typedef unsigned int usGLOBAL1D;

enum Params {
    eNrGenomes,
    eNrGenerations,
    eNrTileTypes,
    eNrEdgeTypes,
    eByteLengthGenome,
    eBitLengthGenome,
    eEdgeTypeBitLength,
    eNrTileOrientations
};

#define mXOR(a, b) (((a)&~(b))|(~(a)&(b)))

#define mBLOCK_ID blockIdx.x
#define mTHREAD_ID_X threadIdx.x
#define mTHREAD_ID_Y threadIdx.y
#define mTHREAD_ID threadIdx.x
#define mNrMemoryBanks warpSize

#define mByteLengthGenome_c c_fParams[eByteLengthGenome]
#define mNrGenomes_c c_fParams[eNrGenomes]
#define mNrGenomes 512
#define mBitLengthGenome_c c_fParams[eBitLengthGenome]
#define mBitLengthEdgeType_c c_fParams[eEdgeTypeBitLength]
#define mNrTileTypes_c c_fParams[eNrTileTypes]
#define mNrEdgeTypes_c c_fParams[eNrEdgeTypes]
#define mNrTileOrientations_c c_fParams[eNrTileOrientations]
                
#define mNrTileTypes 4
#define mNrEdgeTypes 8
#define mNrTileOrientations 4
#define mByteLengthGenome 4
#define mBitLengthGenome 32
#define mBitLengthEdgeType 3

#define EMPTY_TILETYPE 63
//end include globals.inc.cuh

//start include curandinit.inc.cuh
#define m_curand_NR_THREADS_PER_BLOCK 256.0

/*extern "C" __global__ void CurandInitKernel(curandState *state);*/
//end include curandinit.inc.cuh

//start include sorting.inc.cuh
#define m_sorting_NR_THREADS_PER_BLOCK 256

/*__global__ void SortingKernel(float *g_fFFValues);*/
//end include sorting.inc.cuh

//start include ga_utils.inc.cuh

/*TESTED*/
__forceinline__ __device__ unsigned int ga_uiPoissonDistribution(float r_fMean, curandState *g_xCurandState){
    #ifdef EFFICIENT_POISSON
    /* Implement some fast algorithm (see logbook for one based on rejection) and / or use cumulative probability table
       loaded from constant memory
    */
    #else
    //Knuth implementation
    float L = expf(-r_fMean); 
    float p = 1.0f; 
    int k = 0; 
    
    do { 
        k++; 
        p *= curand_uniform(g_xCurandState); 
    } while (p > L); 
    return k - 1;
    #endif
} 
//end include ga_utils.inc.cuh

//start include ga.inc.cuh
#define m_ga_ProbabilityUniformCrossover_c c_fGAParams[e_ga_UniformCrossoverProbability]
#define m_ga_ProbabilitySinglePointCrossover_c c_fGAParams[e_ga_SinglePointCrossoverProbability]
#define m_ga_RateMutation_c c_fGAParams[e_ga_RateMutation]
#define m_ga_NR_THREADS_PER_BLOCK 256
#define m_ga_THREAD_DIM_X 256
#define m_ga_THREAD_DIM_Y 1

enum GAParams {
    e_ga_RateMutation,
    e_ga_ProbabilityUniformCrossover,
    e_ga_ProbabilitySinglePointCrossover,
    e_ga_FlagMixedCrossover
};


//extern "C" __global__ void GAKernel(unsigned char *g_ucGenomes, float *g_fFFValues, unsigned char *g_ucAssembledGrids, curandState *g_xCurandStates);



#define WITH_BANK_CONFLICT


#define WITH_NAIVE_ROULETTE_WHEEL_SELECTION



#define WITH_ASSUME_NORMALIZED_FITNESS_FUNCTION_VALUES



#define WITH_SINGLE_POINT_CROSSOVER


#define WITH_SUREFIRE_MUTATION


//TESTED
__forceinline__ __device__ unsigned int ga_uiSelectionRouletteWheel(curandState *state){
    unsigned int r_uiCutoffIndex = 0;
    float r_fPartSum = 0.0f;
    float r_fRandomNumber = curand_uniform(state);
    for(int i=0;i<mNrGenomes;i++){ //Perform Roulette Wheel Selection on Constant Memory
        r_fPartSum += c_fFitnessListConst[i] / c_fFitnessSumConst;
        if(r_fPartSum >= r_fRandomNumber){
            r_uiCutoffIndex = i;
            break;
        }
    }    
    return r_uiCutoffIndex;
}

//TESTED
__forceinline__ __device__ void ga_CrossoverUniform( unsigned char (&s_ucGenome)[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome],
                                                     unsigned char *g_ucGenomes,
                                                     unsigned int r_uiCutoffIndex,
                                                     curandState *state ){
    unsigned int r_uiNumberOfRandCallsRequired = (unsigned int) ( (mByteLengthGenome - mByteLengthGenome % sizeof(float))/sizeof(float) + 1 ); //Establish number of rand calls required
    float r_fRandBuffer;
    unsigned char r_ucRandMask;
    unsigned int r_uiIndexBuffer;
    for(int i=0;i<r_uiNumberOfRandCallsRequired;i++){
        r_fRandBuffer = curand_uniform(state); 
        
        for(int j=0;j<sizeof(float);j++){      
            r_ucRandMask = (unsigned int) ((reinterpret_cast<int&>(r_fRandBuffer) & (0xFF << 8 * j)) >> 8 * j); //Select next byte from r_fRandBuffer
            r_uiIndexBuffer = i*sizeof(float) + j;
            if(r_uiIndexBuffer < mByteLengthGenome){ 
                /*NOTE: If we decide to pad global memory or so, then we have to adjust these functions here!*/
                s_ucGenome[mTHREAD_ID][r_uiIndexBuffer] = (s_ucGenome[mTHREAD_ID][r_uiIndexBuffer] & r_ucRandMask) + (g_ucGenomes[r_uiCutoffIndex * mByteLengthGenome + r_uiIndexBuffer] & (~r_ucRandMask)) ;
            }    
        }          
        //printf("\n");
    }
}

//TESTED
__forceinline__ __device__ void ga_CrossoverSinglePoint( unsigned char (&s_ucGenome)[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome],
                                                         unsigned char *g_ucGenomes,
                                                         unsigned int r_uiCutoffIndex,
                                                         curandState *state){
    unsigned short int r_uiCrossoverPoint = curand_uniform(state) * mBitLengthGenome;
    unsigned short int r_uiCrossoverBitOffset = r_uiCrossoverPoint % 8;
    unsigned short int r_uiCrossoverByte = (r_uiCrossoverPoint - r_uiCrossoverBitOffset) / 8;
    unsigned int r_uiIndexBuffer;
    for(int j=0;j<=r_uiCrossoverByte;j++){ 
        if(j == r_uiCrossoverByte){     
            s_ucGenome[mTHREAD_ID][j] = ( s_ucGenome[mTHREAD_ID][j] & ( 0xFF >> r_uiCrossoverBitOffset ) ) + ( g_ucGenomes[r_uiCutoffIndex * mByteLengthGenome + j] & ( 0xFF << ( 8 - r_uiCrossoverBitOffset )  ) );         
        } else {
            s_ucGenome[mTHREAD_ID][j] = g_ucGenomes[r_uiCutoffIndex * mByteLengthGenome + j];
        }
    }          
    //printf("SinglePoint: %d (Byte: %d, Offset: %d)\n", r_uiCrossoverPoint, (r_uiCrossoverPoint - r_uiCrossoverBitOffset)/8 , r_uiCrossoverBitOffset);  
}


//TESTED
__forceinline__ __device__ void ga_MutationSurefire( unsigned char (&s_ucGenome)[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome],
                                                      float mutation_rate, 
                                                      curandState *state){
    float r_fRandBuf;
    for(int i=0;i<mByteLengthGenome;i++){
        for(int j=0;j<8;j++){
            r_fRandBuf = curand_uniform(state) * mBitLengthGenome; 
            if(r_fRandBuf <= mutation_rate){
                s_ucGenome[mTHREAD_ID][i] = mXOR(s_ucGenome[mTHREAD_ID][i], 1 << j);
            }
        }
    }
}


//TESTED
__forceinline__ __device__ void ga_MutationSophisticated( unsigned char (&s_ucGenome)[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome],
                                                     unsigned char (&s_ucBufGenome)[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome],
                                                     float mutation_rate, 
                                                     curandState *state){
    /* Note: this can be done more efficiently by just using 
       a dynamic list of mutated indices, however this only works for  > 2.0
       Of course, could try to work around this - however, actually probably performant enough!
    */ 
    for(int i=0;i<mByteLengthGenome;i++){
        s_ucBufGenome[mTHREAD_ID][i] = 0;
    }
    float r_fNrMutations = ga_uiPoissonDistribution(mutation_rate, state);
    if(r_fNrMutations > mBitLengthGenome) r_fNrMutations = mBitLengthGenome;
    float r_fRandBuf;
    short r_ssBitOffset, r_ssByteOffset;
    bool r_bRetry = false;
    for(int i=0;i<r_fNrMutations;i++){
        r_fRandBuf = curand_uniform(state) * mBitLengthGenome;
        r_ssBitOffset =  (signed short) r_fRandBuf % 8 ;
        r_ssByteOffset = ((signed short) r_fRandBuf - r_ssBitOffset) / 8;
        if( ! ( s_ucBufGenome[mTHREAD_ID][r_ssByteOffset] & (1 << r_ssBitOffset) ) ){ //bit set already?
            s_ucBufGenome[mTHREAD_ID][r_ssByteOffset] += (1 << r_ssBitOffset); //Set bit!
            r_bRetry=false;
        } else {
            i--;//Try again to mutate!
        }
    }
    for(int i=0;i<mByteLengthGenome;i++){
        s_ucGenome[mTHREAD_ID][i] = mXOR(s_ucGenome[mTHREAD_ID][i], s_ucBufGenome[mTHREAD_ID][i]);
    }
}
//end include ga.inc.cuh

/* -----------------------------------------------------------------------------
                Define Kernels
----------------------------------------------------------------------------- */
    
    //start include globals.inc.cu
__forceinline__ __device__ void u_Lock(int &mutex){
		while ( atomicCAS( &mutex, 0, 1) != 0);		
}

__forceinline__ __device__ void u_Unlock(int &mutex){
	atomicExch(&mutex, 0);
}
    //end include globals.inc.cu

    //start include curandinit.inc.cu
extern "C" __global__ void CurandInitKernel(curandState *state)
{
    int id = m_curand_NR_THREADS_PER_BLOCK * mBLOCK_ID + mTHREAD_ID;
    //Each thread gets same seed, a different sequence number, no offset 
    curand_init(1234, id, 0, &state[id]);
}
    //end include curandinit.inc.cu

    //start include sorting.inc.cu
__global__ void SortingKernel(float *g_fFFValues, float *g_fFFPartialSums){
    __shared__ float s_fFFValues[m_sorting_NR_THREADS_PER_BLOCK];
    
    unsigned int r_uiGlobalThreadOffset = mBLOCK_ID * m_sorting_NR_THREADS_PER_BLOCK + mTHREAD_ID;
    
    if(r_uiGlobalThreadOffset < mNrGenomes){
        s_fFFValues[mTHREAD_ID] = g_fFFValues[r_uiGlobalThreadOffset];
    } else {
        s_fFFValues[mTHREAD_ID] = 0;
    }
    
    __syncthreads();
        
    int i = m_sorting_NR_THREADS_PER_BLOCK / 2; //Do a reduction of fitness values 
    //REQUIRE: m_sorting_NR_THREADS_PER_BLOCK is power of 2
    while(i != 0){
        if(mTHREAD_ID < i){                
            s_fFFValues[mTHREAD_ID] += s_fFFValues[mTHREAD_ID + i];
        } 
        __syncthreads();
        i /= 2;
    }        
    
    __syncthreads();
    if(mTHREAD_ID == 0){
        g_fFFPartialSums[mBLOCK_ID] =  s_fFFValues[0];
    }

    /*if(mTHREAD_ID == 255){
        g_fFFPartialSums[mBLOCK_ID] = m_sorting_NR_THREADS_PER_BLOCK;
    }*/
    return;
}
    //end include sorting.inc.cu

    //start include ga_utils.inc.cu
/*EMPTY*/
/*TESTED*/
__forceinline__ __device__ usGLOBAL1D ga_xGlobalThreadAnchor(ucTYPELENGTH __typelength){ // This is 1D index 0 of a thread in any block in global memory of type of length 
    return  ( (int) __typelength * (mBLOCK_ID * m_ga_THREAD_DIM_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_Y ) );
}

/*TESTED*/
__forceinline__ __device__ usSHARED1D ga_xSharedThreadAnchor(ucTYPELENGTH __typelength){
    return __typelength*(mTHREAD_ID_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_Y);
}
    //end include ga_utils.inc.cu   

    //start include ga.inc.cu
extern "C" __global__ void GAKernel(unsigned char *g_ucGenomes,
                                    float *g_fFFValues,
                                    unsigned char *g_ucAssembledGrids,
                                    curandState *g_xCurandStates){
    __shared__ unsigned char s_ucGenome[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome];    

    int gen_buf;
    float result=0;

    int debug_code = -1;// threadIdx.x + blockIdx.x* m_ga_NR_THREADS_PER_BLOCK; //Test
    if(debug_code==0) printf("Block CODE: %d \n", m_ga_NR_THREADS_PER_BLOCK * mBLOCK_ID + mTHREAD_ID);

    //i.e. exit thread execution immediately if thread is a surplus one
    if ( m_ga_NR_THREADS_PER_BLOCK * mBLOCK_ID + mTHREAD_ID < mNrGenomes){

    //Initialisation: Load Fitness Function Value from global memory
    
    //__syncthreads();
    unsigned int r_uiCutoffIndex = ga_uiSelectionRouletteWheel(&g_xCurandStates[mTHREAD_ID]); //TEST
    if(debug_code==0) printf("CutoffIndex: %d \n", r_uiCutoffIndex);

    //NOTE: FIXED FOR ASEXUAL REPRODUCTION!
    //Note: Initialisation could be done better by accessing global memory coalescently
    if(debug_code==0) printf("GENOME:\n");
    for(int i=0;i<mByteLengthGenome;i++){ //Initialisation: Load Genome from global memory
        s_ucGenome[mTHREAD_ID][i] = g_ucGenomes[ r_uiCutoffIndex * mByteLengthGenome + i ];
//g_ucGenomes[ blockIdx.x * m_ga_NR_THREADS_PER_BLOCK * (mByteLengthGenome) + threadIdx.x * mByteLengthGenome + i ];//ga_xGlobalThreadAnchor(mByteLengthGenome)];
	if(debug_code==0) printf("[%d]=%d,", i, s_ucGenome[mTHREAD_ID][i]);
        //g_ucGenomes[ r_uiCutoffIndex * mByteLengthGenome + i ];// blockIdx.x * m_ga_NR_THREADS_PER_BLOCK * (mByteLengthGenome) + threadIdx.x * mByteLengthGenome + i ];//ga_xGlobalThreadAnchor(mByteLengthGenome)];
    }

    if(debug_code==0) printf("\n");

    /*#ifdef WITH_MIXED_CROSSOVER //I.e. if we have a mixture of single-point and uniform crossover
    float r_fCrossoverSwitch = curand_uniform(&g_xCurandStates[mTHREAD_ID]);
    if(r_fCrossoverSwitch >= mUniformCrossoverProbability){
        ga_CrossoverUniform(s_ucGenome, g_ucGenomes, r_uiCutoffIndex, &g_xCurandStates[mTHREAD_ID]);    
    } else {
        ga_CrossoverSinglePoint(s_ucGenome, g_ucGenomes, r_uiCutoffIndex, &g_xCurandStates[mTHREAD_ID]);    
    }
    #else
        #ifdef WITH_UNIFORM_CROSSOVER  
            ga_CrossoverUniform(s_ucGenome, g_ucGenomes, r_uiCutoffIndex, &g_xCurandStates[mTHREAD_ID]);    
        #else
            #ifdef WITH_SINGLE_POINT_CROSSOVER
            ga_CrossoverSinglePoint(s_ucGenome, g_ucGenomes, r_uiCutoffIndex, &g_xCurandStates[mTHREAD_ID]);  //TEST
            #endif
        #endif
    #endif*/
    
    
    
    #ifdef WITH_SUREFIRE_MUTATION
    ga_MutationSurefire(s_ucGenome, m_ga_RateMutation_c, &g_xCurandStates[mTHREAD_ID]);    
    #else
    {
        __shared__ unsigned char s_ucBufGenome[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome];
        ga_MutationSophisticated(s_ucGenome, s_ucBufGenome, m_ga_RateMutation_c, &g_xCurandStates[mTHREAD_ID]);    
    }
    #endif
    

    //Evaluate Hamming distance (Fujiama Fitness Function)
    

    
    for(int j=0;j<mByteLengthGenome;j++){ //Copy genome back to global memory
        gen_buf = (int) s_ucGenome[mTHREAD_ID][j];
        gen_buf = gen_buf - ((gen_buf >> 1) & 0x55555555);
        gen_buf = (gen_buf & 0x33333333) + ((gen_buf >> 2) & 0x33333333);
        gen_buf = (((gen_buf + (gen_buf >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        result += gen_buf;
    }
    }
    
    __syncthreads();
    if ( m_ga_NR_THREADS_PER_BLOCK * mBLOCK_ID + mTHREAD_ID < mNrGenomes){

        g_fFFValues[ m_ga_NR_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x] = result;//(float) result; //CRASHES HERE!
    //g_fFFValues[0] = 1; //CRASHES HERE!
    /*
    if(i == 32 ){
        g_fFFValues[512] = 1.0; //set breaking condition
    }*/
    //return;   
    
    }
    __syncthreads();    
    for(int i=0;i<mByteLengthGenome;i++){ //Copy genome back to global memory
        //g_ucGenomes[ga_xGlobalThreadAnchor(mByteLengthGenome)+i] = s_ucGenome[ga_xSharedThreadAnchor(mByteLengthGenome)][i];
        g_ucGenomes[ blockIdx.x * m_ga_NR_THREADS_PER_BLOCK * (mByteLengthGenome) + threadIdx.x * mByteLengthGenome + i] = s_ucGenome[mTHREAD_ID][i];
    }

}
    //end include ga.inc.cu    
}