#define m_ga_ProbabilityUniformCrossover_c c_fGAParams[e_ga_UniformCrossoverProbability]
#define m_ga_ProbabilitySinglePointCrossover_c c_fGAParams[e_ga_SinglePointCrossoverProbability]
#define m_ga_RateMutation_c c_fGAParams[e_ga_RateMutation]
#define m_ga_NR_THREADS_PER_BLOCK {{ ga_nr_threadsperblock }}
#define m_ga_THREAD_DIM_X {{ ga_threaddimx }}
#define m_ga_THREAD_DIM_Y 1

enum GAParams {
    e_ga_RateMutation,
    e_ga_ProbabilityUniformCrossover,
    e_ga_ProbabilitySinglePointCrossover,
    e_ga_FlagMixedCrossover
};


//extern "C" __global__ void GAKernel(unsigned char *g_ucGenomes, float *g_fFFValues, unsigned char *g_ucAssembledGrids, curandState *g_xCurandStates);


{% if with_bank_conflict %}
#define WITH_BANK_CONFLICT
{% endif %}
{% if with_naive_roulette_wheel_selection %}
#define WITH_NAIVE_ROULETTE_WHEEL_SELECTION
{% endif %}
{% if with_mixed_crossover %}
#define WITH_MIXED_CROSSOVER
#define WITH_UNIFORM_CROSSOVER
#define WITH_SINGLE_POINT_CROSSOVER
{% endif %}
{% if with_assume_normalized_fitness_function_values %}
#define WITH_ASSUME_NORMALIZED_FITNESS_FUNCTION_VALUES
{% endif %}
{% if with_uniform_crossover %}
#define WITH_UNIFORM_CROSSOVER
{% endif %}
{% if with_single_point_crossover %}
#define WITH_SINGLE_POINT_CROSSOVER
{% endif %}
{% if with_surefire_mutation %}
#define WITH_SUREFIRE_MUTATION
{% endif %}

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
                s_ucGenome[mTHREAD_ID][r_uiIndexBuffer] = (s_ucGenome[mTHREAD_ID][r_uiIndexBuffer] & r_ucRandMask) + (g_ucGenomes[ r_uiCutoffIndex * mByteLengthGenome + r_uiIndexBuffer ] & (~r_ucRandMask)) ;
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
    for(int j=0;j<=r_uiCrossoverByte;j++){ 
        if(j == r_uiCrossoverByte){     
            s_ucGenome[mTHREAD_ID][j] = ( s_ucGenome[mTHREAD_ID][j] & ( 0xFF >> r_uiCrossoverBitOffset ) ) + ( g_ucGenomes[ r_uiCutoffIndex * mByteLengthGenome + j ] & ( 0xFF << ( 8 - r_uiCrossoverBitOffset )  ) );         
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
    for(int i=0;i<r_fNrMutations;i++){
        r_fRandBuf = curand_uniform(state) * mBitLengthGenome;
        r_ssBitOffset =  (signed short) r_fRandBuf % 8 ;
        r_ssByteOffset = ((signed short) r_fRandBuf - r_ssBitOffset) / 8;
        if( ! ( s_ucBufGenome[mTHREAD_ID][r_ssByteOffset] & (1 << r_ssBitOffset) ) ){ //bit set already?
            s_ucBufGenome[mTHREAD_ID][r_ssByteOffset] += (1 << r_ssBitOffset); //Set bit!
        } else {
            i--;//Try again to mutate!
        }
    }
    for(int i=0;i<mByteLengthGenome;i++){
        s_ucGenome[mTHREAD_ID][i] = mXOR(s_ucGenome[mTHREAD_ID][i], s_ucBufGenome[mTHREAD_ID][i]);
    }
}
