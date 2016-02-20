extern "C" __global__ void GAKernel(unsigned char *g_ucGenomes,
                                    float *g_fFFValues,
                                    unsigned char *g_ucAssembledGrids,
                                    curandState *g_xCurandStates){
    __shared__ unsigned char s_ucGenome[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome];    
    //__shared__ float s_fFitnessFunctionValues[m_ga_NR_THREADS_PER_BLOCK];
    //i.e. exit thread execution immediately if thread is a surplus one
    if ( m_ga_NR_THREADS_PER_BLOCK * mBLOCK_ID + mTHREAD_ID >= mNrGenomes) return ;

    /*Note: Initialisation could be done better by accessing global memory coalescently*/
    for(int i=0;i<mByteLengthGenome;i++){ //Initialisation: Load Genome from global memory
        s_ucGenome[mTHREAD_ID][i] = g_ucGenomes[ga_xGlobalThreadAnchor(mByteLengthGenome)];
    }
    //Initialisation: Load Fitness Function Value from global memory
    //s_fFitnessFunctionValues[mTHREAD_ID] = g_fFFValues[ga_xGlobalThreadAnchor(1)];
    
    __syncthreads();
    unsigned int r_uiCutoffIndex = ga_uiSelectionRouletteWheel(&g_xCurandStates[mTHREAD_ID]);

    #ifdef WITH_MIXED_CROSSOVER //I.e. if we have a mixture of single-point and uniform crossover
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
            ga_CrossoverSinglePoint(s_ucGenome, g_ucGenomes, r_uiCutoffIndex, &g_xCurandStates[mTHREAD_ID]);    
            #endif
        #endif
    #endif

    #ifdef WITH_SUREFIRE_MUTATION
    ga_MutationSurefire(s_ucGenome, m_ga_RateMutation_c, &g_xCurandStates[mTHREAD_ID]);    
    #else
    {
        __shared__ unsigned char s_ucBufGenome[m_ga_NR_THREADS_PER_BLOCK][mByteLengthGenome];
        ga_MutationSophisticated(s_ucGenome, s_ucBufGenome, m_ga_RateMutation_c, &g_xCurandStates[mTHREAD_ID]);    
    }
    #endif

    __syncthreads();    
    for(int i=0;i<mByteLengthGenome;i++){ //Copy genome back to global memory
        g_ucGenomes[ga_xGlobalThreadAnchor(mByteLengthGenome)+i] = s_ucGenome[ga_xSharedThreadAnchor(mByteLengthGenome)][i];
    }
    return;
}
