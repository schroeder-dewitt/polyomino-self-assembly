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
    if(i == {{ fitness_break_value }} ){
        g_fFFValues[{{ fitness_flag_index }}] = 1.0; //set breaking condition
    }*/
    //return;   
    
    }
    __syncthreads();    
    for(int i=0;i<mByteLengthGenome;i++){ //Copy genome back to global memory
        //g_ucGenomes[ga_xGlobalThreadAnchor(mByteLengthGenome)+i] = s_ucGenome[ga_xSharedThreadAnchor(mByteLengthGenome)][i];
        g_ucGenomes[ blockIdx.x * m_ga_NR_THREADS_PER_BLOCK * (mByteLengthGenome) + threadIdx.x * mByteLengthGenome + i] = s_ucGenome[mTHREAD_ID][i];
    }

}
