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
