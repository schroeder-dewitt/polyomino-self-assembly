
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
