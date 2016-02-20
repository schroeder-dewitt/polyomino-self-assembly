/*EMPTY*/
/*TESTED*/
__forceinline__ __device__ usGLOBAL1D ga_xGlobalThreadAnchor(ucTYPELENGTH __typelength){ // This is 1D index 0 of a thread in any block in global memory of type of length 
    return  ( (int) __typelength * (mBLOCK_ID * m_ga_THREAD_DIM_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_Y ) );
}

/*TESTED*/
__forceinline__ __device__ usSHARED1D ga_xSharedThreadAnchor(ucTYPELENGTH __typelength){
    return __typelength*(mTHREAD_ID_X * m_ga_THREAD_DIM_Y + mTHREAD_ID_Y);
}
