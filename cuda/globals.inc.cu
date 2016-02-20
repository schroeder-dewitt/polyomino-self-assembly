__forceinline__ __device__ void u_Lock(int &mutex){
		while ( atomicCAS( &mutex, 0, 1) != 0);		
}

__forceinline__ __device__ void u_Unlock(int &mutex){
	atomicExch(&mutex, 0);
}
