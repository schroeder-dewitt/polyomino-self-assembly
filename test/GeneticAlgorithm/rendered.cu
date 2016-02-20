/* -----------------------------------------------------------------------------
                Declare Textures and Constant Memory
----------------------------------------------------------------------------- */


//texture<float, 2> t_ucInteractionMatrix;

__constant__ float c_fParams[28];
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

/*This file contains all template macros for global simulation - Copyright Christian Schroeder, Oxford University 2012*/

//#define SAFE_MEMORY_MAPPING 
#define mAlignedByteLengthGenome 4
#define mWarpSize 32
#define mBankSize 8
//#define mNrTileOrientations 4
//#define mBitLengthEdgeType 
//#define mNrTileTypes 
//#define mNrEdgeTypes 

//#define EMPTY_TILETYPE 63
//end include globals.inc.cuh

//start include curandinit.inc.cuh
#define m_curand_NR_THREADS_PER_BLOCK 256.0

/*extern "C" __global__ void CurandInitKernel(curandState *state);*/
//end include curandinit.inc.cuh

/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX 
#define m_fit_DimThreadY 
#define m_fit_DimBlockX 

#define m_fit_DimGridX 
#define m_fit_DimGridY 

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define m_fit_LengthMovelist 
#define m_fit_NrRedundancyGridDepth 
#define m_fit_NrRedundancyAssemblies 
#define m_fit_TileIndexStartingTile 

#define mEMPTY_CELL 255
#define mEMPTY_CELL_ML 22<<2 //254
#define mEMPTY_CELL_OUT_OF_BOUNDS 253

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

    //start include fitness.cu
/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX 
#define m_fit_DimThreadY 
#define m_fit_DimBlockX 

#define m_fit_DimGridX 
#define m_fit_DimGridY 

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define m_fit_LengthMovelist 
#define m_fit_NrRedundancyGridDepth 
#define m_fit_NrRedundancyAssemblies 
#define m_fit_TileIndexStartingTile 

#define mEMPTY_CELL 255
#define mEMPTY_CELL_ML 22<<2 //254
#define mEMPTY_CELL_OUT_OF_BOUNDS 253
//#include <stdio.h>




struct xThreadInfo {
        ushort4 data;
        
        __device__  xThreadInfo(unsigned short __usThreadIdX, unsigned short __usThreadIdY, unsigned short __usBlockIdX, unsigned short __usBlockIdY);
        __device__ unsigned short WarpId(void);
        __device__ unsigned short BankId(void);
        __device__ unsigned short FlatThreadId(void);
        __device__ unsigned short FlatBlockId(void);
        __device__ unsigned short GlobId(unsigned short __usTypeLength);
        __device__ void __DEBUG_CALL(void);
};

__device__ xThreadInfo::xThreadInfo(unsigned short __usThreadIdX, unsigned short __usThreadIdY, unsigned short __usBlockIdX, unsigned short __usBlockIdY) {
        this->data.z = threadIdx.y * m_fit_DimThreadX + threadIdx.x; //Flat Thread ID
        this->data.x = this->data.z % mWarpSize; //BankID
        this->data.y = (this->data.z - this->data.x) / mWarpSize; //WarpID
        this->data.w = blockIdx.y * m_fit_DimBlockX + blockIdx.x; //Flat Block ID
}

__device__ unsigned short xThreadInfo::WarpId(void) {
        return this->data.y;
}
__device__ unsigned short xThreadInfo::BankId(void) {
        return this->data.x;
}
__device__ unsigned short xThreadInfo::FlatThreadId(void) {
        return this->data.z;
}
__device__ unsigned short xThreadInfo::FlatBlockId(void) {
        return this->data.w;
}

__device__ unsigned short xThreadInfo::GlobId(unsigned short __usTypeLength) {
        return (this->data.w * m_fit_DimThreadX * m_fit_DimThreadY + this->data.z) * __usTypeLength;
}

struct xGenome {
        union{
            unsigned char one_d[mAlignedByteLengthGenome];
        } data;

        __device__ void CopyFromGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomeSet);
	__device__ void CopyToGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomeSet);
        __device__ unsigned char get_xEdgeType(unsigned char __ucTileId, unsigned char __ucEdgeId);
        __device__ void set_EdgeType(xThreadInfo *__xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId, unsigned char __ucVal);
};

struct xGenomeSet {
        union{        
            xGenome multi_d[mWarpSize];        
            unsigned char one_d[mWarpSize*sizeof(xGenome)];
        } data;

        __device__ void CopyFromGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomeSet);
        __device__ void CopyToGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomeSet);
        __device__ unsigned char get_xEdgeType(xThreadInfo __xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId);
        __device__ unsigned char set_EdgeType(xThreadInfo *__xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId, unsigned char __ucVal);
        __device__ void print(xThreadInfo *__xThreadInfo);
};

__device__ void xGenomeSet::CopyFromGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomes) {
        this->data.multi_d[__xThreadInfo.BankId()].CopyFromGlobal(__xThreadInfo, __g_ucGenomes);
}

__device__ void xGenome::CopyFromGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomes) {
        for (int i = 0; i < mAlignedByteLengthGenome; i ++) {
             this->data.one_d[i] = __g_ucGenomes[__xThreadInfo.GlobId(sizeof(xGenome)) + i];
        }
}

__device__ void xGenome::CopyToGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomes) {
        for (int i = 0; i < mAlignedByteLengthGenome; i += 1) {
                //(*reinterpret_cast<int*> (&this->data.one_d[i])) = (*reinterpret_cast<int*> (&__g_ucGenomeSet[__xThreadInfo->GlobId(sizeof(xGenome)) + i]));
                __g_ucGenomes[__xThreadInfo.GlobId(sizeof(xGenome)) + i] = this->data.one_d[i];
        }
}

__device__ void xGenomeSet::CopyToGlobal(xThreadInfo __xThreadInfo, unsigned char *__g_ucGenomes) {
        this->data.multi_d[__xThreadInfo.BankId()].CopyToGlobal(__xThreadInfo, __g_ucGenomes);
}

__device__ unsigned char xGenome::get_xEdgeType( unsigned char __ucTileId, unsigned char __ucEdgeId){
    if (__ucTileId < mNrTileTypes) {
        unsigned short TmpStartBit = __ucTileId * mBitLengthEdgeType * mNrTileOrientations + __ucEdgeId * mBitLengthEdgeType;
	unsigned short TmpEndBit = TmpStartBit + mBitLengthEdgeType;
	unsigned char TmpRetVal = 0;
	unsigned short TmpByteOffset = 0;
	unsigned short TmpBitOffset = 0;
	//Note: This could be speeded up by copying all bits within a byte simultaneously
	unsigned short j = 0;
	for (int i = TmpStartBit; i < TmpEndBit; i++) {
	    TmpBitOffset = i % 8; //We need to invert index as we start from left to right
	    TmpByteOffset = (i - TmpBitOffset) / 8;
	    TmpBitOffset = 7 - TmpBitOffset;
	    TmpRetVal += mOneOrNil(mBitTest(TmpBitOffset, this->data.one_d[TmpByteOffset])) << (mBitLengthEdgeType - 1 - j);
	    j++;
            //return mBitTest(TmpBitOffset,8);
	}
        return TmpRetVal;
    } else return (unsigned char) 0x00;
}

__device__ unsigned char xGenomeSet::get_xEdgeType(xThreadInfo __xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId){
    return this->data.multi_d[__xThreadInfo.BankId()].get_xEdgeType(__ucTileId, __ucEdgeId);
}

//GENOME TEST KERNEL
__global__ void TestGenomeKernel(unsigned char *dest)
{
    __shared__ xGenomeSet Tmp;
    xThreadInfo Tmpa(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    Tmp.CopyFromGlobal(Tmpa, dest);   
    for(int i=0;i<4;i++){
        Tmp.data.multi_d[Tmpa.BankId()].data.one_d[i] = Tmp.get_xEdgeType(Tmpa, i, 1);
    }
    Tmp.CopyToGlobal(Tmpa, dest);
}



texture<float, 2> t_ucInteractionMatrix;

struct xAssemblyFlags {
	unsigned char bitset;
	unsigned char bitset2;
	unsigned char red;
	unsigned char fullcheckcutoff;

	__device__ void set_Red(unsigned char __ucVal);
	__device__ void set_TrivialUND(void);
	__device__ void set_UnboundUND(void);
	__device__ void set_StericUND(void);
	__device__ void set_BusyFlag(void);
	__device__ bool get_bTrivialUND(void);
	__device__ bool get_bUnboundUND(void);
	__device__ bool get_bStericUND(void);
	__device__ bool get_bBusyFlag(void);
	__device__ bool get_bUNDCondition(void);
	__device__ unsigned char get_ucRed(void);
	__device__ void ClearAll(void);
	__device__ void ClearBitsets(void);
};

__device__ void xAssemblyFlags::set_Red(unsigned char __ucVal) {
    this->red = __ucVal;
}

__device__ void xAssemblyFlags::set_TrivialUND(void) {
    this->bitset |= (1 << 4);  //TEST
}

__device__ void xAssemblyFlags::set_UnboundUND(void) {
    this->bitset |= (1 << 5); //TEST
}

__device__ void xAssemblyFlags::set_StericUND(void) {
    this->bitset |= (1 << 6); //TEST
}

__device__ void xAssemblyFlags::set_BusyFlag(void) {
    this->bitset |= (1 << 7);
}

__device__ bool xAssemblyFlags::get_bTrivialUND(void) {
    return (bool) (this->bitset & (1 << 4));
}

__device__ bool xAssemblyFlags::get_bUnboundUND(void) {
    return (bool) (this->bitset & (1 << 5));
}

__device__ bool xAssemblyFlags::get_bStericUND(void) {
    return (bool) (this->bitset & (1 << 6));
}

__device__ bool xAssemblyFlags::get_bBusyFlag(void) {
    return (bool) (this->bitset & (1 << 7));
}

__device__ bool xAssemblyFlags::get_bUNDCondition(void) {
    return (bool) (this->bitset & 120);
}

__device__ unsigned char xAssemblyFlags::get_ucRed(void) {
    return this->red;
}

__device__ void xAssemblyFlags::ClearAll(void) {
    this->bitset = 0;
    this->bitset2 = 0;
    this->red=0;
    this->fullcheckcutoff=0;
    return;
}

__device__ void xAssemblyFlags::ClearBitsets(void) {
    this->bitset = 0;
    this->bitset2 = 0;
    return;
}


struct xEdgeSort {
	union {
		unsigned char multi_d[mNrEdgeTypes][mNrTileTypes][mNrTileOrientations][mWarpSize];
                unsigned char mix_d[mNrEdgeTypes*mNrTileTypes*mNrTileOrientations][mWarpSize];
		unsigned char one_d[mNrEdgeTypes * mNrTileTypes * mNrTileOrientations * mWarpSize];
	} data;

	union {
		unsigned short multi_d[mNrEdgeTypes][mWarpSize];
		unsigned char one_d[mNrEdgeTypes * mWarpSize * sizeof(short)];
	} length;

	__device__ void Zeroise(xThreadInfo __xThreadInfo);
        __device__ void Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet, short __sEdgeId = -1);
        __device__ unsigned char GetBondingTile(xThreadInfo __xThreadInfo, short __sEdgeId, curandState *__xCurandState, xAssemblyFlags *__xAssemblyFlags);
        __device__ void add_TileOrient(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucOrient, unsigned char __ucTileType);
	__device__ __forceinline__ void set_xLength(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucLength);
        __device__ void add_Tile(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId);
        __device__ unsigned char get_xData(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucTileId, unsigned char __ucOrientation);
        __device__ unsigned char GetBondingTileOrientation(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucTileId, xAssemblyFlags *__xAssemblyFlags);
        __device__ short get_xLength(xThreadInfo __xThreadInfo, unsigned short __sEdgeId);
};

#define mDelta(a,b) ((a==b)?1:0)
__device__ __forceinline__ int InteractionMatrix(int i, int j){
	return (1-i%2)*mDelta(i,j+1)+(i%2)*mDelta(i,j-1);
}

__device__ void xEdgeSort::Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet, short __sEdgeId) {
          
        //DEBUG
        /*for(int i=0; i< mNrEdgeTypes*mNrTileTypes*mNrTileOrientations;i++){
           
        }*/
        //DEBUG

	//if (__sEdgeId == -1) {
/*		for (int k = 0; k < mNrEdgeTypes; k++) {
                //if(__xThreadInfo.WarpId() < mNrEdgeTypes){
                //int k = __xThreadInfo.WarpId();
			this->set_xLength(__xThreadInfo, k, 0);
			bool r_bEdgeAdded = false;
			for (int i = 0; i < mNrTileTypes; i++) { //Traverse all TileTypes
				for (int j = 0; j < mNrTileOrientations; j++) { //Traverse all Orientations
					//if (tex2D(t_ucInteractionMatrix,
				        //		__xGenomeSet->get_xEdgeType(__xThreadInfo, i, j), k)
					//		> 0) { //Does Edge j of Tile i bond to Tile ThreadID
                                        if(InteractionMatrix(__xGenomeSet->get_xEdgeType(__xThreadInfo, i, j), k)){
						this->add_TileOrient(__xThreadInfo, k, j, i);
						r_bEdgeAdded = true;
					} else {
						this->add_TileOrient(__xThreadInfo, k, j, mEMPTY_CELL);
					}
				}
				if (r_bEdgeAdded) { //EdgeAdded?
					this->add_Tile(__xThreadInfo, k); //TEST
					r_bEdgeAdded = 0;
				} else {
					//Do Nothing
				}
			}
                       //set_xLength(__xThreadInfo, 0, 5); //Test: 
		}
*/
               if(__xThreadInfo.WarpId()==0){
               //if(threadIdx.x==0){

               for (int k = 0; k < mNrEdgeTypes; k++) {
			this->set_xLength(__xThreadInfo, k, 0);
			bool r_bEdgeAdded = false;
			for (int i = 0; i < mNrTileTypes; i++) { /*Traverse all TileTypes*/
				for (int j = 0; j < mNrTileOrientations; j++) { /*Traverse all Orientations*/
               //                         printf("%d|", __xGenomeSet->get_xEdgeType(__xThreadInfo, i, j));
					if (tex2D(t_ucInteractionMatrix,
							__xGenomeSet->get_xEdgeType(__xThreadInfo, i, j), k)
							> 0) { /*Does Edge j of Tile i bond to Tile ThreadID*/
						this->add_TileOrient(__xThreadInfo, k, j, i);
						r_bEdgeAdded = true;
					} else {
						this->add_TileOrient(__xThreadInfo, k, j, mEMPTY_CELL);
					}
				}
				if (r_bEdgeAdded) { /*EdgeAdded?*/
					this->add_Tile(__xThreadInfo, k); //TEST
					r_bEdgeAdded = 0;
				} else {
					/*Do Nothing*/
				}
                //                printf ("\n");
			}
		}
        //        printf("Next one...")
                }

	/*} else {
		this->set_xLength(__xThreadInfo, __sEdgeId, 0);
		bool r_bEdgeAdded = false;
		for (int i = 0; i < mNrTileTypes; i++) { //Traverse all TileTypes
			for (int j = 0; j < mNrTileOrientations; j++) {//Traverse all Orientations
				if (tex2D(t_ucInteractionMatrix, __xGenomeSet->get_xEdgeType(
						__xThreadInfo, i, j), __sEdgeId) > 0) { //Does Edge j of Tile i bond to Tile ThreadID
                                //if(InteractionMatrix(__xGenomeSet->get_xEdgeType(__xThreadInfo, i, j), k)){
					this->add_TileOrient(__xThreadInfo, __sEdgeId, j, i);
					r_bEdgeAdded = true;
				} else {
					this->add_TileOrient(__xThreadInfo, __sEdgeId, j,
							mEMPTY_CELL);
				}
			}
			if (r_bEdgeAdded) { //EdgeAdded?
				this->add_Tile(__xThreadInfo, __sEdgeId);
				r_bEdgeAdded = 0;
			} else {
				//Do Nothing
			}
		}
	}*/
}

__device__ __forceinline__ void xEdgeSort::set_xLength(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucLength) {
	this->length.multi_d[__ucEdgeId][__xThreadInfo.BankId()] = __ucLength;
}

__device__ void xEdgeSort::add_TileOrient(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucOrient, unsigned char __ucTileType) {
	this->data.multi_d[__ucEdgeId][this->get_xLength(__xThreadInfo, __ucEdgeId)][__ucOrient][__xThreadInfo.BankId()] = __ucTileType;
}

__device__ short xEdgeSort::get_xLength(xThreadInfo __xThreadInfo, unsigned short __sEdgeId) {
	if (__sEdgeId < mNrEdgeTypes) {
		return this->length.multi_d[__sEdgeId][__xThreadInfo.BankId()];
	} else {
		return 0;
	}
}

void xEdgeSort::add_Tile(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId) {
	this->set_xLength(__xThreadInfo, __ucEdgeId, this->get_xLength(	__xThreadInfo, __ucEdgeId) + 1);
}

__global__ void TestEdgeSortKernel(unsigned char *dest, curandState *states)
{
    __shared__ xGenomeSet s_xGenomeSet;
    __shared__ xEdgeSort s_xEdgeSort;
    xThreadInfo r_xThreadInfo(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    s_xGenomeSet.CopyFromGlobal(r_xThreadInfo, dest);
    s_xEdgeSort.Initialise(r_xThreadInfo, &s_xGenomeSet, -1); 
    for(int i=0;i<4;i++){
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.length.multi_d[i][r_xThreadInfo.BankId()];
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.data.multi_d[6][0][i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = tex2D(t_ucInteractionMatrix, i, 1);
    }
    s_xGenomeSet.CopyToGlobal(r_xThreadInfo, dest);
}













#include <stdio.h>

#define m_fit_LengthMovelist 20
#define mEMPTY_CELL 255




extern "C++"{

template<class T>
struct xLifoList {
        struct {
                signed short pos; //Current position of top element (-1...max_length-1)
        } data;

        __device__ bool bPush(xThreadInfo __xThreadInfo, T __xEntry, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize], unsigned short __uiMaxLength);
	__device__ T xPop(xThreadInfo __xThreadInfo, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize]);
        __device__ short get_sPos();
        __device__ short set_sPos(short __sPos);
};

template<class T>
__device__ bool xLifoList<T>::bPush(xThreadInfo __xThreadInfo, T __xEntry, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize], unsigned short __uiMaxLength) {
        if (this->data.pos < __uiMaxLength) {
                __xStorage[this->data.pos][__xThreadInfo.BankId()] = __xEntry;
                this->data.pos++;
                return true;
        } else {
                return false;
        }
}

template<class T>
__device__ T xLifoList<T>::xPop(xThreadInfo __xThreadInfo, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize]) {
        if (this->data.pos <= 0) { //FXD
                //NOTE: ADAPTED FOR CUDA VECTORTYPES ONLY!
                T buf;
                buf.x = mEMPTY_CELL;
                buf.y = mEMPTY_CELL;
                return buf;
        } else {
                this->data.pos--;
                return __xStorage[this->data.pos][__xThreadInfo.BankId()];
        }
}

template<class T>
__device__ short xLifoList<T>::get_sPos() {
        return this->data.pos;
}

template<class T>
__device__ short xLifoList<T>::set_sPos(short __sPos){
        this->data.pos = __sPos;
        return 0;
}

template<class T>
struct xMoveList {
        struct {
                T multi_d[m_fit_LengthMovelist][mWarpSize];
        } storage;
        struct {
                xLifoList<T> multi_d[mWarpSize];
        } list;

        __device__ void Initialise(xThreadInfo __xThreadInfo);
        __device__ bool bPush(xThreadInfo __xThreadInfo, T __xEntry);
        __device__ T xPop(xThreadInfo __xThreadInfo);
        __device__ short get_sPos(xThreadInfo __xThreadInfo);
        __device__ short set_sPos(xThreadInfo __xThreadInfo, short __sPos);
};

template<class T>
__device__ bool xMoveList<T>::bPush(xThreadInfo __xThreadInfo, T __xEntry) {
        this->list.multi_d[__xThreadInfo.BankId()].bPush(__xThreadInfo, __xEntry, this->storage.multi_d, );
        return true;
}

template<class T>
__device__ void xMoveList<T>::Initialise(xThreadInfo __xThreadInfo) {
        this->list.multi_d[__xThreadInfo.BankId()].data.pos = 0;
}

template<class T>
__device__ T xMoveList<T>::xPop(xThreadInfo __xThreadInfo) {
	return this->list.multi_d[__xThreadInfo.BankId()].xPop(__xThreadInfo, this->storage.multi_d);
}

template<class T>
__device__ short xMoveList<T>::get_sPos(xThreadInfo __xThreadInfo) {
        return this->list.multi_d[__xThreadInfo.BankId()].get_sPos();
}

template<class T>
__device__ short xMoveList<T>::set_sPos(xThreadInfo __xThreadInfo, short __sPos){
        return this->list.multi_d[__xThreadInfo.BankId()].set_sPos(__sPos);
}

}

//MOVELIST TEST KERNEL
__global__ void TestMovelistKernel(unsigned char *dest)
{
    /*__shared__ xGenomeSet Tmp;
    xThreadInfo Tmpa(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    Tmp.CopyFromGlobal(Tmpa, dest);   
    for(int i=0;i<4;i++){
        Tmp.data.multi_d[Tmpa.BankId()].data.one_d[i] = Tmp.get_xEdgeType(Tmpa, i, 1);
    }
    Tmp.CopyToGlobal(Tmpa, dest);*/

    xThreadInfo __xThreadInfo(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y); 
    xMoveList<uchar2> s_xMovelist;
    s_xMovelist.Initialise(__xThreadInfo);
    s_xMovelist.bPush(__xThreadInfo, make_uchar2(9,9));
    s_xMovelist.bPush(__xThreadInfo, make_uchar2(6,6));
    for(int i=0;i<2;i++){
        //dest[__xThreadInfo.BankId()*2 + i] = s_xMovelist.storage.multi_d[0][__xThreadInfo.BankId()].x;//s_xMovelist.xPop(__xThreadInfo).x;
        //dest[__xThreadInfo.BankId()*(s_xMovelist.get_sPos(__xThreadInfo)+1) + i] = s_xMovelist.xPop(__xThreadInfo).x;
        //dest[__xThreadInfo.BankId()*(s_xMovelist.get_sPos(__xThreadInfo)+1) + i] = ;
    }
}

__constant__ unsigned char c_ucFourPermutations[][];

struct xFourPermutation {
	unsigned short WalkIndex;
        uchar4 Perm;
	__device__ xFourPermutation(unsigned short __usPermIndex);
	__device__ unsigned short ucWalk();
	__device__ bool bNotTraversed();
};

__device__ xFourPermutation::xFourPermutation(unsigned short __usPermIndex) {
    this->WalkIndex = 0;
    switch(__usPermIndex % 24){
        case 1: this->Perm = make_uchar4(1,2,3,4); break;
        case 2: this->Perm = make_uchar4(1,2,4,3); break;
        case 3: this->Perm = make_uchar4(1,3,2,4); break;
        case 4: this->Perm = make_uchar4(1,3,4,2); break;
        case 5: this->Perm = make_uchar4(1,4,2,3); break;
        case 6: this->Perm = make_uchar4(1,4,3,2); break;
        case 7: this->Perm = make_uchar4(2,1,3,4); break;
        case 8: this->Perm = make_uchar4(2,1,4,3); break;
        case 9: this->Perm = make_uchar4(2,3,1,4); break;
        case 10: this->Perm = make_uchar4(2,3,4,1); break;
        case 11: this->Perm = make_uchar4(2,4,1,3); break;
        case 12: this->Perm = make_uchar4(2,4,3,1); break;
        case 13: this->Perm = make_uchar4(3,2,1,4); break;
        case 14: this->Perm = make_uchar4(3,2,4,1); break;
        case 15: this->Perm = make_uchar4(3,1,2,4); break;
        case 16: this->Perm = make_uchar4(3,1,4,2); break;
        case 17: this->Perm = make_uchar4(3,4,2,1); break;
        case 18: this->Perm = make_uchar4(3,4,1,2); break;
        case 19: this->Perm = make_uchar4(4,2,3,1); break;
        case 20: this->Perm = make_uchar4(4,2,1,3); break;
        case 21: this->Perm = make_uchar4(4,3,2,1); break;
        case 22: this->Perm = make_uchar4(4,3,1,2); break;
        case 23: this->Perm = make_uchar4(4,1,2,3); break;
        case 0: this->Perm = make_uchar4(4,1,3,2); break;
    }
}

__device__ unsigned short xFourPermutation::ucWalk() {
    //Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
    this->WalkIndex++;
    if (this->WalkIndex - 1 < mNrTileOrientations) {
        //return this->Perm[];//this->WalkIndex-1; //c_ucFourPermutations[this->PermIndex][this->WalkIndex - 1] - 1; //TEST
        switch(this->WalkIndex-1){
            case 0: return this->Perm.x-1;
            case 1: return this->Perm.y-1;
            case 2: return this->Perm.z-1;
            case 3: return this->Perm.w-1;            
        }
    } else return 0;
}

__device__ bool xFourPermutation::bNotTraversed() {
    //Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
    if (this->WalkIndex >= mNrTileOrientations) {
        return false;
    } else return true;
}

extern "C++"{
template<int Length>
struct xLinearIterator {
	unsigned short WalkIndex;
	__device__ xLinearIterator(unsigned short __usPermIndex);
	__device__ unsigned short ucWalk();
	__device__ bool bNotTraversed();
};

template<int Length>
__device__ xLinearIterator<Length>::xLinearIterator(unsigned short __usPermIndex) {
    //this->WalkIndex = 0;
}

template<int Length>
__device__ unsigned short xLinearIterator<Length>::ucWalk() {
    //Require c_fFourPermutations to be numbers 1-4 (NOT 0-3)
    this->WalkIndex++;
    if (this->WalkIndex - 1 < Length) {
        return this->WalkIndex - 1;
    } else return 0;
}

template<int Length>        
__device__ bool xLinearIterator<Length>::bNotTraversed() {
    //Require c_fFourPermutations to be numbers 1-4 (NOT 0-3)
    if (this->WalkIndex >= Length) {
        return false;
    } else return true;
}

struct xCell {
	unsigned char data;
        __device__ void set_Orient(unsigned char __uiOrient);
        __device__ void set_Type(unsigned char __uiType);
        __device__ unsigned char get_xType(void);
        __device__ unsigned char get_xOrient(void);
        __device__ unsigned char get_xCell(void);
        __device__ void set_xCell(unsigned char __ucVal);
};

__device__ void xCell::set_Orient(unsigned char __uiOrient) {
	__uiOrient = __uiOrient % mNrTileOrientations;
	//unsigned char DBGVAL1 = this->data & (255-3);
	//unsigned char DBGVAL2 = __uiOrient;
	//unsigned char DBGVAL3 = this->data & (255-3) + __uiOrient;
	//I THINK THIS FUNCTION DOES NOT WORK!
	this->data = ((this->data & (255-3) ) + __uiOrient);
}

__device__ void xCell::set_Type(unsigned char __uiType) {
#ifndef __NON_FERMI
	if (__uiType > 63) {
		printf("xCell: TileType exceeded 63 limit!\n");
	}
#endif
	this->data = (this->data & 3) + (__uiType << 2);
}

__device__ void xCell::set_xCell(unsigned char __ucVal) {
	this->data = __ucVal;
}

__device__ unsigned char xCell::get_xType(void) {
	return this->data >> 2;
}

__device__ unsigned char xCell::get_xOrient(void) {
	return (this->data & 3);
}

__device__ unsigned char xCell::get_xCell(void) {
	return this->data;
}

struct xCellGrid {
	union {
		xCell multi_d[m_fit_DimGridX][m_fit_DimGridY][m_fit_NrRedundancyGridDepth][mWarpSize];
		xCell mix_d[m_fit_DimGridX * m_fit_DimGridY][m_fit_NrRedundancyGridDepth][mWarpSize];
		xCell one_d[m_fit_DimGridX * m_fit_DimGridY * mWarpSize	* m_fit_NrRedundancyGridDepth];
	} data;

	__device__ void Initialise(xThreadInfo __xThreadInfo, unsigned char __red);
        __device__ xCell get_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red);
        __device__ bool set_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __val);
        __device__ xCell xGetNeighbourCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __dir);
        __device__ uchar2 xGetNeighbourCellCoords(unsigned char __x, unsigned char __y, unsigned char __dir);
        __device__ bool xCompareRed(xThreadInfo __xThreadInfo, unsigned char __red);
        __device__ void print(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
};

__device__ void xCellGrid::Initialise(xThreadInfo __xThreadInfo,
		unsigned char __red) {
	//Surefire-version:
	/*for (int i = 0; i < m_fit_DimGridX; i++) {
		for (int j = 0; j < m_fit_DimGridY; j++) {
			this->data.multi_d[i][j][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL);
		}
	}*/
        /*for (int i = 0; i < m_fit_DimGridX; i++) {
                for (int j = 0; j < m_fit_DimGridY; j++) {
                        this->data.multi_d[i*j][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL);
                }
        }*/
        short offset = (m_fit_DimGridX*m_fit_DimGridY) % mBankSize;
        short myshare = (m_fit_DimGridX*m_fit_DimGridY - offset) / mBankSize; 

        for(int i=0;i<myshare;i++){
                this->data.mix_d[__xThreadInfo.WarpId()*myshare + i][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL); 
        }
        if(__xThreadInfo.WarpId()==mBankSize-1){
                for(int i=0;i<offset;i++){
                        this->data.mix_d[mBankSize*myshare + i][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL);
                }
        }
}

__device__ xCell xCellGrid::get_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red) {
        if ( (__x <= m_fit_DimGridX-1) && (__x >= 0) && (__y <= m_fit_DimGridY-1) && (__y >= 0) ) { // In grid
            return this->data.multi_d[__x][__y][__red%m_fit_NrRedundancyGridDepth][__xThreadInfo.BankId()];
        } else { // Outside of grid
            xCell TmpCell;
            TmpCell.set_xCell(mEMPTY_CELL_OUT_OF_BOUNDS);
            return TmpCell;
        }
}

__device__ bool xCellGrid::set_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __val) {
        if ( (__x <= m_fit_DimGridX-1) && (__x >= 0) && (__y <= m_fit_DimGridY-1) && (__y >= 0) ) { // In grid
            this->data.multi_d[__x][__y][__red%m_fit_NrRedundancyGridDepth][__xThreadInfo.BankId()].set_xCell(__val);
        } 

        if ( (__x >= m_fit_DimGridX-1) || (__x <= 0) ||  (__y >= m_fit_DimGridY-1) || (__y <= 0) ) { // In grid
            return false;
        } else return true;
}

__device__ xCell xCellGrid::xGetNeighbourCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __dir) {
	uchar2 TmpCoords = xGetNeighbourCellCoords(__x, __y, __dir);
	return this->get_xCell(__xThreadInfo, TmpCoords.x, TmpCoords.y, __red);
}

__device__ uchar2 xCellGrid::xGetNeighbourCellCoords(unsigned char __x, unsigned char __y, unsigned char __dir) {
	switch (__dir) {
        case 0: //NORTH
                return make_uchar2(__x, __y - 1);
                //break;
	case 1: //EAST
		return make_uchar2(__x + 1, __y);
		//break;
        case 2: //SOUTH
                return make_uchar2(__x, __y + 1);
                //break;
	case 3: //WEST
		return make_uchar2(__x - 1, __y);
		//break;
	}
	return make_uchar2(mEMPTY_CELL, mEMPTY_CELL);
}

__device__ bool xCellGrid::xCompareRed(xThreadInfo __xThreadInfo, unsigned char __red) {
        unsigned char TmpNextDir = (__red + 1) % m_fit_NrRedundancyGridDepth;
	unsigned char TmpIsDifferent = 0;
	for (int i = 0; i < m_fit_DimGridX * m_fit_DimGridY; i++) {
		if (this->data.mix_d[i][__red][__xThreadInfo.BankId()].get_xCell() != this->data.mix_d[i][TmpNextDir][__xThreadInfo.BankId()].get_xCell() ) {
		    TmpIsDifferent = 1;
		    break;
		}
	}
	if (!TmpIsDifferent)
		return true;
	else
		return false;
}

struct xFitnessGrid {
	texture<xCell, 2> *grid;
	__device__ unsigned char get_xCell(unsigned char i, unsigned char j);
};

struct xAssembly {
	struct {
		xCellGrid grid;
		xEdgeSort edgesort;
		xMoveList<uchar2> movelist;
		xAssemblyFlags flags[mWarpSize];
		curandState *states;
                int assembly_size[mWarpSize];
                int4 hash[mWarpSize];
                int2 corner_lower[mWarpSize];
                int2 corner_upper[mWarpSize];
	} data;

	__device__ void Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
        __device__ bool Assemble_PreProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_PostProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_Movelist(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_InPlace(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ float fEvaluateFitness(xThreadInfo __xThreadInfo);
	__device__ bool bSynchronizeBank(xThreadInfo __xThreadInfo);
};

__device__ void jenkins_init(int &hash){
    hash = 0;
}

__device__ void jenkins_add(char key, int &hash){
    unsigned int tmphash=hash;
    tmphash += key;
    tmphash += (tmphash << 10);
    tmphash ^= (tmphash >> 6);
    hash=tmphash;
}

__device__ unsigned int jenkins_clean_up(int &hash){
    unsigned int tmphash=hash;
    tmphash += (tmphash << 3);
    tmphash ^= (tmphash >> 11);
    tmphash += (tmphash << 15);
    hash = tmphash;
    return hash;
}


__device__ void xAssembly::Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet) {
	unsigned char TmpRed = this->data.flags[__xThreadInfo.BankId()].get_ucRed() % m_fit_NrRedundancyGridDepth;
	this->data.grid.Initialise(__xThreadInfo, TmpRed);
	this->data.movelist.Initialise(__xThreadInfo);
}

__device__ bool xAssembly::Assemble(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet) {
	bool TmpFlag = false;
	this->data.flags[__xThreadInfo.BankId()].ClearAll();
	TmpFlag = true; 
	if (TmpFlag) {
                if(__xThreadInfo.WarpId() == 0){ //DEBUG
              		this->data.edgesort.Initialise(__xThreadInfo, __xGenomeSet, -1); //TEST
                }
                __syncthreads();
		//this->Assemble_PostProcess(__xThreadInfo, __xGenomeSet);
		if (TmpFlag) {
			//for (int i = 0; (i < m_fit_NrRedundancyAssemblies) && (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()); i++) {
                        for (int i = 0; (i < m_fit_NrRedundancyAssemblies); i++) {
                                //__syncthreads();
                                if(!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()){
					this->Initialise(__xThreadInfo, __xGenomeSet); //Empty out assembly grid at red
                                }
                                __syncthreads();
                                if( (__xThreadInfo.WarpId() == 0) && (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()) ){
                                        this->data.flags[__xThreadInfo.BankId()].ClearAll();
             			        this->Assemble_Movelist(__xThreadInfo, __xGenomeSet); //TEST
                                }
				//this->data.flags[__xThreadInfo.BankId()].set_Red(i); //Choose next assembly step!
                                return true; //DEBUG
			}
			return true; //Always true - i.e. indicate assembly did finish (can still be UND, though)
		} else {
			return false; //Indicates that processing before assembly returned either single block, or UND
		}

	} else {
		return false; //Indicates that processing before assembly returned either single block, or UND
	}

}

__device__ bool xAssembly::Assemble_PreProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet) {
	unsigned char TmpSameCounter = 0;

	//NOTE: This should work, however, not clear how to communicate that single tile without initialisation of grid!
	//Check if starting tile is not empty
	for (int j = 0; j < mNrTileOrientations; j++) {
		if (__xGenomeSet->get_xEdgeType(__xThreadInfo, m_fit_TileIndexStartingTile,
				j) == 0)
			TmpSameCounter++;
	}
	if (TmpSameCounter == 4) {
		this->data.grid.get_xCell(__xThreadInfo, m_fit_DimGridX / 2,
				m_fit_DimGridY / 2, 0);
		return true; //Have finished assembly - UND is false, but so is PreProcess (trigger)
	}

	//Replace tile doublettes by empty tiles
	//Works for any number of mNrTileOrientations and mBitLengthEdgeType <= 4 Byte!
	//Note: This would be faster (but more inflexible) if tile-wise accesses!
	TmpSameCounter = 0;
	unsigned char DBGVAL1, DBGVAL2, DBGVAL3;
	for (int k = 0; k < mNrTileTypes - 1; k++) { //Go through all Tiles X (except for last one)
		for (int i = k + 1; i < mNrTileTypes; i++) { //Go through all Tiles X_r to the right
			for (int j = 0; j < mNrTileOrientations; j++) { //Go through all X edges rots
				TmpSameCounter = 0;
				for (int l = 0; l < mNrTileOrientations; l++) { //Cycle through all X edges
					if (__xGenomeSet->get_xEdgeType(__xThreadInfo, k, l)
							== __xGenomeSet->get_xEdgeType(__xThreadInfo, i, (j
									+ l) % mNrTileOrientations)) {
						TmpSameCounter++;
					}
				}
				if (TmpSameCounter == mNrTileOrientations) {
					//Have detected a doublette - replace with empty tile!!
					for (int l = 0; l < mNrTileOrientations; l++) {
						//__xGenomeSet->set_EdgeType(__xThreadInfo, i, l, 0); //TEST
					}
				}
			}
		}
	}
	return true;
}

__device__ bool xAssembly::Assemble_PostProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet) {
	//Optional: start at first tile and see if it can connect to any degenerate entries in EdgeSort directly
	//Note: If we can refrain from assembly, then save time for grid initialisation!
	unsigned char TmpBondingCounter = 0;
	unsigned char TmpEdgeTypeLength = 0;
	for (int j = 0; j < mNrTileOrientations; j++) {
		TmpEdgeTypeLength = this->data.edgesort.get_xLength(__xThreadInfo, j);
		if (TmpEdgeTypeLength > 1) {
			this->data.flags[__xThreadInfo.BankId()].set_TrivialUND(); //TEST
			return false;
		} else if (TmpEdgeTypeLength == 0) {
			TmpBondingCounter++;
		}
	}

	if (TmpBondingCounter == 4) {
		//(Single-tile assembly: PostProcess return value is false, but UND is also false (trigger) )
		this->data.grid.set_xCell(__xThreadInfo, m_fit_DimGridX / 2, m_fit_DimGridY / 2, 0, 0);
		return false;
	}
	//Note: (Optional) Could now check for periodicity (can return to tile X first tile starting at X at same orientation)
	//Note: (Optional) Could now check for 2x2 assembly, etc (quite rare though)
	//NOTE: TODO, have to check in EdgeSort whether Tile is symmetric, i.e. then remove bonding orientations
	return true;
}

__device__ bool xAssembly::Assemble_Movelist(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet) {
	//Place tiletype 0 on center of grid
	this->data.grid.set_xCell(__xThreadInfo, m_fit_DimGridX / 2, m_fit_DimGridY / 2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 0);
	//Add first four moves to movelist (even iff they might be empty)
	uchar2 X; //X be current position in grid
	X.x = m_fit_DimGridX / 2; //X is abused here as a buffer (reset at loop head)
	X.y = m_fit_DimGridY / 2;
        //return false; //TEST

        { //Keep all this local
	xFourPermutation BUF((int) (curand_uniform(&this->data.states[__xThreadInfo.BankId()])*24.0f));
	unsigned char index;
	while (BUF.bNotTraversed()) {
		index = BUF.ucWalk();
		//unsigned char DBGVAL = TmpAddPerm.WalkIndex;
                this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) (index)));
                this->data.grid.set_xCell( __xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) (index)).x, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) (index)).y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), mEMPTY_CELL_ML);
                //this->data.grid.set_xCell( __xThreadInfo, 0, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 250); //TEST
	}
        }
 
        //return false;
        //this->data.assembly_size[__xThreadInfo.BankId()] = 1;

        //BEGIN DEBUG
        /*for(int i=0;i<mNrEdgeTypes;i++){
        	for(int j=0;j<this->data.edgesort.get_xLength(__xThreadInfo, i);j++){
                       for(int k=0;k<4;k++){
	 	               this->data.grid.set_xCell( __xThreadInfo, j*4+k, i, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.edgesort.data.multi_d[i][j][k][__xThreadInfo.BankId()]);
			}
                }
                this->data.grid.set_xCell( __xThreadInfo, this->data.edgesort.get_xLength(__xThreadInfo, i)*4, i, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 33<<2);
        } 
        return;*/
        //END DEBUG
 
	//We use movelist approach to assemble grids
	//Will switch to in-place assembly if either movelist full, or some other pre-defined condition.
	//Note: If we want mixed redundancy detection, need to implement some Single-Assembly Flag in AssemblyFlags that will switch.
	//Also: SynchronizeBank() needs to be adapted to not wait for other threads iff Many-thread approach!

	xCell N; //N(E_X) be non-empty neighbouring cells
	unsigned char Mirr; // Mirr(E_X, N(E_X)) be tile edge neighbouring E_X
	xCell T, TmpT; // T(Mirr(E_X, N(E_X)) be potential bonding tiles

        //BEGIN DEBUG
        //int DBG_MAXREP = 355;
        //int DBG_COUNTER = 0;
        //END DEBUG
	
        //For all elements M in Movelist (and while not UND condition detected)
 	while (this->data.movelist.get_sPos(__xThreadInfo) > 0) {
                //BEGIN DEBUG 
                //if(DBG_COUNTER >= DBG_MAXREP) return;
                //if(this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()){
                //	return;
                //}
                //STOP DEBUG

		//Choose position X from Movelist and remove it from Movelist
		X = this->data.movelist.xPop(__xThreadInfo);
                //Now remove grid marking to indicate movelist has been traversing this entry
                //this->data.grid.set_xCell( __xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), mEMPTY_CELL);

		T.set_xCell(mEMPTY_CELL);
                TmpT.set_xCell(mEMPTY_CELL);
		for (int E_X = 0; E_X < mNrTileOrientations; E_X++) {
			//BEGIN DEBUG
                        //this->data.grid.set_xCell( __xThreadInfo, 0, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 33<<2);
                        //END DEBUG

			//::Let N(E_X) be non-empty neighbouring cells.
			N = this->data.grid.xGetNeighbourCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), (unsigned char) E_X);

                        //BEGIN DEBUG
                        //this->data.grid.set_xCell( __xThreadInfo, E_X, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), N.get_xCell());
                        //END DEBUG

			if ( (N.get_xCell() != mEMPTY_CELL) && (N.get_xCell() != mEMPTY_CELL_ML) && (N.get_xCell() != mEMPTY_CELL_OUT_OF_BOUNDS) ) { //For all N(E_X)
				//::Let Mirr(E_X, N(E_X)) be tile neighbouring E_X
				unsigned char TmpMirrorCoord = (4 - N.get_xOrient() + (E_X + mNrTileOrientations / 2) % mNrTileOrientations) % mNrTileOrientations;
				Mirr = __xGenomeSet->get_xEdgeType(__xThreadInfo, N.get_xType(), TmpMirrorCoord);
				//For all Mirr(E_X, N(E_X)), let T(Mirr(E_X, N(E_X)) be potential bonding tiles
				TmpT.set_xCell(this->data.edgesort.GetBondingTile( __xThreadInfo, Mirr, &this->data.states[__xThreadInfo.BankId()], &this->data.flags[__xThreadInfo.BankId()]));

				//BEGIN DEBUG
                                /*this->data.grid.set_xCell( __xThreadInfo, E_X, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), Mirr);
				this->data.grid.set_xCell( __xThreadInfo, E_X, 2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), TmpT.get_xCell());*/
                                //END DEBUG

                                //TmpT.set_Orient((TmpT.get_xOrient() + E_X) % mNrTileOrientations);
				//NOTE: TrivialUND can arise in three ways:
				//1. For some Mirr, there is more than 1 bonding tile T (TrivialUND raised by GetBondingTile)
				//2. For some T, there is more than one orientation O
				//3. T does not agree between all N
				//Else if | T( Mirr( E_X, N(E_X) ) ) | == 0
				//If | T( Mirr( E_X, N(E_X) ) ) | > 0
				//Raise TrivialUND condition
				//Else If | T( Mirr( E_X, N(E_X) ) ) | == 1
				//if ( T.get_xCell() != mEMPTY_CELL ){ //Check if already tile there ??
				if (TmpT.get_xCell() != mEMPTY_CELL) {
                                        TmpT.set_Orient((TmpT.get_xOrient() + E_X) % mNrTileOrientations);
					if( (TmpT.get_xCell() != T.get_xCell()) && (T.get_xCell() != mEMPTY_CELL) ){
						//Raise TrivialUND!
						this->data.flags[__xThreadInfo.WarpId()].set_TrivialUND();
                                                /*BEGIN DEBUG*/
                                                //this->data.grid.set_xCell( __xThreadInfo, 2, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 3<<2); 
                                                //this->data.grid.set_xCell( __xThreadInfo, 0, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), TmpT.get_xCell() << 2);
                                                //this->data.grid.set_xCell( __xThreadInfo, 0, 2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), T.get_xCell() << 2);
                                                /*END DEBUG*/
                                                return;
					}
					T.set_xCell(TmpT.get_xCell());
					//As Bonding Cell is rotated such that bonding edge is facing North,
					//we need to rotate tile T such that bonding edge faces bonding site
					//Note: bonding orientations are handled above (GetBondingTile includes orientation).
					//::Let O(T) be all bonding orientations of T
					//If |O(T)| > 1
					//Else If |O(T)| = 1 --> Check Steric, if not --> Assemble
					//Let T* be T rotated such that E_T*(E_X) == E_T(O(T))
					//T.set_Orient((T.get_xOrient() + E_X) % mNrTileOrientations); //Rotate TmpT instead!
				}
			}
		} //Now we have looked for all neighbours of X and filtered the possible bonding tiles
		if (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition() && T.get_xCell() != mEMPTY_CELL) {

			//NOTE: StericUND can arise in two ways:
			//1. T does not agree with tile from previous assembly run
			//2. T does not agree with tile already at X in same run (multiple threads only)
                        xCell TmpT2;
			if (this->data.flags[__xThreadInfo.BankId()].get_ucRed() > 0) {
				TmpT2 = this->data.grid.get_xCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed() - 1);
				if (TmpT2.get_xCell() != T.get_xCell()) { //We have detected steric non-determinism!
					this->data.flags[__xThreadInfo.BankId()].set_StericUND(); //TEST
                                        /*START DEBUG*/
                                        //this->data.grid.set_xCell( __xThreadInfo, 0, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 7<<2); 
                                        /*END DEBUG*/
                                        return;
				}
			}

			//If X is not BorderCell
			//Assemble T* at X
			//Note: set_xCell will return false if BorderCell case!
			if (T.get_xCell() != mEMPTY_CELL) { 
                                //BEGIN DEBUG
                                //DBG_COUNTER++;
                                //bool test_flag=false;
                                //END DEBUG
                               
                                //Now: Assemble tile!
				if (!this->data.grid.set_xCell(	__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), T.get_xCell())) {
					this->data.flags[__xThreadInfo.BankId()].set_UnboundUND(); //TEST
                                        return;
                                        /*START DEBUG*/
                                        //this->data.grid.set_xCell( __xThreadInfo, 1, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 1<<2); 
                                        //this->data.grid.set_xCell( __xThreadInfo, 0, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), T.get_xCell());
                                        /*END DEBUG*/
                                        //return; //TEST
                                        //test_flag=true;
				}
                                //if(!test_flag){ //DEBUG
				xFourPermutation TmpAddPerm((int) (curand_uniform(&this->data.states[__xThreadInfo.BankId()]) * 24.0f));
				unsigned char index2; //Buffer
				while (TmpAddPerm.bNotTraversed()) {
					index2 = TmpAddPerm.ucWalk();
					//For all n(E_X)
					N = this->data.grid.xGetNeighbourCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), (unsigned char) index2);
					//::Let n(E_X) be empty neighbour cells (i.e. no tile and not on movelist already).
                                        //if(!test_flag){ //DEBUG
					if (N.get_xCell() == mEMPTY_CELL) {
                                                //if(!test_flag){ //DEBUG
                                               
						this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) index2)); 
                                                // } else { //DEBUG
                                                //this->data.grid.set_xCell( __xThreadInfo, 0, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) index2).x);
                                                //this->data.grid.set_xCell( __xThreadInfo, 0, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) index2).y); 
                                                //return;
                                                //} //DEBUG
                                                this->data.grid.set_xCell( __xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) (index2)).x, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) (index2)).y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), mEMPTY_CELL_ML);
                                                //this->data.grid.set_xCell( __xThreadInfo, 0, index2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 99);
					}//}
				}
			} 
		} else { //Remove movelist marking from grid
   	               this->data.grid.set_xCell( __xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), mEMPTY_CELL);
                }

	}
}

__device__ unsigned char xEdgeSort::GetBondingTile(xThreadInfo __xThreadInfo,
                short __sEdgeId, curandState *__xCurandState,
                xAssemblyFlags *__xAssemblyFlags) {
        //Takes: Edge Type to which the tile should bond, FitFlags which will be set according to UND conditions
        //Returns: Cell of Bonding Tile type which is rotated such that the bonding tile is facing NORTH (0),
        //If nothing bonds, will return mEMPTY_CELL instead.
        if (this->get_xLength(__xThreadInfo, __sEdgeId) == 1) {
                xCell TmpCell;
                unsigned char TmpBondBuffer = GetBondingTileOrientation(__xThreadInfo,
                                __sEdgeId, 0, __xAssemblyFlags);
                if(TmpBondBuffer == mEMPTY_CELL) return mEMPTY_CELL;
                TmpCell.set_xCell(4 - TmpBondBuffer);
                TmpCell.set_Type(this->get_xData(__xThreadInfo, __sEdgeId, 0,
                                TmpBondBuffer)); //TEST (0 anstelle TmpCell.get_xOrient()) b-fore
                return TmpCell.get_xCell();
        } else if (this->get_xLength(__xThreadInfo, __sEdgeId) == 0) {
                return mEMPTY_CELL;
        } else {
                __xAssemblyFlags->set_TrivialUND();
                return mEMPTY_CELL;
        }
}

__device__ unsigned char xEdgeSort::GetBondingTileOrientation(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucTileId, xAssemblyFlags *__xAssemblyFlags) {
	unsigned char TmpCounter = 0, TmpTile, TmpOrient = mEMPTY_CELL;
	for (int i = 0; i < mNrTileOrientations; i++) {
		TmpTile = this->get_xData(__xThreadInfo, __ucEdgeId, __ucTileId, i);
		if (TmpTile != mEMPTY_CELL) {
			TmpOrient = i;
			TmpCounter++;
			if (TmpCounter >= 2) {
				__xAssemblyFlags->set_TrivialUND();
                                TmpOrient = mEMPTY_CELL;
				break;
			}
		}
	}
	return TmpOrient; //should never be mEMPTY_CELL!
	//Returns edge-id of neighbouring tile that bonds
}

__device__ unsigned char xEdgeSort::get_xData(xThreadInfo __xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucTileId, unsigned char __ucOrientation) {
	return this->data.multi_d[__ucEdgeId][__ucTileId][__ucOrientation][__xThreadInfo.BankId()];
}

__device__ bool xAssembly::Assemble_InPlace(xThreadInfo __xThreadInfo,	xGenomeSet *__xGenomeSet) {
        return true;
}

}

__global__ void TestAssemblyKernel(unsigned char *g_ucGenomes, float *g_ucFitnessValues, unsigned char *g_ucGrids, curandState *states)
{
    __shared__ xGenomeSet s_xGenomeSet;
    //__shared__ xEdgeSort s_xEdgeSort;
    __shared__ xAssembly s_xAssembly;
    s_xAssembly.data.states = states;
    xThreadInfo r_xThreadInfo(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    s_xGenomeSet.CopyFromGlobal(r_xThreadInfo, g_ucGenomes);
    //s_xEdgeSort.Initialise(r_xThreadInfo, &s_xGenomeSet, -1);
    s_xAssembly.Assemble(r_xThreadInfo, &s_xGenomeSet);
    for(int i=0;i<4;i++){
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.length.multi_d[i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.data.multi_d[6][0][i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = tex2D(t_ucInteractionMatrix, i, 1);
    }
    s_xGenomeSet.CopyToGlobal(r_xThreadInfo, g_ucGenomes); 
    for(int i=0;i<m_fit_DimGridY;i++){
	 for(int j=0;j<m_fit_DimGridX;j++){
             xCell TMP = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0);
             g_ucGrids[r_xThreadInfo.BankId()*m_fit_DimGridX*m_fit_DimGridY + j*m_fit_DimGridX + i] = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0).get_xType();
         }
    }
}


    //end include fitness.cu

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