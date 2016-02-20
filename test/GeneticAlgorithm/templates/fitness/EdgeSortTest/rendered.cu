extern "C++"{
#include <curand_kernel.h>
}

#define m_curand_NR_THREADS_PER_BLOCK 256

extern "C" __global__ void CurandInitKernel(curandState *state)
{
    int id = m_curand_NR_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    //Each thread gets same seed, a different sequence number, no offset 
    curand_init(1234, id, 0, &state[id]);
}

#include <stdio.h>

/*This file contains all template macros for global simulation - Copyright Christian Schroeder, Oxford University 2012*/

#define SAFE_MEMORY_MAPPING 1
#define mAlignedByteLengthGenome 8
#define mWarpSize 32
#define mNrTileOrientations 4
#define mBitLengthEdgeType 3
#define mNrTileTypes 4
#define mNrEdgeTypes 8
/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX 1
#define m_fit_DimThreadY 1
#define m_fit_DimBlockX 1

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define mEMPTY_CELL 255

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
    this->bitset |= (1 << 4); 
}

__device__ void xAssemblyFlags::set_UnboundUND(void) {
    this->bitset |= (1 << 5);
}

__device__ void xAssemblyFlags::set_StericUND(void) {
    this->bitset |= (1 << 6);
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
    return (bool) (this->bitset & 7);
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

__device__ void xEdgeSort::Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet, short __sEdgeId) {
	if (__sEdgeId == -1) {
		for (int k = 0; k < mNrEdgeTypes; k++) {
			this->set_xLength(__xThreadInfo, k, 0);
			bool r_bEdgeAdded = false;
			for (int i = 0; i < mNrTileTypes; i++) { /*Traverse all TileTypes*/
				for (int j = 0; j < mNrTileOrientations; j++) { /*Traverse all Orientations*/
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
			}
                       //set_xLength(__xThreadInfo, 0, 5); //Test: 
		}
	} else {
		this->set_xLength(__xThreadInfo, __sEdgeId, 0);
		bool r_bEdgeAdded = false;
		for (int i = 0; i < mNrTileTypes; i++) { /*Traverse all TileTypes*/
			for (int j = 0; j < mNrTileOrientations; j++) { /*Traverse all Orientations*/
				if (tex2D(t_ucInteractionMatrix, __xGenomeSet->get_xEdgeType(
						__xThreadInfo, i, j), __sEdgeId) > 0) { /*Does Edge j of Tile i bond to Tile ThreadID*/
					this->add_TileOrient(__xThreadInfo, __sEdgeId, j, i);
					r_bEdgeAdded = true;
				} else {
					this->add_TileOrient(__xThreadInfo, __sEdgeId, j,
							mEMPTY_CELL);
				}
			}
			if (r_bEdgeAdded) { /*EdgeAdded?*/
				this->add_Tile(__xThreadInfo, __sEdgeId);
				r_bEdgeAdded = 0;
			} else {
				/*Do Nothing*/
			}
		}
	}
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












