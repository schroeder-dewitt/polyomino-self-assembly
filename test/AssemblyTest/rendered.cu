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

#define m_fit_DimGridX 19
#define m_fit_DimGridY 19

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define m_fit_LengthMovelist 244
#define m_fit_NrRedundancyGridDepth 2
#define m_fit_NrRedundancyAssemblies 10
#define m_fit_TileIndexStartingTile 0

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













#include <stdio.h>

#define m_fit_LengthMovelist 20
#define mEMPTY_CELL 255

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

#define m_fit_DimGridX 19
#define m_fit_DimGridY 19

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define m_fit_LengthMovelist 244
#define m_fit_NrRedundancyGridDepth 2
#define m_fit_NrRedundancyAssemblies 10
#define m_fit_TileIndexStartingTile 0

#define mEMPTY_CELL 255

/*struct xThreadInfo {
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
}*/

extern "C++"{

template<class T>
struct xLifoList {
        //union {
        struct {
                signed short pos; //Current position of top element (-1...max_length-1)
                //unsigned char one_d[sizeof(signed short)];
        } data;

        __device__ bool bPush(xThreadInfo __xThreadInfo, T __xEntry, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize], unsigned short __uiMaxLength);
//        __device__ T xPop(T* __xStorage);
	__device__ T xPop(xThreadInfo __xThreadInfo, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize]);
        __device__ short get_sPos();
        __device__ short set_sPos(short __sPos);
};

template<class T>
__device__ bool xLifoList<T>::bPush(xThreadInfo __xThreadInfo, T __xEntry, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize], unsigned short __uiMaxLength) {
        this->data.pos++;
        if (this->data.pos < __uiMaxLength) {
                __xStorage[this->data.pos][__xThreadInfo.BankId()] = __xEntry;
                return true;
        } else {
                this->data.pos--;
                return false;
        }
}

template<class T>
//__device__ T xLifoList<T>::xPop(xThreadInfo __xThreadInfo, T* __xStorage) {
__device__ T xLifoList<T>::xPop(xThreadInfo __xThreadInfo, T (&__xStorage)[m_fit_LengthMovelist][mWarpSize]) {
        if (this->data.pos < 0) {
                //NOTE: ADAPTED FOR CUDA VECTORTYPES ONLY!
                T buf;
                buf.x = mEMPTY_CELL;
                return buf;
        } else {
                this->data.pos--;
                return __xStorage[this->data.pos+1][__xThreadInfo.BankId()];
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
        //union {
        struct {
                T multi_d[m_fit_LengthMovelist][mWarpSize];
                //unsigned char one_d[mWarpSize * m_fit_LengthMovelist * sizeof(T)];
        } storage;
        //union {
        struct {
                xLifoList<T> multi_d[mWarpSize];
                //unsigned char one_d[sizeof(xLifoList<T>) * mWarpSize];
        } list;

        __device__ void Initialise(xThreadInfo __xThreadInfo);
        __device__ bool bPush(xThreadInfo __xThreadInfo, T __xEntry);
        __device__ T xPop(xThreadInfo __xThreadInfo);
        __device__ short get_sPos(xThreadInfo __xThreadInfo);
        __device__ short set_sPos(xThreadInfo __xThreadInfo, short __sPos);
};

template<class T>
__device__ bool xMoveList<T>::bPush(xThreadInfo __xThreadInfo, T __xEntry) {
//        this->list.multi_d[__xThreadInfo.BankId()].bPush(__xEntry, this->storage.multi_d[__xThreadInfo.BankId()], 20); // sizeof(this->storage) / sizeof(T));
        this->list.multi_d[__xThreadInfo.BankId()].bPush(__xThreadInfo, __xEntry, this->storage.multi_d, 244); // sizeof(this->storage) / sizeof(T));
        //this->storage.multi_d[0][__xThreadInfo.BankId()].x = 7;
        //this->list.multi_d[__xThreadInfo.BankId()].data.pos = 1;
        return true;
}

template<class T>
__device__ void xMoveList<T>::Initialise(xThreadInfo __xThreadInfo) {
        this->list.multi_d[__xThreadInfo.BankId()].data.pos = -1;
}

template<class T>
__device__ T xMoveList<T>::xPop(xThreadInfo __xThreadInfo) {
        //return this->list.multi_d[__xThreadInfo.BankId()].xPop(this->storage.multi_d[__xThreadInfo.BankId()]);
	return this->list.multi_d[__xThreadInfo.BankId()].xPop(__xThreadInfo, this->storage.multi_d);
        //return this->storage.multi_d[0][__xThreadInfo.BankId()];
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

__constant__ unsigned char c_ucFourPermutations[24][4];

struct xFourPermutation {
	unsigned short PermIndex;
	unsigned short WalkIndex;
	__device__ xFourPermutation(unsigned short __usPermIndex);
	__device__ unsigned short ucWalk();
	__device__ bool bNotTraversed();
};

__device__ xFourPermutation::xFourPermutation(unsigned short __usPermIndex) {
    this->PermIndex = __usPermIndex % 24;
    this->WalkIndex = 0;
}

__device__ unsigned short xFourPermutation::ucWalk() {
    //Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
    this->WalkIndex++;
    if (this->WalkIndex - 1 < mNrTileOrientations) {
        return c_ucFourPermutations[this->PermIndex][this->WalkIndex - 1] - 1;
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
    this->WalkIndex = 0;
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
	for (int i = 0; i < m_fit_DimGridX; i++) {
		for (int j = 0; j < m_fit_DimGridY; j++) {
			this->data.multi_d[i][j][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL);
		}
	}
}

__device__ xCell xCellGrid::get_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red) {
	if ((__x < m_fit_DimGridX) && (__y < m_fit_DimGridY)) {
		return this->data.multi_d[__x][__y][__red][__xThreadInfo.BankId()];
	} else {
		xCell TmpCell;
		TmpCell.set_xCell(mEMPTY_CELL);
		return TmpCell;
	}
}

__device__ bool xCellGrid::set_xCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __val) {
	if ((__x < m_fit_DimGridX - 1) && (__y < m_fit_DimGridY - 1)) {
		this->data.multi_d[__x][__y][__red][__xThreadInfo.BankId()].set_xCell(
				__val);
		return true;
	} else if (__x == (m_fit_DimGridX - 1) || (__y == (m_fit_DimGridY - 1))) {
		//UnboundUND condition! Return false.
		this->data.multi_d[__x][__y][__red][__xThreadInfo.BankId()].set_xCell(
				__val);
		return false;
	} else {
		return false;
	}
}

__device__ xCell xCellGrid::xGetNeighbourCell(xThreadInfo __xThreadInfo, unsigned char __x, unsigned char __y, unsigned char __red, unsigned char __dir) {
	uchar2 TmpCoords = xGetNeighbourCellCoords(__x, __y, __dir);
	return this->get_xCell(__xThreadInfo, TmpCoords.x, TmpCoords.y, __red);
}

__device__ uchar2 xCellGrid::xGetNeighbourCellCoords(unsigned char __x, unsigned char __y, unsigned char __dir) {
	switch (__dir) {
	case 1: //EAST
		return make_uchar2(__x + 1, __y);
		//break;
	case 3: //WEST
		return make_uchar2(__x - 1, __y);
		//break;
	case 2: //SOUTH
		return make_uchar2(__x, __y + 1);
		//break;
	case 0: //NORTH
		return make_uchar2(__x, __y - 1);
		//break;
	default:
		break;
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
		curandState *states;//[mWarpSize];
		unsigned int synccounter[mWarpSize]; //Will be used to synchronize between Warps
	} data;

	__device__ void Initialise(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
        __device__ bool Assemble_PreProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_PostProcess(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_Movelist(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ bool Assemble_InPlace(xThreadInfo __xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__ float fEvaluateFitness(xThreadInfo __xThreadInfo, xFitnessGrid *__fFitnessGrid, bool __bSingleBlockId);
	__device__ float fEvaluateFitnessForSingleGrid(xThreadInfo __xThreadInfo, xFitnessGrid *__xSingleFitnessGrid, bool __bIsSingleBlock);
	__device__ bool bSynchronizeBank(xThreadInfo __xThreadInfo);
};

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
		this->data.edgesort.Initialise(__xThreadInfo, __xGenomeSet); //TEST
		//this->Assemble_PostProcess(__xThreadInfo, __xGenomeSet);
		if (TmpFlag) {
			for (int i = 0; (i < m_fit_NrRedundancyAssemblies) && (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()); i++) {
				this->Initialise(__xThreadInfo, __xGenomeSet); //Empty out assembly grid at red
				bool TmpController = this->Assemble_Movelist(__xThreadInfo, __xGenomeSet);
/*				if (!TmpController) TmpController = this->Assemble_InPlace(__xThreadInfo, __xGenomeSet);
				if (!TmpController) {
					// Both assembly processes did not finish! (should NEVER happen)
					return false; //Always false - indicate assembly did not finish properly (should not happen!)
				}
				this->data.flags[__xThreadInfo.BankId()].set_Red(i); //Choose next assembly step!
*/
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
					DBGVAL1 = __xGenomeSet->get_xEdgeType(__xThreadInfo, k, l);
					DBGVAL2 = __xGenomeSet->get_xEdgeType(__xThreadInfo, i, (j
							+ l) % mNrTileOrientations);
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
	this->data.grid.set_xCell(__xThreadInfo, m_fit_DimGridX / 2, m_fit_DimGridY / 2, 0, 0);
	//Add first four moves to movelist (even iff they might be empty)
	uchar2 X; //X be current position in grid
	X.x = m_fit_DimGridX / 2;
	X.y = m_fit_DimGridY / 2;
        //return false; //TEST

	this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) 0));
	this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) 1));
	this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) 2));
	this->data.movelist.bPush(__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) 3));

        //return false; //TEST
        
	//We use movelist approach to assemble grids
	//Will switch to in-place assembly if either movelist full, or some other pre-defined condition.

	//Note: If we want mixed redundancy detection, need to implement some Single-Assembly Flag in AssemblyFlags that will switch.
	//Also: SynchronizeBank() needs to be adapted to not wait for other threads iff Many-thread approach!

#ifndef m_fit_MULTIPLE_WARPS
	xCell N; //N(E_X) be non-empty neighbouring cells
	unsigned char Mirr; // Mirr(E_X, N(E_X)) be tile edge neighbouring E_X
	xCell T, TmpT; // T(Mirr(E_X, N(E_X)) be potential bonding tiles
	//For all elements M in Movelist (and while not UND condition detected)
	while ((this->data.movelist.get_sPos(__xThreadInfo) >= 0) && (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition())) {
		//Choose position X from Movelist and remove it from Movelist

		//this->data.grid.print(__xThreadInfo, __xGenomeSet);
                //return false;
		X = this->data.movelist.xPop(__xThreadInfo);
                //return false;
		T.set_xCell(mEMPTY_CELL);
		for (int E_X = 0; (E_X < mNrTileOrientations)
				&& (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()); E_X++) {
			//::Let N(E_X) be non-empty neighbouring cells.
			N = this->data.grid.xGetNeighbourCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), (unsigned char) E_X);
			if (N.get_xCell() != mEMPTY_CELL) { //For all N(E_X)
				//::Let Mirr(E_X, N(E_X)) be tile neighbouring E_X
				//Mirr = __xGenomeSet->get_xEdgeType(__xThreadInfo, N.get_xType(), (mNrTileOrientations-(E_X+mNrTileOrientations/2)%mNrTileOrientations)%mNrTileOrientations );
				unsigned char DBGVAL = N.get_xOrient();
				unsigned char TmpMirrorCoord = (4 - N.get_xOrient() + (E_X + mNrTileOrientations / 2) % mNrTileOrientations) % mNrTileOrientations;
				Mirr = __xGenomeSet->get_xEdgeType(__xThreadInfo, N.get_xType(), TmpMirrorCoord);
				//For all Mirr(E_X, N(E_X)), let T(Mirr(E_X, N(E_X)) be potential bonding tiles
				TmpT.set_xCell(this->data.edgesort.GetBondingTile( __xThreadInfo, Mirr, &this->data.states[__xThreadInfo.BankId()], &this->data.flags[__xThreadInfo.BankId()]));

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
					//if( TmpT.get_xCell() != T.get_xCell() ){
					//	//Raise TrivialUND!
					//	this->data.flags[__xThreadInfo->WarpId()].set_TrivialUND();
					//}
					T.set_xCell(TmpT.get_xCell());
					//As Bonding Cell is rotated such that bonding edge is facing North,
					//we need to rotate tile T such that bonding edge faces bonding site
					//Note: bonding orientations are handled above (GetBondingTile includes orientation).
					//::Let O(T) be all bonding orientations of T
					//If |O(T)| > 1
					//Else If |O(T)| = 1 --> Check Steric, if not --> Assemble
					//Let T* be T rotated such that E_T*(E_X) == E_T(O(T))
					//unsigned char DBGVAL10 = (T.get_xOrient() + E_X) % mNrTileOrientations;
					//unsigned char DBGVAL11 = T.get_xOrient();
					//printf("CELL DBG %d and Orient: %d:\n", T.get_xCell(), T.get_xOrient());
					T.set_Orient((T.get_xOrient() + E_X) % mNrTileOrientations);
				}
			}
		}
		if (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition() && T.get_xCell() != mEMPTY_CELL) {

			//NOTE: StericUND can arise in two ways:
			//1. T does not agree with tile from previous assembly run
			//2. T does not agree with tile already at X in same run (multiple threads only)
#ifdef m_fit_MULTIPLE_WARPS
			//NOTE: Multi-threading only: Check if there is already a different non-empty tile at X!
			TmpT = this->data.grid.get_xCell(__xThreadInfo, X.x, X.y, this->flags.get_ucRed());
			if(TmpT.get_xCell() != mEMPTY_CELL) {
				if(TmpT.get_xCell() != T.get_xCell()) {
					this->data.flags[__xThreadInfo.BankId()].set_StericUND(); //TEST
				}
			}
#endif

			if (this->data.flags[__xThreadInfo.BankId()].get_ucRed()) {
				TmpT = this->data.grid.get_xCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed() - 1);
				if (TmpT.get_xCell() != T.get_xCell()) { //We have detected steric non-determinism!
					this->data.flags[__xThreadInfo.BankId()].set_StericUND(); //TEST
				}
			}
			if (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()) {
				//If X is not BorderCell
				//Assemble T* at X
				//Note: set_xCell will return false if BorderCell case!
				if (T.get_xCell() != mEMPTY_CELL) {
					if (!this->data.grid.set_xCell(	__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), T.get_xCell())) {
						this->data.flags[__xThreadInfo.BankId()].set_UnboundUND();
					}
					if (!this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()) {
						xFourPermutation TmpAddPerm((int) (curand_uniform(
								&this->data.states[__xThreadInfo.BankId()])
								* 24.0f));
						unsigned char E_X;
						while (TmpAddPerm.bNotTraversed()) {
							E_X = TmpAddPerm.ucWalk();
							unsigned char DBGVAL = TmpAddPerm.WalkIndex;
							//For all n(E_X)
							N = this->data.grid.xGetNeighbourCell(__xThreadInfo, X.x, X.y, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), (unsigned char) E_X);
							//::Let n(E_X) be empty neighbour cells.
							if (N.get_xCell() == mEMPTY_CELL) {
								if (!this->data.movelist.bPush(	__xThreadInfo, this->data.grid.xGetNeighbourCellCoords(X.x, X.y, (unsigned char) E_X))) {
									this->data.flags[__xThreadInfo.BankId()].set_BusyFlag();
								}
							}
						}
					}
				}
			}
		}
	}
	if (!this->data.flags[__xThreadInfo.BankId()].get_bBusyFlag())
		return true;
	else
		return false; //i.e. Need to continue with in-place assembly!
#else
#endif
}

__device__ unsigned char xEdgeSort::GetBondingTile(xThreadInfo __xThreadInfo,
                short __sEdgeId, curandState *__xCurandState,
                xAssemblyFlags *__xAssemblyFlags) {
        //Takes: Edge Type to which the tile should bond, FitFlags which will be set according to UND conditions
        //Returns: Cell of Bonding Tile type which is rotated such that the bonding tile is facing NORTH (0),
        //If nothing bonds, will return mEMPTY_CELL instead.
        if (this->get_xLength(__xThreadInfo, __sEdgeId) == 1) {
                xCell TmpCell;
                unsigned char DBGVAL2, DBGVAL3, DBGVAL = GetBondingTileOrientation(
                                __xThreadInfo, __sEdgeId, 0, __xAssemblyFlags);
                unsigned char TmpBondBuffer = GetBondingTileOrientation(__xThreadInfo,
                                __sEdgeId, 0, __xAssemblyFlags);
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

