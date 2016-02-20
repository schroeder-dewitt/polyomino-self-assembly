//#include <stdio.h>

{# {% include "header_inc.cuh" %} #}
{# {% include "fit_header_inc.cuh" %} #}

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
