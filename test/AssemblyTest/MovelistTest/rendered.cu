#include <stdio.h>

#define m_fit_LengthMovelist 20
#define mEMPTY_CELL 255

/*This file contains all template macros for global simulation - Copyright Christian Schroeder, Oxford University 2012*/

#define SAFE_MEMORY_MAPPING 1
#define mAlignedByteLengthGenome 8
#define mWarpSize 32
#define mNrTileOrientations 4
#define mBitLengthEdgeType 3
#define mNrTileTypes 8
/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX 1
#define m_fit_DimThreadY 1
#define m_fit_DimBlockX 1

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

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
        this->list.multi_d[__xThreadInfo.BankId()].bPush(__xThreadInfo, __xEntry, this->storage.multi_d, 20); // sizeof(this->storage) / sizeof(T));
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
        dest[__xThreadInfo.BankId()*(s_xMovelist.get_sPos(__xThreadInfo)+1) + i] = i;
    }
}