//#include <stdio.h>

//#define m_fit_LengthMovelist 20
//#define mEMPTY_CELL 255

{# {% include "header_inc.cuh" %} #}
{# {% include "fit_header_inc.cuh" %} #}

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
        this->list.multi_d[__xThreadInfo.BankId()].bPush(__xThreadInfo, __xEntry, this->storage.multi_d, {{ fit_length_movelist }});
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
