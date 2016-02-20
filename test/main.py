import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>

#define m_fit_SAFE_MEMORY_MAPPING
#define mAlignedByteLengthGenome 8
#define mWarpSize 32
#define m_fit_THREAD_DIM_X 32
#define m_fit_THREAD_DIM_Y 1
#define m_fit_BLOCK_DIM_X 1

struct xThreadInfo {
        ushort4 data;__device__
        xThreadInfo(unsigned short __usThreadIdX, unsigned short __usThreadIdY,
                        unsigned short __usBlockIdX, unsigned short __usBlockIdY);__device__
        unsigned short WarpId(void);__device__
        unsigned short BankId(void);__device__
        unsigned short FlatThreadId(void);__device__
        unsigned short FlatBlockId(void);__device__
        unsigned short GlobId(unsigned short __usTypeLength);__device__
        void __DEBUG_CALL(void);
};

__device__ xThreadInfo::xThreadInfo(unsigned short __usThreadIdX,
                unsigned short __usThreadIdY, unsigned short __usBlockIdX,
                unsigned short __usBlockIdY) {
        this->data.z = threadIdx.y * m_fit_THREAD_DIM_X + threadIdx.x; //Flat Thread ID
        this->data.x = this->data.z % mWarpSize; //BankID
        this->data.y = (this->data.z - this->data.x) / mWarpSize; //WarpID
        this->data.w = blockIdx.y * m_fit_BLOCK_DIM_X + blockIdx.x; //Flat Block ID
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
        return (this->data.w * m_fit_THREAD_DIM_X * m_fit_THREAD_DIM_Y
                        + this->data.z) * __usTypeLength;
}

struct xGenome {
        struct {
                unsigned char one_d[mAlignedByteLengthGenome];
                //NOTE: Here require a meta-check if padding length is nonzero!
                //unsigned char padding[mAlignedByteLengthGenome - mByteLengthGenome];
        } data;
        __device__ xGenome(){} //Note: disallowed for union!
        __device__ ~xGenome(){} //Note: disallowed for union!
        __device__
        void CopyFromGlobal(xThreadInfo __xThreadInfo,
                        unsigned char *__g_ucGenomeSet);__device__
        void CopyToGlobal(xThreadInfo __xThreadInfo,
                        unsigned char *__g_ucGenomeSet);__device__
        unsigned char get_xEdgeType(xThreadInfo *__xThreadInfo,
                        unsigned char __ucTileId, unsigned char __ucEdgeId);__device__
        void set_EdgeType(xThreadInfo *__xThreadInfo, unsigned char __ucTileId,
                        unsigned char __ucEdgeId, unsigned char __ucVal);
};

struct xGenomeSet {
        struct {
                xGenome multi_d[mWarpSize];
                //unsigned char one_d[sizeof(xGenome) * mWarpSize];
        } data;__device__
        xGenomeSet() {
        }
        __device__
        ~xGenomeSet() {
        }
        __device__
        void CopyFromGlobal(xThreadInfo __xThreadInfo,
                        unsigned char *__g_ucGenomeSet);__device__
        void CopyToGlobal(xThreadInfo *__xThreadInfo,
                        unsigned char *__g_ucGenomeSet);__device__
        unsigned char get_xEdgeType(xThreadInfo *__xThreadInfo,
                        unsigned char __ucTileId, unsigned char __ucEdgeId);__device__
        unsigned char set_EdgeType(xThreadInfo *__xThreadInfo,
                        unsigned char __ucTileId, unsigned char __ucEdgeId,
                        unsigned char __ucVal);__device__
        void print(xThreadInfo *__xThreadInfo);
};

__device__ void xGenomeSet::CopyFromGlobal(xThreadInfo __xThreadInfo,
                unsigned char *__g_ucGenomes) {
        this->data.multi_d[__xThreadInfo.WarpId()].CopyFromGlobal(__xThreadInfo, __g_ucGenomes);
}

__device__ void xGenome::CopyFromGlobal(xThreadInfo __xThreadInfo,
                unsigned char *__g_ucGenomes) {
        for (int i = 0; i < mAlignedByteLengthGenome; i += 1) {
                if(!threadIdx.x) printf("%d ",i);
                this->data.one_d[i] = __g_ucGenomes[__xThreadInfo.GlobId(sizeof(xGenome)) + i];
        }
}


__global__ void multiply_them(unsigned char *dest)
{
    __shared__ xGenomeSet Tmp;
    xThreadInfo Tmpa(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    Tmp.CopyFromGlobal(Tmpa, dest);    
    //printf(Tmp.);
}
""", arch="compute_20", code="sm_20")

multiply_them = mod.get_function("multiply_them")

#a = numpy.random.randn(400).astype(numpy.uint8)
#b = numpy.random.randn(400).astype(numpy.uint8)

dest = numpy.zeros(8).astype(numpy.uint8)
dest_h = drv.mem_alloc(dest.nbytes)
drv.memcpy_htod(dest_h, dest)
multiply_them(
        dest_h,
        block=(1,1,1), grid=(1,1))

print dest
