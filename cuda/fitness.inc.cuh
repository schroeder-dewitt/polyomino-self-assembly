#define m_fit_NrFourPermutations_c c_fFitnessParams[e_fit_NrFourPermutations]
#define m_fit_NrFourPermutations {{ fit_nr_fourpermutations }}
#define m_fit_NrAssemblyRedundancy_c c_fFitnessParams[e_fit_NrAssemblyRedundancy]
#define m_fit_NrAssemblyRedundancy {{ fit_assembly_redundancy }}
#define m_fit_THREAD_DIM_X_c c_fFitnessParams[e_fit_DimThreadX]
#define m_fit_THREAD_DIM_Y_c c_fFitnessParams[e_fit_DimThreadY]
#define m_fit_THREAD_DIM_X {{ fit_dimthreadx }}
#define m_fit_THREAD_DIM_Y {{ fit_dimthready }}
#define m_fit_NR_BLOCKS_c c_fFitnessParams[e_fit_NrBlocks]
#define m_fit_NrFitnessFunctionGrids_c c_fFitnessParams[e_fit_NrFitnessFunctionGrids]
#define m_fit_NR_THREADS_PER_BLOCK_c c_fFitnessParams[eNrThreadsPerBlock]
#define m_fit_NR_THREADS_PER_BLOCK {{ fit_nr_threadsperblock }}
#define m_fit_DimSubgridX_c c_fFitnessParams[e_fit_DimSubgridX]
#define m_fit_DimSubgridY_c c_fFitnessParams[e_fit_DimSubgridY]
#define m_fit_DimSubgridX {{ fit_dimsubgridx }}
#define m_fit_DimSubgridY {{ fit_dimsubgridy }}
#define m_fit_DimGridX_c c_fFitnessParams[e_fit_DimGridX]
#define m_fit_DimGridY_c c_fFitnessParams[e_fit_DimGridY]
#define m_fit_NrSubgridsPerBank_c c_fFitnessParams[e_fit_NrSubgridsPerBank]
#define m_fit_NrSubgridsPerBank {{ fit_nr_subgridsperbank }}
#define m_fit_DimGridX {{ fit_dimgridx }}
#define m_fit_DimGridY {{ fit_dimgridy }}
#define m_fit_NrFitnessFunctionGrids {{ fit_nr_fitnessfunctiongrids }}
#define m_fit_NrPixelInSubgrid (m_fit_DimSubgridX*m_fit_DimSubgridY)
#define m_fit_CurrentGrid 0
#define m_fit_PreviousGrid 1
#define WarpSize {{ WarpSize }}
#define m_fit_LengthMovelist {{ fit_lengthmovelist }}

#define EMPTY_CELLTYPE 255

typedef  uchar2 us2GRID2D;

enum FitnessParams {
    e_fit_DimGridX,
    e_fit_DimGridY,
    e_fit_NrFitnessFunctionGrids,
    e_fit_NrThreadsPerBlock,
    e_fit_NrBlocks,
    e_fit_DimSubgridX,
    e_fit_DimSubgridY,
    e_fit_NrSubgridsPerBank,
    e_fit_NrFourPermutations,
    e_fit_NrAssemblyRedundancy,
    e_fit_DimThreadX,
    e_fit_DimThreadY
};

enum FitnessMutexParams {
    mut_fit_BusyFlagMutex,
    mut_fit_FitnessValueMutex
};

//--------------------------------------------------------------------------------
typedef unsigned char xTILETYPE;

//DEPRECATED START
typedef unsigned char ucTILETYPE;
//DEPRECATED STOP

#define mWarpSize 32
#define m_fit_MAX_WARP_ID 8 //should be the number of warps scheduled per block
//#define m_fit_MULTIPLE_WARPS //should be defined if have multi-thread version
#define m_fit_SAFE_MEMORY_MAPPING //define if you wish to access shared memory safely rather than some bank-conflict free variant
#define mAlignedByteLengthGenome {{ genome_alignedbytelength }}
#define m_fit_BLOCK_DIM_X {{ fit_blockdimx }}
#define mEMPTY_CELL 255
typedef unsigned char ucCELL;
typedef unsigned char ucEDGETYPE;

struct xThreadInfo{
    ushort4 data;
    __device__ xThreadInfo(unsigned short __usThreadIdX, unsigned short __usThreadIdY, unsigned short __usBlockIdX, unsigned short __usBlockIdY);
    __device__ unsigned short WarpId(void);
    __device__ unsigned short BankId(void);
    __device__ unsigned short FlatThreadId(void);
    __device__ unsigned short FlatBlockId(void);
    __device__ unsigned short GlobId(unsigned short __usTypeLength);
    __device__ void __DEBUG_CALL(void){
         #ifndef __NON_FERMI
         printf("DBG CALL - xThreadInfo: [BankID=%d][WarpID=%d][FlatThreadID=%d][FlatBlockID=%d]", this->data.x, this->data.y, this->data.z, this->data.w);
         #endif
    }
};

__device__ void THROW_ErrorHeader(xThreadInfo *__xThreadInfo);

struct xGenome{
    union {
        unsigned char one_d[mByteLengthGenome];
        //NOTE: Here require a meta-check if padding length is nonzero!
        //unsigned char padding[mAlignedByteLengthGenome - mByteLengthGenome];        
    } data;
    //xGenome(){} //Note: disallowed for union!
    //~xGenome(){} //Note: disallowed for union!
    __device__ void CopyFromGlobal(xThreadInfo *__xThreadInfo, unsigned char *__g_ucGenomeSet);
    __device__ void CopyToGlobal(xThreadInfo *__xThreadInfo, unsigned char *__g_ucGenomeSet);
    __device__ unsigned char get_xEdgeType(xThreadInfo *__xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId);
};

#define mFFOrNIL(param) (param?0xFF:0x00)
#define mBitTest(index,byte) (((unsigned short)byte*(0x01<<index))>>index) 

struct xGenomeSet{
    union {
        xGenome multi_d[mWarpSize];
        unsigned char one_d[sizeof(xGenome) * mWarpSize];
    } data;
    __device__ xGenomeSet(){}
    __device__ ~xGenomeSet(){}
    __device__ void CopyFromGlobal(xThreadInfo *__xThreadInfo, unsigned char *__g_ucGenomeSet);
    __device__ void CopyToGlobal(xThreadInfo *__xThreadInfo, unsigned char *__g_ucGenomeSet);
    __device__ unsigned char get_xEdgeType(xThreadInfo *__xThreadInfo, unsigned char __ucTileId, unsigned char __ucEdgeId);
};

struct xList{
//Push, Pop, ...
};

struct xCell{
    unsigned char data;
    __device__ void set_Orient(unsigned int __uiOrient);
    __device__ void set_Type(unsigned int __uiType);
    __device__ unsigned char get_xType(void);
    __device__ unsigned char get_xOrient(void);
    __device__ unsigned char get_xCell(void);
};

struct xFitFlags{
    unsigned char bitset;
    __device__ xFitFlags(void){this->bitset = 0;}
    __device__ ~xFitFlags(void){}
    __device__ void set_TrivialUND(void){ this->bitset |= (1 << 0);}
    __device__ void set_UnboundUND(void){ this->bitset |= (1 << 1);}
    __device__ void set_StericUND(void){ this->bitset |= (1 << 2);}
    __device__ void set_BusyFlag(void){ this->bitset |= (1 << 7);}
    __device__ bool get_bTrivialUND(void){ return (bool)( this->bitset * (1 << 0) );}
    __device__ bool get_bUnboundUND(void){ return (bool)( this->bitset * (1 << 1) );}
    __device__ bool get_bStericUND(void){ return (bool)( this->bitset * (1 << 2) );}
    __device__ bool get_bBusyFlag(void){ return (bool)( this->bitset * (1 << 7) );}
    __device__ bool get_bUNDCondition(void){ return (bool)( this->bitset * 7);}
};

struct xEdgeSort {
    union {
        ucTILETYPE multi_d[mNrEdgeTypes][mNrTileTypes][mNrTileOrientations][mWarpSize];
        unsigned char one_d[mNrEdgeTypes*mNrTileTypes*mNrTileOrientations*mWarpSize];
    } data;

    union {
        unsigned short multi_d[mNrEdgeTypes][WarpSize];
        unsigned char one_d[mNrEdgeTypes*WarpSize*sizeof(short)];
    } length;

    __device__ xEdgeSort(){}
    __device__ ~xEdgeSort(){}
    __device__ void Zeroise (xThreadInfo *__xThreadInfo);
    __device__ void Initialise(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet, short __sEdgeId = -1);
    //__device__ ucTILETYPE GetBondingTile(xThreadInfo *__xThreadInfo, unsigned short __sEdgeId, curandState *__xCurandState, xFitFlags *__xFitFlags);
    __device__ ucCELL GetBondingTile(xThreadInfo *__xThreadInfo, short __sEdgeId, curandState *__xCurandState, xFitFlags *__xFitFlags);
    __device__ void add_TileOrient(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucOrient, unsigned char __ucTileType);
    __device__ __forceinline__ void set_xLength(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucLength);
    __device__ void add_Tile(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId);
    __device__ unsigned char get_xData(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId, unsigned char __ucTileId, unsigned char __ucOrientation);
    __device__ unsigned char GetBondingTileOrientation(xThreadInfo *__xThreadInfo, unsigned char  __ucEdgeId, unsigned char __ucTileId, xFitFlags __xFitFlags);
    __device__ void __DEBUG_CALL(xThreadInfo *__xThreadInfo){
         #ifndef __NON_FERMI
         printf("DBG CALL - xEdgeSort:");
         for(int i=0;i<mNrEdgeTypes;i++){
             printf("Edge: %d\n", i);
             for(int j=0;j<this->get_xLength(__xThreadInfo, i);j++){
                 printf("[");
                 for(int k=0;k<mNrTileOrientations;k++){
                     printf("%d,", this->data.multi_d[i][j][k][__xThreadInfo->WarpId()]);
                 }
                 printf("]");
             }
             printf("\n");
         }
         #endif
    }
    __device__ short get_xLength(xThreadInfo *__xThreadInfo, unsigned short __sEdgeId){
         #ifdef m_fit_SAFE_MEMORY_MAPPING
         if(__sEdgeId < mNrEdgeTypes){
             return this->length.multi_d[__sEdgeId][__xThreadInfo->WarpId()];
         } else {
             #ifndef __NON_FERMI
             THROW_ErrorHeader(*__xThreadInfo);
             printf(" get_xLength in xEdgeSort was called with EdgeId out of bounds!\n");
             #endif
             return 0;
         }
         #else
         #error "Safe Memory Mapping not implemented yet!"
         #endif
    }
};
