
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

#define mNrRedundancyAssemblies 1
#define mNrRedundancyGridDepth 2
#define mTileIndexStartingTile 0
#define m_fit_InPlaceFullCheckCutoff 20 //Delimits after how many idle steps should be doing a full grid check during in-place assembly
#define m_fit_NrPixelPerSubgrid 50 //ADJUST THIS
#define m_fit_NrFitnessGrids 5

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

//PARAMETER START ----------------------------------------------------------------------------------

//DEVICE HEADER START ------------------------------------------------------------------------------
typedef unsigned char ucEDGETYPE;
typedef unsigned char ucTILETYPE;
typedef enum {
	u_eNORTH=0, u_eEAST=1, u_eSOUTH=2, u_eWEST=3
} ucDIRECTION;

#define mXOR(a, b) (((a)&~(b))|(~(a)&(b)))
#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x01<<index))
#define mEmptyOrZero(param) (param==mEMPTY_CELL?mEMPTY_CELL:0x00)

struct xFourPermutation {
	unsigned short PermIndex;
	unsigned short WalkIndex;
	__device__ xFourPermutation(unsigned short __usPermIndex) {
		this->PermIndex = __usPermIndex % 24;
		this->WalkIndex = 0;
	}
        __device__ unsigned short ucWalk() {
		//Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
		this->WalkIndex++;
		if (this->WalkIndex - 1 < mNrTileOrientations) {
			return c_ucFourPermutations[this->PermIndex][this->WalkIndex - 1]
					- 1;
		} else
			return 0;
	}
	__device__ bool bNotTraversed() {
		//Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
		if (this->WalkIndex >= mNrTileOrientations) {
			return false;
		} else
			return true;
	}
};
extern "C++" {
template<int Length>
struct xLinearIterator {
	unsigned short WalkIndex;
	__device__ xLinearIterator(unsigned short __usPermIndex) {
		this->WalkIndex = 0;
	}
	__device__ unsigned short ucWalk() {
		//Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
		this->WalkIndex++;
		if (this->WalkIndex - 1 < Length) {
			return this->WalkIndex - 1;
		} else
			return 0;
	}
	__device__ bool bNotTraversed() {
		//Require c_ucFourPermutations to be numbers 1-4 (NOT 0-3)
		if (this->WalkIndex >= Length) {
			return false;
		} else
			return true;
	}
};

}

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

__device__ void xThreadInfo::__DEBUG_CALL(void) {
#ifndef __NON_FERMI
        //#ifdef __PRINTF__

                printf(
                                "DBG CALL - xThreadInfo: [BankID=%d][WarpID=%d][FlatThreadID=%d][FlatBlockID=%d]",
                                this->data.x, this->data.y, this->data.z, this->data.w);
        //#endif
#endif
        }


__device__ void THROW_ErrorHeader(xThreadInfo *__xThreadInfo);

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

extern "C++" {
//Note: T MUST be a CUDA vector-type!
template<class T>
struct xLifoList {
	union {
		signed short pos; //Current position of top element (-1...max_length-1)
		unsigned char one_d[sizeof(signed short)];
	} data;

	//__device__ xLifoList(void){
	//    this->pos = -1;
	//}
	//__device__ ~xLifoList(){}
	__device__
	bool bPush(T __xEntry, T* __xStorage, unsigned short __uiMaxLength);__device__
	T xPop(T* __xStorage);__device__
	short get_sPos();
	short set_sPos(short __sPos);
};

template<class T>
struct xMoveList {
	union {
		T multi_d[m_fit_LengthMovelist][mWarpSize];
		unsigned char one_d[mWarpSize * m_fit_LengthMovelist * sizeof(T)];
	} storage;
	union {
		xLifoList<T> multi_d[mWarpSize];
		unsigned char one_d[sizeof(xLifoList<T> ) * mWarpSize];
	} list;
	//xMoveList(){}
	//~xMoveList(){}
	__device__	void Initialise(xThreadInfo *__xThreadInfo);
	__device__	bool bPush(xThreadInfo *__xThreadInfo, T __xEntry);
	__device__	T xPop(xThreadInfo *__xThreadInfo);
	__device__	short get_sPos(xThreadInfo *__xThreadInfo);
	__device__	short set_sPos(xThreadInfo *__xThreadInfo, short __sPos);
	__device__	void print_grid(xThreadInfo *__xThreadInfo);
};

}

struct xCell {
	unsigned char data;__device__
	void set_Orient(unsigned char __uiOrient);__device__
	void set_Type(unsigned char __uiType);__device__
	unsigned char get_xType(void);__device__
	unsigned char get_xOrient(void);__device__
	unsigned char get_xCell(void);__device__
	void set_xCell(unsigned char __ucVal);
};

struct xAssemblyFlags {
	unsigned char bitset;
	unsigned char bitset2;
	unsigned char red;
	unsigned char fullcheckcutoff;
	//__device__ xAssemblyFlags(void){this->bitset = 0;}
	//__device__ ~xAssemblyFlags(void){}
	__device__
	void set_Red(unsigned char __ucVal) {
		this->red = __ucVal;
	}
	__device__ void set_TrivialUND(void);
	__device__ void set_UnboundUND(void);
	__device__ void set_StericUND(void) {
		this->bitset |= (1 << 6);
	}
	__device__ void set_BusyFlag(void) {
		this->bitset |= (1 << 7);
	}
	__device__ bool get_bTrivialUND(void) {
		return (bool) (this->bitset & (1 << 4));
	}
	__device__ bool get_bUnboundUND(void) {
		return (bool) (this->bitset & (1 << 5));
	}
	__device__ bool get_bStericUND(void) {
		return (bool) (this->bitset & (1 << 6));
	}
	__device__ bool get_bBusyFlag(void) {
		return (bool) (this->bitset & (1 << 7));
	}
	__device__
	bool get_bUNDCondition(void) {
		//printf("\nMUHH: %d\n", (this->bitset & 7));
		return (bool) (this->bitset & 7);
	}
	__device__
	unsigned char get_ucRed(void) {
		return this->red;
	}
	__device__
	void ClearAll(void) {
		this->bitset = 0;
		this->bitset2 = 0;
		this->red=0;
		this->fullcheckcutoff=0;
		return;
	}
	__device__
	void ClearBitsets(void) {
		this->bitset = 0;
		this->bitset2 = 0;
		return;
	}
};

__device__ void xAssemblyFlags::set_TrivialUND(void) {
    //this->bitset |= (1 << 4); //TEST
}
__device__ void xAssemblyFlags::set_UnboundUND(void) {
    this->bitset |= (1 << 5);
}


struct xCellGrid {
	union {
		xCell
				multi_d[m_fit_DimGridX][m_fit_DimGridY][mNrRedundancyGridDepth][mWarpSize];
		xCell
				mix_d[m_fit_DimGridX * m_fit_DimGridY][mNrRedundancyGridDepth][mWarpSize];
		xCell one_d[m_fit_DimGridX * m_fit_DimGridY * mWarpSize
				* mNrRedundancyGridDepth];
	} data;

	__device__
	void Initialise(xThreadInfo *__xThreadInfo, unsigned char __red);__device__
	xCell get_xCell(xThreadInfo *__xThreadInfo, unsigned char __x,
			unsigned char __y, unsigned char __red);__device__
	bool set_xCell(xThreadInfo *__xThreadInfo, unsigned char __x,
			unsigned char __y, unsigned char __red, unsigned char __val);__device__
	xCell xGetNeighbourCell(xThreadInfo *__xThreadInfo, unsigned char __x,
			unsigned char __y, unsigned char __red, ucDIRECTION __dir);__device__
	uchar2 xGetNeighbourCellCoords(unsigned char __x, unsigned char __y,
			ucDIRECTION __dir);__device__
	bool xCompareRed(xThreadInfo *__xThreadInfo, unsigned char __red);__device__
	void print(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet);
};

struct xEdgeSort {
	union {
		ucTILETYPE
				multi_d[mNrEdgeTypes][mNrTileTypes][mNrTileOrientations][mWarpSize];
		unsigned char one_d[mNrEdgeTypes * mNrTileTypes * mNrTileOrientations
				* mWarpSize];
	} data;

	union {
		unsigned short multi_d[mNrEdgeTypes][mWarpSize];
		unsigned char one_d[mNrEdgeTypes * mWarpSize * sizeof(short)];
	} length;

	//__device__ xEdgeSort(){}
	//__device__ ~xEdgeSort(){}
	__device__
	void Zeroise(xThreadInfo *__xThreadInfo);__device__
	void Initialise(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet,
			short __sEdgeId = -1);__device__
	ucCELL GetBondingTile(xThreadInfo *__xThreadInfo, short __sEdgeId,
			curandState *__xCurandState, xAssemblyFlags *__xAssemblyFlags);__device__
	void add_TileOrient(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId,
			unsigned char __ucOrient, unsigned char __ucTileType);
	__device__ __forceinline__ void set_xLength(xThreadInfo *__xThreadInfo,
			unsigned char __ucEdgeId, unsigned char __ucLength);__device__
	void add_Tile(xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId);__device__
	unsigned char get_xData(xThreadInfo *__xThreadInfo,
			unsigned char __ucEdgeId, unsigned char __ucTileId,
			unsigned char __ucOrientation);__device__
	unsigned char GetBondingTileOrientation(xThreadInfo *__xThreadInfo,
			unsigned char __ucEdgeId, unsigned char __ucTileId,
			xAssemblyFlags *__xAssemblyFlags);__device__
	short get_xLength(xThreadInfo *__xThreadInfo, unsigned short __sEdgeId);__device__
	void __DEBUG_CALL(xThreadInfo *__xThreadInfo);
};

struct xFitnessGrid {
//#ifndef __NON_FERMI
	//Note: could do better coalesced memory-mapping here maybe
//	const xCell grid[m_fit_DimGridX][m_fit_DimGridY];
//#else
	//texture<unsigned char, 2> grid;
//#endif
	__device__ unsigned char get_xCell(unsigned char i, unsigned char j);
};

struct xAssembly {
	//union {
	struct {
		xCellGrid grid;
		xEdgeSort edgesort;
		xMoveList<uchar2> movelist;
		xAssemblyFlags flags[mWarpSize];
		curandState *states[mWarpSize];
		unsigned int synccounter[mWarpSize]; //Will be used to synchronize between Warps
		//unsigned char one_d[sizeof(xCellGrid) + sizeof(xEdgeSort) + sizeof(xMoveList<uchar2>) + sizeof(xAssemblyFlags) + sizeof(4)];
	} data;
	__device__	void Initialise(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__	bool Assemble(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__	bool Assemble_PreProcess(xThreadInfo *__xThreadInfo,
			xGenomeSet *__xGenomeSet);
	__device__	bool Assemble_PostProcess(xThreadInfo *__xThreadInfo,
			xGenomeSet *__xGenomeSet);
	__device__	bool Assemble_Movelist(xThreadInfo *__xThreadInfo,
					xGenomeSet *__xGenomeSet);
	__device__	bool Assemble_InPlace(xThreadInfo *__xThreadInfo, xGenomeSet *__xGenomeSet);
	__device__  float fEvaluateFitness(xThreadInfo *__xThreadInfo,
			xFitnessGrid *__fFitnessGrid, bool __bSingleBlockId);
	__device__	float fEvaluateFitnessForSingleGrid(xThreadInfo *__xThreadInfo,
			xFitnessGrid *__xSingleFitnessGrid, bool __bIsSingleBlock);
	__device__	bool bSynchronizeBank(xThreadInfo *__xThreadInfo);
};

//DEVICE HEADER STOP  ------------------------------------------------------------------------------

//DEVICE CODE START --------------------------------------------------------------------------------

__device__ void THROW_ErrorHeader(xThreadInfo *__xThreadInfo) {
#ifndef __NON_FERMI
	#ifdef __PRINTF__
	printf("ERROR [FBlck:%d,FThrd:%d,WId:%d,BId:%d]: ",
			__xThreadInfo->FlatBlockId(), __xThreadInfo->FlatThreadId(),
			__xThreadInfo->WarpId(), __xThreadInfo->BankId());
	#endif
#endif
}

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

__device__ bool xAssembly::Assemble(xThreadInfo *__xThreadInfo,
		xGenomeSet *__xGenomeSet) {
	bool TmpFlag = false;
	this->data.flags[__xThreadInfo->WarpId()].ClearAll();
	TmpFlag = true; //this->Assemble_PreProcess(__xThreadInfo, __xGenomeSet);
	if (TmpFlag) {
		//this->data.edgesort.Initialise(__xThreadInfo, __xGenomeSet); //TEST
		this->data.edgesort.__DEBUG_CALL(__xThreadInfo);
		//this->Assemble_PostProcess(__xThreadInfo, __xGenomeSet);
	/*	if (TmpFlag) {
			for (int i = 0; (i < mNrRedundancyAssemblies)
					&& (!this->data.flags[__xThreadInfo->WarpId()].get_bUNDCondition()); i++) {
				this->Initialise(__xThreadInfo, __xGenomeSet); //Empty out assembly grid at red
				//this->data.grid.print(__xThreadInfo);//DBG
				bool TmpController = this->Assemble_Movelist(__xThreadInfo,
						__xGenomeSet);
				//this->data.grid.print(__xThreadInfo, __xGenomeSet); //DBG
				if (!TmpController)
					TmpController = this->Assemble_InPlace(__xThreadInfo,
							__xGenomeSet);
				if (!TmpController) {
					// Both assembly processes did not finish! (should NEVER happen)
#ifndef __NON_FERMI
					THROW_ErrorHeader(__xThreadInfo);
					#ifdef __PRINTF__
					printf(
							" Assemble - Both assembly processes did not finish! (FATAL ERROR)\n");
					#endif
#endif
					return false; //Always false - indicate assembly did not finish properly (should not happen!)
				}
				this->data.flags[__xThreadInfo->WarpId()].set_Red(i); //Choose next assembly step!
			}
			return true; //Always true - i.e. indicate assembly did finish (can still be UND, though)
		} else {
			return false; //Indicates that processing before assembly returned either single block, or UND
		}*/
	} else {
		return false; //Indicates that processing before assembly returned either single block, or UND
	
	}
}

__device__ bool xAssembly::Assemble_PreProcess(xThreadInfo *__xThreadInfo,
		xGenomeSet *__xGenomeSet) {
	unsigned char TmpSameCounter = 0;
/*
	//NOTE: This should work, however, not clear how to communicate that single tile without initialisation of grid!
	//Check if starting tile is not empty
	for (int j = 0; j < mNrTileOrientations; j++) {
		if (__xGenomeSet->get_xEdgeType(__xThreadInfo, mTileIndexStartingTile,
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
						__xGenomeSet->set_EdgeType(__xThreadInfo, i, l, 0);
					}
				}
			}
		}
	}
	return true;
*/
}


__device__ void xCell::set_Orient(unsigned char __uiOrient) {
	__uiOrient = __uiOrient % mNrTileOrientations;
	//unsigned char DBGVAL1 = this->data & (255-3);
	//unsigned char DBGVAL2 = __uiOrient;
	//unsigned char DBGVAL3 = this->data & (255-3) + __uiOrient;
	//I THINK THIS FUNCTION DOES NOT WORK!
	//this->data = ((this->data & 0b11111100) + __uiOrient);
	this->data = ((this->data & (255-3) ) + __uiOrient);
}

__device__ void xCell::set_Type(unsigned char __uiType) {
#ifndef __NON_FERMI
	#ifdef __PRINTF__
	if (__uiType > 63) {
		printf("xCell: TileType exceeded 63 limit!\n");
	}
	#endif
#endif
	//this->data = (this->data & 0b11) + (__uiType << 0b10);
	this->data = (this->data & 3) + (__uiType << 2);
}

__device__ void xCell::set_xCell(unsigned char __ucVal) {
	this->data = __ucVal;
}

__device__ unsigned char xCell::get_xType(void) {
	//return this->data >> 0b00000010;
	return this->data >> 2;
}

__device__ unsigned char xCell::get_xOrient(void) {
	//return (this->data & 0b00000011);
	return (this->data & 3);
}

__device__ unsigned char xCell::get_xCell(void) {
	return this->data;
}


__device__ void xEdgeSort::Initialise(xThreadInfo *__xThreadInfo,
		xGenomeSet *__xGenomeSet, short __sEdgeId) {

#ifndef m_fit_MULTIPLE_WARPS
	if (__sEdgeId == -1) {
		for (int k = 0; k < mNrEdgeTypes; k++) {
			this->set_xLength(__xThreadInfo, k, 0);
			bool r_bEdgeAdded = false;
			for (int i = 0; i < mNrTileTypes; i++) { /*Traverse all TileTypes*/
				for (int j = 0; j < mNrTileOrientations; j++) { /*Traverse all Orientations*/
					unsigned char DBGVAL = __xGenomeSet->get_xEdgeType(
							__xThreadInfo, i, j);
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
#else
#error "MULTIPLE THREADS NOT YET IMPLEMENTED"
#endif
}

__device__ ucCELL xEdgeSort::GetBondingTile(xThreadInfo *__xThreadInfo,
		short __sEdgeId, curandState *__xCurandState,
		xAssemblyFlags *__xAssemblyFlags) {
	//Takes: Edge Type to which the tile should bond, FitFlags which will be set according to UND conditions
	//Returns: Cell of Bonding Tile type which is rotated such that the bonding tile is facing NORTH (0),
	//If nothing bonds, will return mEMPTY_CELL instead.
	if (this->get_xLength(__xThreadInfo, __sEdgeId) == 1) {
		xCell TmpCell;
		unsigned char DBGVAL2, DBGVAL3, DBGVAL = GetBondingTileOrientation(
				__xThreadInfo, __sEdgeId, 0, __xAssemblyFlags);
		//TmpCell.set_Orient((GetBondingTileOrientation(__xThreadInfo, __sEdgeId, 0, __xAssemblyFlags)+mNrTileOrientations/2)%mNrTileOrientations);
		//TmpCell.set_Orient(4-GetBondingTileOrientation(__xThreadInfo, __sEdgeId, 0, __xAssemblyFlags));
		unsigned char TmpBondBuffer = GetBondingTileOrientation(__xThreadInfo,
				__sEdgeId, 0, __xAssemblyFlags);
		TmpCell.set_xCell(4 - TmpBondBuffer);
		//DBGVAL3 = TmpCell.get_xOrient();
		//DBGVAL2 = this->get_xData(__xThreadInfo, __sEdgeId, 0, TmpCell.get_xOrient());
		TmpCell.set_Type(this->get_xData(__xThreadInfo, __sEdgeId, 0,
				TmpBondBuffer)); //TEST (0 anstelle TmpCell.get_xOrient()) b-fore
		//this->__DEBUG_CALL(__xThreadInfo);
		return TmpCell.get_xCell();
	} else if (this->get_xLength(__xThreadInfo, __sEdgeId) == 0) {
		return mEMPTY_CELL;
	} else {
		__xAssemblyFlags->set_TrivialUND();
		return mEMPTY_CELL;
	}
}

__device__ void xEdgeSort::add_TileOrient(xThreadInfo *__xThreadInfo,
		unsigned char __ucEdgeId, unsigned char __ucOrient,
		unsigned char __ucTileType) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
	this->data.multi_d[__ucEdgeId][this->get_xLength(__xThreadInfo, __ucEdgeId)][__ucOrient][__xThreadInfo->WarpId()]
			= __ucTileType;
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}

__device__ __forceinline__ void xEdgeSort::set_xLength(
		xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId,
		unsigned char __ucLength) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
	this->length.multi_d[__ucEdgeId][__xThreadInfo->WarpId()] = __ucLength;
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}

__device__ void xEdgeSort::add_Tile(xThreadInfo *__xThreadInfo,
		unsigned char __ucEdgeId) {

	this->set_xLength(__xThreadInfo, __ucEdgeId, this->get_xLength(
			__xThreadInfo, __ucEdgeId) + 1);
}

__device__ unsigned char xEdgeSort::get_xData(xThreadInfo *__xThreadInfo,
		unsigned char __ucEdgeId, unsigned char __ucTileId,
		unsigned char __ucOrientation) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
	unsigned char
			DBGVAL =
					this->data.multi_d[__ucEdgeId][__ucTileId][__ucOrientation][__xThreadInfo->WarpId()];
	return this->data.multi_d[__ucEdgeId][__ucTileId][__ucOrientation][__xThreadInfo->WarpId()];
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}

__device__ unsigned char xEdgeSort::GetBondingTileOrientation(
		xThreadInfo *__xThreadInfo, unsigned char __ucEdgeId,
		unsigned char __ucTileId, xAssemblyFlags *__xAssemblyFlags) {
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

__device__ short xEdgeSort::get_xLength(xThreadInfo *__xThreadInfo,
		unsigned short __sEdgeId) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
	if (__sEdgeId < mNrEdgeTypes) {

		return this->length.multi_d[__sEdgeId][__xThreadInfo->WarpId()];
	} else {
#ifndef __NON_FERMI
	#ifdef __PRINTF__

		THROW_ErrorHeader(__xThreadInfo);
		printf(
				" get_xLength in xEdgeSort was called with EdgeId out of bounds!\n");
	#endif
#endif
		return 0;
	}
#else
#error "Safe Memory Mapping not implemented yet!"
#endif
}

__device__ unsigned char xGenome::get_xEdgeType(xThreadInfo *__xThreadInfo,
		unsigned char __ucTileId, unsigned char __ucEdgeId) {
	if (__ucTileId < mNrTileTypes) {
		unsigned short TmpStartBit = __ucTileId * mBitLengthEdgeType
				* mNrTileOrientations + __ucEdgeId * mBitLengthEdgeType;
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
			TmpRetVal
					+= mOneOrNil(mBitTest(TmpBitOffset, this->data.one_d[TmpByteOffset]))
							<< (mBitLengthEdgeType - 1 - j); //(mFFOrNIL(mBitTest(TmpBitOffset, this->data.one_d[TmpByteOffset])) * 0x01) << (mBitLengthEdgeType-1-j);
			j++;
		}
		return TmpRetVal;
	} else {
#ifndef __NON_FERMI
		THROW_ErrorHeader(__xThreadInfo);
		#ifdef __PRINTF__
		printf("Genome was accessed with EdgeType out-of range (%d)!\n",
				__ucTileId);
		#endif
#endif
		return (ucEDGETYPE) 0x00;
	}
}


__device__ void xGenome::CopyFromGlobal(xThreadInfo __xThreadInfo,
		unsigned char *__g_ucGenomes) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
#ifndef m_fit_MULTIPLE_WARPS
	if(!threadIdx.x) __xThreadInfo.__DEBUG_CALL();
	for (int i = 0; i < mAlignedByteLengthGenome; i += 1) {
		printf("NExT GENOME LINE: %d\n", __xThreadInfo.GlobId(sizeof(xGenome)) + i);
		this->data.one_d[i] = __g_ucGenomes[__xThreadInfo.GlobId(sizeof(xGenome)) + i];
		if(!threadIdx.x) printf("BYTE %d:\n", i, this->data.one_d[i]);
		//(*reinterpret_cast<int*> (&this->data.one_d[i]))
		//		= (*reinterpret_cast<int*> (&__g_ucGenomeSet[__xThreadInfo.GlobId(
		//				sizeof(xGenome)) + i]));
	}
#else
#error "MULTIPLE THREADS NOT YET IMPLEMENTED"
#endif
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}


__device__ unsigned char xGenomeSet::get_xEdgeType(xThreadInfo *__xThreadInfo,
		unsigned char __ucTileId, unsigned char __ucEdgeId) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
	return this->data.multi_d[__xThreadInfo->WarpId()].get_xEdgeType(
			__xThreadInfo, __ucTileId, __ucEdgeId);
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}

__device__ void xGenomeSet::CopyFromGlobal(xThreadInfo __xThreadInfo,
		unsigned char *__g_ucGenomes) {
#ifdef m_fit_SAFE_MEMORY_MAPPING
#ifndef m_fit_MULTIPLE_WARPS
	//if(!threadIdx.x) __xThreadInfo.__DEBUG_CALL(); //printf("Initialising Genome...\n");
	this->data.multi_d[__xThreadInfo.WarpId()].CopyFromGlobal(__xThreadInfo,
			__g_ucGenomes);
#else
#error "MULTIPLE THREADS NOT YET IMPLEMENTED"
#endif
#else
#error "SAFE MEMORY MAPPING NOT IMPLEMENTED YET!"
#endif
}

__device__ void xEdgeSort::__DEBUG_CALL(xThreadInfo *__xThreadInfo) {
#ifndef __NON_FERMI
	//#ifdef __PRINTF__
	printf("DBG CALL - xEdgeSort:");
	for (int i = 0; i < mNrEdgeTypes; i++) {
		printf("Edge: %d\n", i);
		for (int j = 0; j < this->get_xLength(__xThreadInfo, i); j++) {
			printf("[");
			for (int k = 0; k < mNrTileOrientations; k++) {
				//printf("%d,",
				//		this->data.multi_d[i][j][k][__xThreadInfo->WarpId()]);
			}
			printf("]");
		}
		printf("\n");
	}
	//#endif
#endif
}

__device__ void xGenomeSet::print(xThreadInfo *__xThreadInfo) {
//	#ifdef __PRINTF_i_
	//if(!threadIdx.x) __xThreadInfo.__DEBUG_CALL();
	for (int i = 0; i < mNrTileTypes; i++) {
		printf("[");
		for (int j = 0; j < mNrTileOrientations; j++) {
			printf("%d,", this->get_xEdgeType(__xThreadInfo, i, j));
		}
		printf("], ");
	}
//	#endif
}
__global__ void FitnessKernel(unsigned char *g_ucGenomes, float *g_fFFValues, unsigned char *g_ucAssembledGrids, curandState *g_xCurandStates, xMutex *g_xMutexe) {
	#ifdef __PRINTF__
	printf("Entering Fitness Kernel!\n");
	#endif

	if(threadIdx.x==0) printf("HELLO! \n");

	__shared__ xEdgeSort s_xEdgeSort;
	__shared__ xGenomeSet s_xGenomeSet;
	__shared__ xAssembly s_xAssembly;
	xThreadInfo r_xThreadInfo(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

	//if(!threadIdx.x) r_xThreadInfo.__DEBUG_CALL();
	printf("EHLLO!\n");
	
	s_xGenomeSet.CopyFromGlobal(r_xThreadInfo, g_ucGenomes);

	//s_xGenomeSet.print(&r_xThreadInfo);
	#ifdef __PRINTF__
	printf("\n\n");
	#endif
	//s_xEdgeSort.Initialise(&r_xThreadInfo, &s_xGenomeSet);
	//s_xAssembly.Assemble(&r_xThreadInfo, &s_xGenomeSet); //TEST

	//s_xGenomeSet.CopyToGlobal(&r_xThreadInfo, g_ucGenomes);

	//s_xAssembly.data.grid.print(&r_xThreadInfo, &s_xGenomeSet);

	//if(r_xThreadInfo.GlobId(1)==0) s_xEdgeSort.__DEBUG_CALL(&r_xThreadInfo);
}

//DEVICE CODE STOP  --------------------------------------------------------------------------------,
