{% include "./EdgeSortTest/beta.cu" %}
{% include "./MovelistTest/beta.cu" %}

__constant__ unsigned char c_ucFourPermutations[{{ fit_nr_four_permutations }}][{{ glob_nr_tile_orientations }}];

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
        //short one_d_off = m_fit_DimGridX*m_fit_DimGridY*m_fit_NrRedundancyAssemblies*__xThreadInfo.BankId() + m_fit_DimGridX*m_fit_DimGridY*__red; 
        //_fit_DimGridX*m_fit_DimGridY;

        for(int i=0;i<myshare;i++){
                this->data.mix_d[__xThreadInfo.WarpId()*myshare + i][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL); 
        }
        if(__xThreadInfo.WarpId()==mBankSize-1){
                for(int i=0;i<offset;i++){
                        //this->data.one_d[one_d_off + mBankSize*myshare + i].set_xCell(mEMPTY_CELL);
                        this->data.mix_d[mBankSize*myshare + i][__red][__xThreadInfo.BankId()].set_xCell(mEMPTY_CELL);
                }
        }
        //__syncthreads(); 
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
                float2 gravity[mWarpSize];
                int assembly_size[mWarpSize];
                int left[mWarpSize];
                int bottom[mWarpSize];
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

__device__ float xAssembly::fEvaluateFitness(xThreadInfo __xThreadInfo){
	if(this->data.flags[__xThreadInfo.BankId()].get_bUNDCondition()){
            this->data.left[__xThreadInfo.BankId()] = 33;
        } else {
            this->data.left[__xThreadInfo.BankId()] = 11;
        }
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
                //if(__xThreadInfo.WarpId() == 0){
              		this->data.edgesort.Initialise(__xThreadInfo, __xGenomeSet); //TEST
                //}
                //__syncthreads();
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
                                //__syncthreads(); //TEST
                                //this->Initialise(__xThreadInfo, __xGenomeSet); //Empty out assembly grid at red //TEST
                                //return true; //TEST
                                //__syncthreads();
                                //__syncthreads();
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
        }*/
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


__global__ void SearchSpaceKernel(unsigned char *g_ucGenomes,  unsigned char *g_ucGrids, float *g_ucFitnessLeft, float *g_ucFitnessBottom,  curandState *states)
{
    __shared__ xGenomeSet s_xGenomeSet;
    //__shared__ xEdgeSort s_xEdgeSort;
    __shared__ xAssembly s_xAssembly;
    s_xAssembly.data.states = states;
    xThreadInfo r_xThreadInfo(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    //s_xGenomeSet.CopyFromGlobal(r_xThreadInfo, g_ucGenomes);
    //s_xEdgeSort.Initialise(r_xThreadInfo, &s_xGenomeSet, -1);

    /*s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 44; //132; // 40; //33;//4; //33; //40
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[1] = 181; //6; //162; //160 //128; //0;//32;//16;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[2] = 20; //2; //138; //138;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[3] = 0;*/

    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 41; //132; // 40; //33;//4; //33; //40
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[1] = 192; //6; //162; //160 //128; //0;//32;//16;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[2] = 0; //2; //138; //138;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[3] = 0;

    s_xAssembly.Assemble(r_xThreadInfo, &s_xGenomeSet);
    //__syncthreads();
    s_xAssembly.fEvaluateFitness(r_xThreadInfo);

    if(r_xThreadInfo.WarpId()==0){
        //g_ucFitnessLeft[r_xThreadInfo.GlobId(1)] = 3; //s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
        g_ucFitnessLeft[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.left[r_xThreadInfo.BankId()];//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;//(float)  s_xAssembly.data.assembly_size[r_xThreadInfo.BankId()];//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
//s_xAssembly.data.left[r_xThreadInfo.BankId()];//(int) s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
// (blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId();//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
        g_ucFitnessBottom[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.bottom[r_xThreadInfo.BankId()]; //s_xAssembly.data.gravity[r_xThreadInfo.BankId()].y;
//s_xAssembly.data.bottom[r_xThreadInfo.BankId()];//(int) s_xAssembly.data.gravity[r_xThreadInfo.BankId()].y;
    //g_ucFitnessBottom[462] = 7;
    }

    //for(int i=0;i<4;i++){
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.length.multi_d[i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.data.multi_d[6][0][i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = tex2D(t_ucInteractionMatrix, i, 1);
    //}
    //s_xGenomeSet.CopyToGlobal(r_xThreadInfo, g_ucGenomes);

    //Copy to grid
    for(int i=0;i<m_fit_DimGridY;i++){
         for(int j=0;j<m_fit_DimGridX;j++){
             xCell TMP = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0);
             g_ucGrids[r_xThreadInfo.FlatBlockId()*m_fit_DimGridX*m_fit_DimGridY*32 + r_xThreadInfo.BankId()*m_fit_DimGridX*m_fit_DimGridY + j*m_fit_DimGridX + i] = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0).get_xType(); //Orient, Type, Cell
         }
    }
}

