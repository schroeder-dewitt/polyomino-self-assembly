typedef unsigned char ucSUBGRIDCYCLEINDEX;
typedef unsigned char ucEDGECYCLEINDEX;
typedef unsigned char ucEDGETYPE;
typedef unsigned char ucTILETYPE;
typedef unsigned char ucCELL;
typedef unsigned char ucORIENTATION;
typedef unsigned char ucGENOME; //[mByteLengthGenome];
typedef unsigned char ucOFFSET;
typedef enum {u_eNORTH, u_eEAST, u_eSOUTH, u_eWEST} ucDIRECTION;

#define mEMPTY_CELL 255
#define mEMPTY_COORD 255

//First oder functions
#define mWarpSize ((int){{ WarpSize }})
#define mNrWarps ((int){{ fit_dimthreadx * fit_dimthready / WarpSize  }})
#define mNrSubgridsPerGridX ((int){{ fit_dimgridx / fit_dimsubgridx }})
#define mNrSubgridsPerGridY ((int){{ fit_dimgridy / fit_dimsubgridy  }})

/*TESTED - NEW*/
__forceinline__ __device__ ucEDGETYPE fit_xGetUnrotatedEdgeCycleIndex (ucORIENTATION orient, ucEDGECYCLEINDEX eindex){
    return (int) ( mNrTileOrientations - (orient - eindex) ) % (int) mNrTileOrientations;
}

/*TESTED*/
__forceinline__ __device__ ucTILETYPE fit_xGetTileTypeFromCell (ucCELL cell){
    return cell >> 2;
}

/*TESTED*/
__forceinline__ __device__ ucORIENTATION fit_xGetOrientationFromCell (ucCELL cell){
    return cell & 3;
}

/*TESTED*/
__forceinline__ __device__ ucCELL fit_xCell (ucTILETYPE tiletype, ucORIENTATION orientation){
    return (tiletype << 2) + (orientation);
}

/*ADAPTED*/
__forceinline__ __device__ usGLOBAL1D fit_xGlobalBlockAnchor(unsigned int BankID, ucTYPELENGTH __typelength){ // This is 1D index 0 of a block in any block in global memory of type of length
    return __typelength * mBLOCK_ID * WarpSize + BankID * __typelength;
}

/*TESTED*/
__forceinline__ __device__ usGLOBAL1D fit_xGlobalThreadAnchor(ucTYPELENGTH __typelength){ // This is 1D index 0 of a thread in any block in global memory of type of length
    return  ( (int) __typelength * (mBLOCK_ID * m_fit_THREAD_DIM_X * m_fit_THREAD_DIM_Y + mTHREAD_ID_X * m_fit_THREAD_DIM_Y + mTHREAD_ID_Y ) );
}

/*ADAPTED, UNTESTED*/
__forceinline__ __device__ us2GRID2D fit_xGridThreadAnchor(unsigned short WarpID){ // This is subgrid pixel 0 position in Grid
    //Note: mNrSubgridsPerGridX, mNrSubgridsPerGridY must both be integer and also, NrSubgridsPerGridX*NrSubgridsPerGridY must be equal to number of threads per assembly (8 typically)
    unsigned int xcoord = WarpID % mNrSubgridsPerGridX;
    unsigned int ycoord = (WarpID - xcoord) / mNrSubgridsPerGridY;
    return make_uchar2(xcoord, ycoord);
}

/*TESTED*/
__forceinline__ __device__ us2GRID2D fit_xGridPixelAnchor(unsigned short WarpID, ucSUBGRIDCYCLEINDEX ucSubgridCycleIndex){ // This is subgrid pixel position in Grid
    uchar2 __ret = fit_xGridThreadAnchor(WarpID);
    unsigned short buf = ucSubgridCycleIndex % (int) m_fit_DimSubgridX;
    __ret.x += buf;
    __ret.y += (ucSubgridCycleIndex  - buf) / m_fit_DimSubgridX;
    return __ret;
}

/*UNTESTED*/
__forceinline__ __device__ unsigned int uiBankMapChar(unsigned int id_in_bank, unsigned int bankid){
    unsigned int four_offset = id_in_bank % 4; //Nr of chars per bank
    unsigned int OrthoID = (id_in_bank - id_in_bank % 4) / 4; //Nr of chars per bank
    return  OrthoID * WarpSize * 4 + bankid * 4 + four_offset; //Nr of chars per bank
    //return OrthoID; //Nr of chars per bank //TEST
}

//UNSURE!
__forceinline__ __device__ unsigned int uiBankMapuchar2(unsigned int id_in_bank, unsigned int bankid){
    unsigned int four_offset = id_in_bank % 2; //Nr of shorts per bank
    /*unsigned int id_in_bank_clean = (id_in_bank - four_offset) % 2; //Nr of shorts per bank
    return WarpSize * 2 * id_in_bank_clean + bankid * 2 + four_offset; //Nr of shorts per bank*/
    unsigned int OrthoID = (id_in_bank - id_in_bank % 2) / 2;
    return OrthoID * WarpSize * 2 + bankid * 2 + four_offset; 
}

/*UNTESTED*/
extern "C++" __forceinline__ __device__ ucTILETYPE* fit_xMap (unsigned int bankid, unsigned int edgeid, unsigned int tileid, unsigned int orientid , ucTILETYPE (&s_ucEdgeSort)[mNrEdgeTypes * mNrTileTypes * mNrTileOrientations * WarpSize]){
    unsigned int id_in_bank = edgeid * (mNrTileTypes * mNrTileOrientations) + tileid * (mNrTileOrientations) + orientid;
    return &s_ucEdgeSort[uiBankMapChar(id_in_bank, bankid)]; //Nr of chars per bank
}

/*UNTESTED*/
extern "C++" __forceinline__ __device__ ucCELL* fit_xMap (unsigned int bankid, unsigned int assid, signed int x, signed int y , ucCELL (&s_ucGrid)[2 * m_fit_DimGridX * m_fit_DimGridY * WarpSize]){
    unsigned int id_in_bank = assid*(m_fit_DimGridY*m_fit_DimGridX) + y*(m_fit_DimGridX) + x;
    //TEST START
    //return &s_ucGrid[id_in_bank];
    //TEST STOP
    return &s_ucGrid[uiBankMapChar(id_in_bank, bankid)]; //Nr of chars per bank
}

/*UNTESTED*/
extern "C++" __forceinline__ __device__ ucGENOME* fit_xMap (unsigned int bankid, unsigned int byteid, ucGENOME (&s_ucGenome)[mByteLengthGenome * WarpSize]){
    unsigned int id_in_bank = byteid;
    return &s_ucGenome[uiBankMapChar(id_in_bank, bankid)];
}

/*UNTESTED*/
extern "C++" __forceinline__ __device__ uchar2* fit_xMap (unsigned int bankid, unsigned int item_id, uchar2 (&s_s2MoveListStorage)[WarpSize *  m_fit_LengthMovelist]){
    unsigned int id_in_bank = item_id;
    return &s_s2MoveListStorage[uiBankMapuchar2(id_in_bank, bankid)];
}

/*UNTESTED*/
__forceinline__ __device__ unsigned char fit_xFFOrNIL(unsigned char param){
    return param ? 0xFF : 0x00;
}

/*TESTED*/
__forceinline__ __device__ unsigned char fit_ucBitTest(unsigned short index, unsigned char byte){
    return ( (unsigned short) byte & ( 0x01 << index ) ) >> index ;
}

/*ADAPTED - UNTESTED*/
__forceinline__ __device__ ucEDGETYPE fit_xGetEdgeTypeFromGenome(unsigned char tileindex, unsigned char edgeindex, ucGENOME (&s_ucGenome)[mByteLengthGenome * WarpSize], unsigned short bankid){
    if( tileindex < mNrTileTypes){
        unsigned short start_bit_address = (tileindex) * mBitLengthEdgeType * mNrTileOrientations + edgeindex * mBitLengthEdgeType;
        unsigned short end_bit_address = start_bit_address + mBitLengthEdgeType;
        unsigned char retval = 0;
        unsigned short byte_offset = 0;
        unsigned short bit_offset = 0;
        /*Note: This could be speeded up by copying all bits within a byte simultaneously*/
        unsigned short j = 0;
        for(int i = start_bit_address; i < end_bit_address; i++){
            bit_offset =  i % 8 ;//sizeof(char); //We need to invert index as we start from left to right
            byte_offset = (i - bit_offset) / 8;
            bit_offset = 7 - bit_offset;
            retval += ( fit_xFFOrNIL( fit_ucBitTest( bit_offset, *fit_xMap( bankid, byte_offset, s_ucGenome ) ) ) & 0x01 ) << ( mBitLengthEdgeType - 1 - j);
            j++;
        }
        return retval;
    } else return (ucEDGETYPE) 0x00;
}

//ADAPTED - UNTESTED
__forceinline__ __device__ us2GRID2D fit_xGetNeighbourCellCoords(us2GRID2D CurrentPos, unsigned int direction){
    us2GRID2D buf = CurrentPos;
    switch(direction){
        case 0://u_eEAST:
            if( (buf.x+1) < (m_fit_DimGridX) ){
                buf.x++;
                return buf;
            }
            else{
                /*Do nothing: let fall through*/
            }
            break;
        case 1://u_eWEST:
            if( (buf.x-1) >= 0 ){
                buf.x--;
                return buf;
            }
            else{
                /*Do nothing: let fall through*/
            }
            break;
        case 2://u_eSOUTH:
            if( (buf.y+1) < (m_fit_DimGridY) ){
                buf.y++;
                return buf;
            }
            else{
                /*Do nothing: let fall through*/
            }
            break;
        case 3://u_eNORTH:
            if( (buf.y-1) >= 0 ){
                buf.y--;
                return buf;
            }
            else{
                /*Do nothing: let fall through*/
            }
            break;
        default:
        break;
    }
    return make_uchar2( mEMPTY_COORD, mEMPTY_COORD); //Cell out-of-bounds
}

/*ADAPTED - UNTESTED*/
__forceinline__ __device__ ucCELL fit_xGetNeighbourCell(unsigned int BankID, us2GRID2D CurrentPos, unsigned int dir, unsigned int RedID, ucCELL (&s_ucGrid)[2 * m_fit_DimGridX * m_fit_DimGridY * WarpSize]){
    if(threadIdx.x==0) printf("-----> Cycle index: %d\n", dir);
    us2GRID2D buf = fit_xGetNeighbourCellCoords(CurrentPos, dir);
    if(threadIdx.x==0) printf("-----> Neighbour Cell coords: %d, %d\n", buf.x, buf.y);
    if( (buf.x >= m_fit_DimGridX) && (buf.y >= m_fit_DimGridY) ){ //TEST
        if(threadIdx.x==0) printf("----->Index (%d, %d) out of grid range!\n", buf.x, buf.y);
        return 255;
    }
    return (*fit_xMap (BankID, RedID, buf.x, buf.y, s_ucGrid)); //TEST
}

/*TESTED - NEW*/
__forceinline__ __device__ ucEDGETYPE fit_xGetEdgeTypeFromCell (ucCELL cell, ucEDGECYCLEINDEX eindex,  ucGENOME (&s_ucGenome)[mByteLengthGenome * WarpSize], unsigned short BankID){
    return fit_xGetEdgeTypeFromGenome( fit_xGetTileTypeFromCell(cell) ,
                                       fit_xGetUnrotatedEdgeCycleIndex( fit_xGetOrientationFromCell(cell) , eindex),
                                       s_ucGenome, BankID );
}

struct xFitFlags{
    bool UNDCondition;
    unsigned char BusyFlag;
    unsigned short RedID;
};


extern "C++" {
    template <class T>
    class LifoList{ // NOT REENTRANT
            public:
            signed short pos; //Current position of top element (0...max_length-1)
            unsigned short max_length; //Max length of entry list

            __device__ LifoList(void){
                this->pos = -1;
                this->max_length = 0;
            }
    
            __device__ LifoList(unsigned short max_length){
                this->pos = -1;
                this->max_length = max_length;
            }
            __device__ ~LifoList(){}
    
            __device__ bool push(T entry, uchar2 (&s_s2MoveListStorage)[WarpSize *  m_fit_LengthMovelist], unsigned short BankID){
                this->pos++;
                if(this->pos < this->max_length){
                    //this->entries[this->pos] = entry;
                    s_s2MoveListStorage[uiBankMapuchar2(this->pos, BankID)] = entry;
                    return true;
                } else {
                    this->pos--;
                    return false;
                }
            }
            __device__ T pop(uchar2 (&s_s2MoveListStorage)[WarpSize *  m_fit_LengthMovelist], unsigned short BankID){
                if(this->pos < 0){
                    //NOTE: ADAPTED FOR uchar2 ONLY!
                    uchar2 buf;
                    buf.x = mEMPTY_CELL;
                    buf.y = mEMPTY_CELL;
                    return buf; //We will define EMPTY as (255, 255) later on...
                } else {
                    this->pos--;
                    //return this->entries[this->pos+1];
                    return s_s2MoveListStorage[uiBankMapuchar2(this->pos+1, BankID)];
                }
            }
    };
}

/*TESTED*/
__forceinline__ __device__ bool fit_bIsBorderCell(us2GRID2D coords){
    if( ( ( coords.x == 0 ) || ( coords.x == m_fit_DimGridX - 1 ) ) || ( ( coords.y == 0 ) || ( coords.y == m_fit_DimGridY - 1 ) ) ){
        return true;
    } else return false;
}

/*TESTED*/
__forceinline__ __device__ bool fit_bIsCentralCell(us2GRID2D coords){
    if( (coords.x == m_fit_DimGridX/2) && (coords.y == m_fit_DimGridY/2) ){
        return true;
    } else return false;
}
