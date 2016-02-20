typedef int xMutex;
typedef unsigned char ucTYPELENGTH; 
typedef unsigned short usSHARED1D;
typedef unsigned int usGLOBAL1D;

enum Params {
    eNrGenomes,
    eNrGenerations,
    eNrTileTypes,
    eNrEdgeTypes,
    eByteLengthGenome,
    eBitLengthGenome,
    eEdgeTypeBitLength,
    eNrTileOrientations
};

#define mXOR(a, b) (((a)&~(b))|(~(a)&(b)))

#define mBLOCK_ID blockIdx.x
#define mTHREAD_ID_X threadIdx.x
#define mTHREAD_ID_Y threadIdx.y
#define mTHREAD_ID threadIdx.x
#define mNrMemoryBanks warpSize

#define mByteLengthGenome_c c_fParams[eByteLengthGenome]
#define mNrGenomes_c c_fParams[eNrGenomes]
#define mNrGenomes {{ glob_nr_genomes }}
#define mBitLengthGenome_c c_fParams[eBitLengthGenome]
#define mBitLengthEdgeType_c c_fParams[eEdgeTypeBitLength]
#define mNrTileTypes_c c_fParams[eNrTileTypes]
#define mNrEdgeTypes_c c_fParams[eNrEdgeTypes]
#define mNrTileOrientations_c c_fParams[eNrTileOrientations]
                
#define mNrTileTypes {{ glob_nr_tiletypes }}
#define mNrEdgeTypes {{ glob_nr_edgetypes }}
#define mNrTileOrientations {{ glob_nr_tileorientations }}
#define mByteLengthGenome {{ aligned_byte_length_genome }}
#define mBitLengthGenome {{ genome_bitlength }}
#define mBitLengthEdgeType {{ glob_bitlength_edgetype }}

/*This file contains all template macros for global simulation - Copyright Christian Schroeder, Oxford University 2012*/

//#define SAFE_MEMORY_MAPPING {{ safe_memory_mapping }}
#define mAlignedByteLengthGenome {{ aligned_byte_length_genome }}
#define mWarpSize {{ warpsize }}
#define mBankSize {{ banksize }}
//#define mNrTileOrientations 4
//#define mBitLengthEdgeType {{ bit_length_edge_type }}
//#define mNrTileTypes {{ nr_tile_types }}
//#define mNrEdgeTypes {{ nr_edge_types }}

//#define EMPTY_TILETYPE 63
