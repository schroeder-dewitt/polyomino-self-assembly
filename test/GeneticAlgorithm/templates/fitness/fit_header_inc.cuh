/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX {{ fit_dim_thread_x }}
#define m_fit_DimThreadY {{ fit_dim_thread_y }}
#define m_fit_DimBlockX {{ fit_dim_block_x }}

#define m_fit_DimGridX {{ fit_dim_grid_x }}
#define m_fit_DimGridY {{ fit_dim_grid_y }}

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define m_fit_LengthMovelist {{ fit_length_movelist }}
#define m_fit_NrRedundancyGridDepth {{ fit_nr_redundancy_grid_depth }}
#define m_fit_NrRedundancyAssemblies {{ fit_nr_redundancy_assemblies }}
#define m_fit_TileIndexStartingTile {{ fit_tile_index_starting_tile }}

#define mEMPTY_CELL 255
#define mEMPTY_CELL_ML 22<<2 //254
#define mEMPTY_CELL_OUT_OF_BOUNDS 253
