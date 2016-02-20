/*This header file includes all template macros for the Fitness Kernel - copyright Christian Schroeder, Oxford University, 2012*/

#define m_fit_DimThreadX {{ fit_dim_thread_x }}
#define m_fit_DimThreadY {{ fit_dim_thread_y }}
#define m_fit_DimBlockX {{ fit_dim_block_x }}

#define mFFOrNil(param) (param?0xFF:0x00)
#define mOneOrNil(param) (param?0x01:0x00)
#define mBitTest(index,byte) (byte & (0x1<<index))

#define mEMPTY_CELL 255
