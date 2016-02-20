#-------------------------------------------------------------------------------
# Name:        GA test module (template hopefully)
# Purpose:
#
# Author:      Christian Schroeder, University of Oxford
#
# Created:     31/01/2012
# Copyright:   (c) CHRIS 2012
# Licence:     GPL
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import math

from pycuda.compiler import SourceModule
from bitstring import BitStream, BitArray
from jinja2 import Environment, PackageLoader

#helper functions

def bitlength(arg):
    return math.ceil( math.log( arg , 2) )

def bytelength(arg):
    return math.ceil( arg / 8.0)

#Initialise data for GPU constant memory / user-defined constants - Global
NrGenomes = 1000
NrTileTypes = 2
NrEdgeTypes = 8
NrTileOrientations = 4 # because square! don't change this... it might blow everything up.
NrGenerations = 10000

#Initialise data for GPU constant memory / user-defined constants - FitnessKernel
DimGridX = 20
DimGridY = 20
NrFitnessFunctionGrids = 5
NrAssemblyRedundancy = 10
InteractionMatrix = np.zeros( ( NrEdgeTypes, NrEdgeTypes ) ).astype(np.float32)
WithStoreAssembledGridsInGlobalMemory = 1.0

#Initialise data for Sorting Kernel in GPU
Sort_ThreadDimX = 256

#Initialise data for GPU constant memory / user-defined constants - GAKernel
RateMutation = 4.8
ProbabilityUniformCrossover = 0.2
ProbabilitySinglePointCrossover = 0.3
WithMixedCrossover = 0.0
WithBankConflict = 1
WithNaiveRouletteWheelSelection = 1
WithAssumeNormalizedFitnessFunctionValues = 1
WithUniformCrossover = 0.0
WithSinglePointCrossover = 1
WithSurefireMutation = 1

#Parameters which probably should not be changed if you don't re-structure the whole simulation
NrMutexe = 2
FourPermutations = np.array([ [1,2,3,4],
                              [1,2,4,3],
                              [1,3,2,4],
                              [1,3,4,2],
                              [1,4,2,3],
                              [1,4,3,2],
                              [2,1,3,4],
                              [2,1,4,3],
                              [2,3,1,4],
                              [2,3,4,1],
                              [2,4,1,3],
                              [2,4,3,1],
                              [3,2,1,4],
                              [3,2,4,1],
                              [3,1,2,4],
                              [3,1,4,2],
                              [3,4,2,1],
                              [3,4,1,2],
                              [4,2,3,1],
                              [4,2,1,3],
                              [4,3,2,1],
                              [4,3,1,2],
                              [4,1,2,3],
                              [4,1,3,2],]).astype(np.uint8)
NrFourPermutations = FourPermutations.shape[0] #don't change this if you don't change NrTileOrientations

#Software-Tunable parameters for CurandInitKernel
CurandInit_NrThreadsPerBlock = 256.0
CurandInit_NrBlocks = int( math.ceil( float(NrGenomes) / float(CurandInit_NrThreadsPerBlock) ) )

#Software-Tunable parameters for FitnessKernel
SubgridDimX = 2
SubgridDimY = 1
ThreadDimX = 16
ThreadDimY = 16
NrSubgridsPerBank = 13
Fit_NrThreadsPerBlock = ThreadDimX * ThreadDimY

#Software-Tunable parameters for GAKernel
ThreadDim = 256

#Initialise Parameters calculated from user-defined parameters
ByteLengthGenome = int( bytelength( 4 * bitlength(NrEdgeTypes) * NrTileTypes ) )
BitLengthGenome = int( 4 * bitlength(NrEdgeTypes) * NrTileTypes )
EdgeTypeBitLength = int( bitlength(NrEdgeTypes)  )

fit_grid = (NrGenomes, 1)
fit_blocks = (DimGridX, DimGridY, 1)

ga_grids = ( int( math.ceil( float(NrGenomes) / float(ThreadDimX) ) ), 1)
ga_blocks = (ThreadDimX, 1, 1)
GA_NrThreadsPerBlock = int( math.ceil( float(NrGenomes) / float(ThreadDimX) ) )

sorting_blocks = (Sort_ThreadDimX, 1, 1)
sorting_grids = ( int( math.ceil( float(NrGenomes) / float(Sort_ThreadDimX) ) ), 1)

#Main Simulation Params
GlobalParamsDict = {
    "NrGenomes":int(NrGenomes),
    "NrGenerations":int(NrGenerations),
    "NrTileTypes":int(NrTileTypes),
    "NrEdgeTypes":int(NrEdgeTypes),
    "ByteLengthGenome":int(ByteLengthGenome),
    "BitLengthGenome":int(BitLengthGenome),
    "EdgeTypeBitLength":int(EdgeTypeBitLength),
    "NrTileOrientations":int(NrTileOrientations)
    }

#Fitness Simulation Params
FitnessParamsDict = {
    "DimGridX":int(DimGridX),
    "DimGridY":int(DimGridY),
    "NrFitnessFunctionGrids":int(NrFitnessFunctionGrids),
    "NrThreadsPerBlock":int(fit_blocks[0]*fit_blocks[1]*fit_blocks[2]),
    "NrBlocks":int(fit_grid[0]*fit_grid[1]),
    "SubgridDimX": int(SubgridDimX),
    "SubgridDimY": int(SubgridDimY),
    "ThreadDimX": int(ThreadDimX),
    "ThreadDimY": int(ThreadDimY),
    "NrSubgridsPerBank": int(NrSubgridsPerBank),
    "NrFourPermutations": int(NrFourPermutations),
    "NrAssemblyRedundancy": int(NrAssemblyRedundancy)
    }

#Sorting Kernel Params

#GA Simulation Params
GAParamsDict = {
    "RateMutation": RateMutation, # This is Mu * L, expectation value of number of flipped bits
    "ProbabilityUniformCrossover": ProbabilityUniformCrossover, # Gives the probability that crossover is point-wise (so if zero, all cross-over will be uniform)
    "ProbabilitySinglePointCrossover": ProbabilitySinglePointCrossover, # Gives the probability that crossover is point-wise (so if zero, all cross-over will be uniform)
    "WithMixedCrossover":WithMixedCrossover,
    "WithBankConflict":WithBankConflict,
    "WithNaiveRouletteWheelSelection":WithNaiveRouletteWheelSelection,
    "WithAssumeNormalizedFitnessFunctionValues":WithAssumeNormalizedFitnessFunctionValues,
    "WithUniformCrossover":WithUniformCrossover,
    "WithSinglePointCrossover":WithSinglePointCrossover,
    "WithSurefireMutation":WithSurefireMutation,
    "NrThreadsPerBlock":GA_NrThreadsPerBlock,
    "ThreadDim":ThreadDimX,
    "WithStoreAssembledGridsInGlobalMemory":WithStoreAssembledGridsInGlobalMemory
    }

#Initialise data for GPU texture memory
FitnessFunctionGrids = np.zeros((NrFitnessFunctionGrids, DimGridX, DimGridY)).astype(np.uint8)

#Initialise data for GPU global memory
Genomes = np.zeros( ( NrGenomes, ByteLengthGenome) ).astype(np.uint8) #Genomes = np.zeros( ( NrGenomes, NrTileTypes, 4) ).astype(np.uint8)
FitnessPartialSums = np.zeros( ( sorting_grids[0] ) ).astype(np.float32)
FitnessValues = np.zeros( (NrGenomes) ).astype(np.float32)
AssembledGrids = np.zeros( ( NrGenomes, DimGridX, DimGridY ) ).astype(np.uint8)
Mutexe = np.zeros( (NrMutexe) ).astype(np.uint32)
ReductionList = np.zeros( (sorting_grids[0]) ).astype(np.float32)

#Handles for the values in global memory
FitnessFunctionGrids_h = []
Genomes_h = None
FitnessValues_h = None
AssembledGrids_h = None
GlobalParams_h = None
FitnessParams_h = None
GAParams_h = None
CurandStates_h = None
Mutexe_h = None
FourPermutations_h = None
InteractionMatrix_h = None
FitnessSumConst_h = None
FitnessListConst_h = None
ReductionList_h = None

#Storing the kernel source
KernelFile = "./kernel.cu"
KernelSourceModule = None

def main():

    #Initialise InteractionMatrix
    def Delta(a,b):
        if a==b:
            return 1
        else:
            return 0
    for i in range(InteractionMatrix.shape[0]):
        for j in range(InteractionMatrix.shape[1]):
            InteractionMatrix[i][j] = ( 1 - i % 2 ) * Delta( i, j+1 ) + ( i % 2 ) * Delta( i, j-1 )

    #Initialise GPU (equivalent of autoinit)
    drv.init()
    assert drv.Device.count() >= 1
    dev = drv.Device(0)
    ctx = dev.make_context(0)

    #Convert GlobalParams to List
    GlobalParams = np.zeros(len(GlobalParamsDict.values())).astype(np.float32)
    count = 0
    for x in GlobalParamsDict.keys():
        GlobalParams[count] = GlobalParamsDict[x]
        count += 1

    #Convert FitnessParams to List
    FitnessParams = np.zeros(len(FitnessParamsDict.values())).astype(np.float32)
    count = 0
    for x in FitnessParamsDict.keys():
        FitnessParams[count] = FitnessParamsDict[x]
        count += 1

    #Convert GAParams to List
    GAParams = np.zeros(len(GAParamsDict.values())).astype(np.float32)
    count = 0
    for x in GAParamsDict.keys():
        GAParams[count] = GAParamsDict[x]
        count += 1

    # Set environment for template package Jinja2
    env = Environment(loader=PackageLoader('main', 'cuda'))

    # Load source code from file
    Source = env.get_template('kernel.cu') #Template( file(KernelFile).read() )

    #Create dictionary argument for rendering
    RenderArgs= {"params_size":GlobalParams.nbytes,\
                "fitnessparams_size":FitnessParams.nbytes,\
                "gaparams_size":GAParams.nbytes,\
                "genome_bytelength":int(ByteLengthGenome),\
                "genome_bitlength":int(BitLengthGenome),\
                "ga_nr_threadsperblock":GA_NrThreadsPerBlock,\
                "textures":range( 0, NrFitnessFunctionGrids ),\
                "curandinit_nr_threadsperblock":CurandInit_NrThreadsPerBlock,\
                "with_mixed_crossover":WithMixedCrossover,
                "with_bank_conflict":WithBankConflict,
                "with_naive_roulette_wheel_selection":WithNaiveRouletteWheelSelection,
                "with_assume_normalized_fitness_function_values":WithAssumeNormalizedFitnessFunctionValues,
                "with_uniform_crossover":WithUniformCrossover,
                "with_single_point_crossover":WithSinglePointCrossover,
                "with_surefire_mutation":WithSurefireMutation,
                "with_storeassembledgridsinglobalmemory":WithStoreAssembledGridsInGlobalMemory,
                "ga_threaddimx":int(ThreadDim),
                "glob_nr_tiletypes":int(NrTileTypes),
                "glob_nr_edgetypes":int(NrEdgeTypes),
                "glob_nr_tileorientations":int(NrTileOrientations),
                "fit_dimgridx":int(DimGridX),
                "fit_dimgridy":int(DimGridY),
                "fit_nr_fitnessfunctiongrids":int(NrFitnessFunctionGrids),
                "fit_nr_fourpermutations":int(NrFourPermutations),
                "fit_assembly_redundancy":int(NrAssemblyRedundancy),
                "fit_nr_threadsperblock":int(Fit_NrThreadsPerBlock),
                "sort_threaddimx":int(Sort_ThreadDimX),
                "glob_nr_genomes":int(NrGenomes),
                "fit_dimthreadx":int(ThreadDimX),
                "fit_dimthready":int(ThreadDimY),
                "fit_dimsubgridx":int(SubgridDimX),
                "fit_dimsubgridy":int(SubgridDimY),
                "fit_nr_subgridsperbank":int(NrSubgridsPerBank),
                "glob_bitlength_edgetype":int(EdgeTypeBitLength)
                }

    # Render source code
    RenderedSource = Source.render( RenderArgs )

    # Save rendered source code to file
    f = open('./rendered.cu', 'w')
    f.write(RenderedSource)
    f.close()

    #Load source code into module
    KernelSourceModule = SourceModule(RenderedSource, options=None, no_extern_c=True, arch="compute_11", code="sm_11", cache_dir=None)

    #Allocate values on GPU
    Genomes_h = drv.mem_alloc(Genomes.nbytes)
    FitnessPartialSums_h = drv.mem_alloc(FitnessPartialSums.nbytes)
    FitnessValues_h = drv.mem_alloc(FitnessValues.nbytes)
    AssembledGrids_h = drv.mem_alloc(AssembledGrids.nbytes)
    Mutexe_h = drv.mem_alloc(Mutexe.nbytes)
    ReductionList_h = drv.mem_alloc(ReductionList.nbytes)

    #Copy values to global memory
    drv.memcpy_htod(Genomes_h, Genomes)
    drv.memcpy_htod(FitnessPartialSums_h, FitnessPartialSums)
    drv.memcpy_htod(FitnessValues_h, FitnessValues)
    drv.memcpy_htod(AssembledGrids_h, AssembledGrids)
    drv.memcpy_htod(Mutexe_h, Mutexe)

    #Copy values to constant / texture memory
    for id in range(0, NrFitnessFunctionGrids):
        FitnessFunctionGrids_h.append( KernelSourceModule.get_texref("t_ucFitnessFunctionGrids%d"%(id)) )
        drv.matrix_to_texref( FitnessFunctionGrids[id], FitnessFunctionGrids_h[id] , order="C")
    InteractionMatrix_h = KernelSourceModule.get_texref("t_ucInteractionMatrix")
    drv.matrix_to_texref( InteractionMatrix, InteractionMatrix_h , order="C")

    GlobalParams_h = KernelSourceModule.get_global("c_fParams") # Constant memory address
    drv.memcpy_htod(GlobalParams_h[0], GlobalParams)
    FitnessParams_h = KernelSourceModule.get_global("c_fFitnessParams") # Constant memory address
    drv.memcpy_htod(FitnessParams_h[0], FitnessParams)
    GAParams_h = KernelSourceModule.get_global("c_fGAParams") # Constant memory address
    drv.memcpy_htod(GAParams_h[0], GAParams)
    FourPermutations_h = KernelSourceModule.get_global("c_ucFourPermutations") # Constant memory address
    drv.memcpy_htod(FourPermutations_h[0], FourPermutations)
    FitnessSumConst_h = KernelSourceModule.get_global("c_fFitnessSumConst")
    FitnessListConst_h = KernelSourceModule.get_global("c_fFitnessListConst")

    #Set up curandStates
    curandState_bytesize = 40 # This might be incorrect, depending on your compiler (info from Tomasz Rybak's pyCUDA cuRAND wrapper)
    CurandStates_h = drv.mem_alloc(curandState_bytesize * NrGenomes)

    #Compile kernels
    curandinit_fnc = KernelSourceModule.get_function("CurandInitKernel")
    fitness_fnc = KernelSourceModule.get_function("FitnessKernel")
    sorting_fnc = KernelSourceModule.get_function("SortingKernel")
    ga_fnc = KernelSourceModule.get_function("GAKernel")

    #Initialise Curand
    curandinit_fnc(CurandStates_h, block=(int(CurandInit_NrThreadsPerBlock), 1, 1), grid=(int(CurandInit_NrBlocks), 1))

    #Build parameter lists for FitnessKernel and GAKernel
    FitnessKernelParams = (Genomes_h, FitnessValues_h, AssembledGrids_h, CurandStates_h, Mutexe_h);
    SortingKernelParams = (FitnessValues_h, FitnessPartialSums_h)
    GAKernelParams = (Genomes_h, FitnessValues_h, AssembledGrids_h, CurandStates_h);

    #TEST ONLY
    return
    #TEST ONLY

    #Initialise CUDA timers
    start = drv.Event()
    stop = drv.Event()

    #execute kernels for specified number of generations
    start.record()
    for gen in range(0, GlobalParamsDict["NrGenerations"]):
        #print "Processing Generation: %d"%(gen)

        #fitness_fnc(*(FitnessKernelParams), block=fit_blocks, grid=fit_grid)

        #Launch CPU processing (should be asynchroneous calls)

        sorting_fnc(*(SortingKernelParams), block=sorting_blocks, grid=sorting_grids) #Launch Sorting Kernel

        drv.memcpy_dtoh(ReductionList, ReductionList_h) #Copy from Device to Host and finish sorting
        FitnessSumConst = ReductionList.sum()
        drv.memcpy_htod(FitnessSumConst_h[0], FitnessSumConst) #Copy from Host to Device constant memory
        drv.memcpy_dtod(FitnessListConst_h[0], FitnessValues_h, FitnessValues.nbytes) #Copy FitneValues from Device to Device Const

        ga_fnc(*(GAKernelParams), block=ga_blocks, grid=ga_grids)

        drv.memcpy_dtoh(Genomes, Genomes_h) #Copy data from GPU
        drv.memcpy_dtoh(FitnessValues, FitnessValues_h)
        drv.memcpy_dtoh(AssembledGrids, AssembledGrids_h)

    stop.record()
    stop.synchronize()
    print "Total kernel time taken: %fs"%(start.time_till(stop)*1e-3)
    print "Mean time per generation: %fs"%(start.time_till(stop)*1e-3 / NrGenerations)
    pass

if __name__ == '__main__':
    main()
