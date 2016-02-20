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

#from tables import *  # PyTables

from pycuda.compiler import SourceModule
from bitstring import BitStream, BitArray
from jinja2 import Environment, PackageLoader

#helper functions

def bitlength(arg):
    return math.ceil( math.log( arg , 2) )

def bytelength(arg):
    return math.ceil( arg / 8.0)

#Initialise data for GPU constant memory / user-defined constants - Global
NrGenomes = 512 #ADAPTED
NrTileTypes = 4 #ADAPTED
NrEdgeTypes = 8
NrTileOrientations = 4 # because square! don't change this... it might blow everything up.
NrGenerations = 20000 #ADAPTED

#Initialise data for GPU constant memory / user-defined constants - FitnessKernel
DimGridX = 16 #odd dims seem to work - good because only then really centered tile
DimGridY = 16 #odd dims seem to work - good because only then really centered tile
NrFitnessFunctionGrids = 1
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
NrMemoryBanks = 32
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
ThreadDimX = 8 #Reduced to avoid out-of-resources launch fail
ThreadDimY = 16 #Reduced to avoid out-of-resources launch fail
NrSubgridsPerBank = 4 #8
Fit_NrThreadsPerBlock = ThreadDimX * ThreadDimY

#Software-Tunable parameters for GAKernel
GA_ThreadDim = 256

#Initialise Parameters calculated from user-defined parameters
#ADJUSTED FOR FUJIAMA KERNEL
ByteLengthGenome = 4#int( bytelength( 4 * bitlength(NrEdgeTypes) * NrTileTypes ) )
BitLengthGenome = 4*8 #int( 4 * bitlength(NrEdgeTypes) * NrTileTypes )
EdgeTypeBitLength = int( bitlength(NrEdgeTypes)  )

fit_grid = (NrGenomes, 1)
#fit_blocks = (DimGridX, DimGridY, 1)#ADAPTED
fit_blocks = (ThreadDimX, ThreadDimY, 1) #ADAPTED

ga_grids = ( int( math.ceil( float(NrGenomes) / float(ThreadDimX) ) ), 1)
ga_blocks = (GA_ThreadDim, 1, 1)
GA_NrThreadsPerBlock =  GA_ThreadDim #int( math.ceil( float(NrGenomes) / float(ThreadDimX) ) )

sorting_blocks = (Sort_ThreadDimX, 1, 1)
sorting_grids = ( int( math.ceil( float(NrGenomes) / float(Sort_ThreadDimX) ) ), 1)

#ADAPTED START
#Check validity of above parameters - FitnessKernel
if fit_blocks[0] * fit_blocks[1] * SubgridDimX * SubgridDimY != DimGridX * DimGridY: #Check Number of cells allocated match with number of threads / grid size available
    print "FITNESS_KERNEL:Scheduled cells", fit_blocks[0] * fit_blocks[1] * SubgridDimX * SubgridDimY, " not equal to Grid Dims: ", DimGridX * DimGridY,
    quit()
if NrSubgridsPerBank * NrMemoryBanks * SubgridDimX * SubgridDimY < DimGridX * DimGridY:
    print "Not enough SubgridsPerBank scheduled to accommodate all cells! I.e. ",NrSubgridsPerBank * NrMemoryBanks * SubgridDimX * SubgridDimY,"is smaller than ",DimGridX * DimGridY
    quit()
#ADAPTED STOP

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
#ADAPTED START
#FitnessFunctionGrids[0][int(DimGridX/2)][int(DimGridY/2)] = 255
#ADAPTED STOP

#Initialise data for GPU global memory
#Genomes = np.zeros( ( NrGenomes, ByteLengthGenome) ).astype(np.uint8) #ADAPTED
#001001001001 010011101011 011011011110 000000000100 #ADAPTED
#00100100 10010100 11101011 01101101 11100000 00000100 #ADAPTED
#From Iain's Paper - pyramid polyomino
#Genomes = np.array( [ [0b00100100, 0b10010100, 0b11101011, 0b01101101, 0b11100000, 0b00000100] ] ).astype(np.uint8); #ADAPTED
#Genomes = np.array( [ [0b00100100, 0b10010111, 0b01011010, 0b01101101, 0b11100000, 0b00000100] ] ).astype(np.uint8); #ADAPTED
#print Genomes #ADAPTED TEST
#quit() #ADAPTED TEST
Genomes = np.zeros( (NrGenomes * ByteLengthGenome) ).astype(np.uint8)
FitnessPartialSums = np.zeros( ( sorting_grids[0] ) ).astype(np.float32)
FitnessValues = np.zeros( (NrGenomes) ).astype(np.float32) #ADAPTED FOR DISCOVERY TIME KERNEL
AssembledGrids = np.zeros( ( NrGenomes, DimGridX, DimGridY ) ).astype(np.uint8)
Mutexe = np.zeros( (NrMutexe) ).astype(np.uint32)
#ReductionList = np.ones( (sorting_grids[0]) ).astype(np.float32) #np.zeros( (sorting_grids[0]) ).astype(np.float32)

#ADAPTED: Assign result arrays
Genomes_res = np.zeros_like(Genomes)
FitnessValues_res = np.zeros_like(FitnessValues)
AssembledGrids_res = np.zeros_like(AssembledGrids)

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
    env = Environment(loader=PackageLoader('main_discoverytime', './templates'))

    # Load source code from file
    Source = env.get_template('./kernel.cu') #Template( file(KernelFile).read() )

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
                "ga_threaddimx":int(GA_ThreadDim),
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
                "glob_bitlength_edgetype":int(EdgeTypeBitLength),
                "fitness_break_value":int(BitLengthGenome),   # ADAPTED FOR DISCOVERY KERNEL
                "fitness_flag_index":int(NrGenomes)
                }

    # Render source code
    RenderedSource = Source.render( RenderArgs )

    # Save rendered source code to file
    f = open('./rendered.cu', 'w')
    f.write(RenderedSource)
    f.close()

    #Load source code into module
    KernelSourceModule = SourceModule(RenderedSource, options=None, no_extern_c=True, arch="compute_20", code="sm_20", cache_dir=None)

    #Allocate values on GPU
    Genomes_h = drv.mem_alloc(Genomes.nbytes)
    FitnessPartialSums_h = drv.mem_alloc(FitnessPartialSums.nbytes)
    FitnessValues_h = drv.mem_alloc(FitnessValues.nbytes)
    AssembledGrids_h = drv.mem_alloc(AssembledGrids.nbytes)
    Mutexe_h = drv.mem_alloc(Mutexe.nbytes)
    #ReductionList_h = drv.mem_alloc(ReductionList.nbytes)

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
    #fitness_fnc = KernelSourceModule.get_function("FitnessKernel")
    sorting_fnc = KernelSourceModule.get_function("SortingKernel")
    ga_fnc = KernelSourceModule.get_function("GAKernel")

    #Initialise Curand
    curandinit_fnc(CurandStates_h, block=(int(CurandInit_NrThreadsPerBlock), 1, 1), grid=(int(CurandInit_NrBlocks), 1))

    #Build parameter lists for FitnessKernel and GAKernel
    FitnessKernelParams = (Genomes_h, FitnessValues_h, AssembledGrids_h, CurandStates_h, Mutexe_h)
    SortingKernelParams = (FitnessValues_h, FitnessPartialSums_h)
    GAKernelParams = (Genomes_h, FitnessValues_h, AssembledGrids_h, CurandStates_h)

    #TEST ONLY
    #return #ADAPTED
    #TEST ONLY

    #START ADAPTED
    print "GENOMES NOW:\n"
    print Genomes
    print ":::STARTING KERNEL EXECUTION:::"
    #STOP ADAPTED

    #Discovery time parameters
    min_fitness_value = BitLengthGenome # Want all bits set
    mutation_rate = -2


    #Define Numpy construct to sideways join arrays (glue columns together)
    #Taken from: http://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays
    #def join_struct_arrays(arrays):
    #    sizes = np.array([a.itemsize for a in arrays])
    #    offsets = np.r_[0, sizes.cumsum()]
    #    n = len(arrays[0])
    #    joint = np.empty((n, offsets[-1]), dtype=np.int32)
    #    for a, size, offset in zip(arrays, sizes, offsets):
    #        joint[:,offset:offset+size] = a.view(np.int32).reshape(n,size)
    #    dtype = sum((a.dtype.descr for a in arrays), [])
    #    return joint.ravel().view(dtype)
    #Test join_struct_arrays:
    #a = np.array([[1, 2], [11, 22],  [111, 222]]).astype(np.int32);
    #b = np.array([[3, 4], [33, 44],  [333, 444]]).astype(np.int32);
    #c = np.array([[5, 6], [55, 66],  [555, 666]]).astype(np.int32);
    #print "Test join_struct_arrays:"
    #print join_struct_arrays([a, b, c]) #FAILED
    #Set up PYTABLES
    #class GAGenome(IsDescription):
        #gen_id = Int32Col()
        #fitness_val = Float32Col()
        #genome = StringCol(mByteLengthGenome)
        #last_nr_mutations = Int32Col() # Contains the Nr of mutations genome underwent during this generation
        #mother_id = Int32Col() # Contains the crossover "mother"
        #father_id = Int32Col()  # Contains the crossover "father" (empty if no crossing over)
        #assembledgrid      = StringCol(DimGridX*DimGridY)   # 16-character String
    #class GAGenerations(IsDescription):
    #    nr_generations = Int32Col()
    #    nr_genomes = Int32Col()
    #    mutation_rate = Float32Col() # Contains the Nr of mutations genome underwent during this generation

    #from datetime import datetime

    #filename = "fujiama_"+str(NrGenomes)+"_"+str(RateMutation)+"_"+".h5"
    #print filename
    #h5file = openFile(filename, mode = "w", title = "GA FILE")
    #group = h5file.createGroup("/", 'fujiama_ga', 'Fujiama Genetic Algorithm output')
    #table = h5file.createTable(group, 'GaGenerations', GAGenerations, "Raw data")
    #atom = Atom.from_dtype(np.float32)

    #Initialise File I/O
    FILE = open("fujiamakernel_nrgen-" + str(NrGenomes) + "_discoverytime.plot", "w")

    #Initialise CUDA timers
    start = drv.Event()
    stop = drv.Event()


    while mutation_rate < 1:

        #ds = h5file.createArray(f.root, 'ga_raw_'+str(mutation_rate), atom, x.shape)
        mutation_rate += 0.1
        GAParams[0]  = 10.0 ** mutation_rate
        drv.memcpy_htod(GAParams_h[0], GAParams)
        print "Mutation rate: ", GAParams[0]

        #ADAPTED: Initialise global memory (absolutely necessary!!)
        #drv.memcpy_htod(Genomes_h, Genomes)
        #drv.memcpy_htod(FitnessValues_h, FitnessValues)
        #drv.memcpy_htod(AssembledGrids_h, AssembledGrids)
        #drv.memcpy_htod(Mutexe_h, Mutexe)

        #execute kernels for specified number of generations
        #start.record()

        biggest_fit = 0
	reprange = 100
        average_breakup = np.zeros( (reprange) ).astype(np.float32)
        for rep in range(0, int(reprange) ):
	    breakup_generation = GlobalParamsDict["NrGenerations"]
            dontcount = 0
	    #ADAPTED: Initialise global memory (absolutely necessary!!)
            drv.memcpy_htod(Genomes_h, Genomes)
            drv.memcpy_htod(FitnessValues_h, FitnessValues)
            drv.memcpy_htod(AssembledGrids_h, AssembledGrids)
            drv.memcpy_htod(Mutexe_h, Mutexe)

            #execute kernels for specified number of generations
            start.record()

            for gen in range(0, GlobalParamsDict["NrGenerations"]):
                #print "Processing Generation: %d"%(gen)

                #Launch CPU processing (should be asynchroneous calls)

                sorting_fnc(*(SortingKernelParams), block=sorting_blocks, grid=sorting_grids) #Launch Sorting Kernel
                drv.memcpy_dtoh(FitnessPartialSums, FitnessPartialSums_h) #Copy from Device to Host and finish sorting
                FitnessSumConst = FitnessPartialSums.sum()
                drv.memcpy_htod(FitnessSumConst_h[0], FitnessSumConst) #Copy from Host to Device constant memory
                #drv.memcpy_dtod(FitnessListConst_h[0], FitnessValues_h, FitnessValues.nbytes) #Copy FitnessValues from Device to Device Const #TEST

                ga_fnc(*(GAKernelParams), block=ga_blocks, grid=ga_grids) #TEST
                #Note: Fitness Function is here integrated into GA kernel!

                drv.memcpy_dtoh(Genomes_res, Genomes_h) #Copy data from GPU
                drv.memcpy_dtoh(FitnessValues_res, FitnessValues_h)
                #drv.memcpy_dtoh(AssembledGrids_res, AssembledGrids_h) #Takes about as much time as the whole simulation!

                #print FitnessValues_res

                maxxie = FitnessValues_res.max()
                if maxxie > biggest_fit:
                    biggest_fit = maxxie
                    #print "max fitness:", maxxie
                if maxxie >= 25.0 and breakup_generation == 20000:
                    breakup_generation = gen
                    break
                # else:
                #    breakup_generation = -1
                #if FitnessValues[NrGenomes]  == float(1):
                #    breakup_generation = i
                #    break
                #else:
                #    breakup_generation = -1

                #maxxie = FitnessValues_res.max()
                #if maxxie >= 30:
                #    print "Max fitness value: ", FitnessValues_res.max()
                #ds[:]  = FitnessValues #join_struct_arrays(Genomes,  FitnessValues,  AssembledGrids);
                #trow = table.row
                #trow['nr_generations'] = NrGenerations
                #trow['nr_genomes'] = NrGenomes
                #trow['mutation_rate'] = mutation_rate
                #trow.append()
                #trow.flush()

            stop.record()
            stop.synchronize()
            print "Total kernel time taken: %fs"%(start.time_till(stop)*1e-3)
            print "Mean time per generation: %fs"%(start.time_till(stop)*1e-3 / NrGenerations)
            print "Discovery time (generations) for mutation rate %f: %d"%(GAParams[0],  breakup_generation)
            #print "Max:", biggest_fit
            #if breakup_generation==0:
            #    print FitnessValues_res
            #    print "Genomes: "
            #    print Genomes_res
            #FILE.write( str(GAParams[0]) + " " + str(breakup_generation) + "\n");
            average_breakup[rep] = breakup_generation / reprange
            #if breakup_generation == -1:
            #    dontcount = 1
            #    break

        #if dontcount == 1:
        #    average_breakup.fill(20000)
        FILE.write( str(GAParams[0]) + " " + str(np.median(average_breakup)) + " " + str(np.std(average_breakup)) + "\n");
        FILE.flush()
            

    #Clean-up pytables
    #h5file.close()
    #Clean up File I/O
    FILE.close()

if __name__ == '__main__':
    main()
