import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time

from pycuda.compiler import SourceModule
from jinja2 import Environment, PackageLoader

def main():

    numpy.set_printoptions(precision=4,
                       threshold=10000,
                       linewidth=150)
    #Set up global timer
    tot_time = time.time()

    #Define constants
    BankSize = 8 # Do not go beyond 8!
    WarpSize = 32 #Do not change...
    DimGridX = 19
    DimGridY = 19
    BlockDimX = 4#256
    BlockDimY = 4#256
    SearchSpaceSize = 2**24 #BlockDimX * BlockDimY  * 32
    FitnessValDim = BlockDimX*BlockDimY*WarpSize #SearchSpaceSize
    GenomeDim = BlockDimX*BlockDimY*WarpSize #SearchSpaceSize
    AlignedByteLengthGenome = 4

    print "Total number of genomes:", GenomeDim

    #Create dictionary argument for rendering
    RenderArgs= {"safe_memory_mapping":1,
                 "aligned_byte_length_genome":AlignedByteLengthGenome,
                 "bit_length_edge_type":3,
                 "curand_nr_threads_per_block":32,
                 "nr_tile_types":2,
                 "nr_edge_types":8,
                 "warpsize":WarpSize,
                 "fit_dim_thread_x":32*BankSize,
                 "fit_dim_thread_y":1,
                 "fit_dim_block_x":BlockDimX,
                 "fit_dim_grid_x":19,
                 "fit_dim_grid_y":19,
                 "fit_nr_four_permutations":24,
                 "fit_length_movelist":244,
                 "fit_nr_redundancy_grid_depth":2,
                 "fit_nr_redundancy_assemblies":10,
                 "fit_tile_index_starting_tile":0,
                 "glob_nr_tile_orientations":4,
                 "banksize":BankSize,
                 "curand_dim_block_x":BlockDimX
                }
    # Set environment for template package Jinja2
    env = Environment(loader=PackageLoader('main', './'))
    # Load source code from file
    Source = env.get_template('./alpha.cu') #Template( file(KernelFile).read() )
    # Render source code
    RenderedSource = Source.render( RenderArgs )

    # Save rendered source code to file
    f = open('./rendered.cu', 'w')
    f.write(RenderedSource)
    f.close()

    #Load source code into module
    KernelSourceModule = SourceModule(RenderedSource, options=None, arch="compute_20", code="sm_20")
    Kernel = KernelSourceModule.get_function("SearchSpaceKernel")
    CurandKernel = KernelSourceModule.get_function("CurandInitKernel")


    #Initialise InteractionMatrix
    InteractionMatrix = numpy.zeros( ( 8, 8) ).astype(numpy.float32)
    def Delta(a,b):
        if a==b:
            return 1
        else:
            return 0
    for i in range(InteractionMatrix.shape[0]):
        for j in range(InteractionMatrix.shape[1]):
            InteractionMatrix[i][j] = ( 1 - i % 2 ) * Delta( i, j+1 ) + ( i % 2 ) * Delta( i, j-1 )

    #Set up our InteractionMatrix
    InteractionMatrix_h = KernelSourceModule.get_texref("t_ucInteractionMatrix")
    drv.matrix_to_texref( InteractionMatrix, InteractionMatrix_h , order="C")
    print InteractionMatrix

    #Set-up genomes
    #dest = numpy.arange(GenomeDim*4).astype(numpy.uint8)
    #for i in range(0, GenomeDim/4):
        #dest[i*8 + 0] = int('0b00100101',2) #CRASHES
        #dest[i*8 + 1] = int('0b00010000',2) #CRASHES
        #dest[i*8 + 0] = int('0b00101000',2)
        #dest[i*8 + 1] = int('0b00000000',2)
        #dest[i*8 + 2] = int('0b00000000',2)
        #dest[i*8 + 3] = int('0b00000000',2)
        #dest[i*8 + 4] = int('0b00000000',2)
        #dest[i*8 + 5] = int('0b00000000',2)
        #dest[i*8 + 6] = int('0b00000000',2)
        #dest[i*8 + 7] = int('0b00000000',2)
    #    dest[i*4 + 0] = 40
    #    dest[i*4 + 1] = 0
    #    dest[i*4 + 2] = 0
    #    dest[i*4 + 3] = 0

    dest_h = drv.mem_alloc(GenomeDim*AlignedByteLengthGenome) #dest.nbytes)
    dest = drv.pagelocked_zeros((GenomeDim*AlignedByteLengthGenome), numpy.uint8, "C", 0)
    #drv.memcpy_htod(dest_h, dest)
    #print "Genomes before: "
    #print dest

    #Set-up grids
    #grids = numpy.zeros((10000, DimGridX, DimGridY)).astype(numpy.uint8) #TEST
    #grids_h = drv.mem_alloc(GenomeDim*DimGridX*DimGridY) #TEST
    #drv.memcpy_htod(grids_h, grids)
    #print "Grids:"
    #print grids    

    #Set-up fitness values
    #fitness = numpy.zeros(FitnessValDim).astype(numpy.float32)
    #fitness_h = drv.mem_alloc(fitness.nbytes)
    #fitness_size = numpy.zeros(FitnessValDim).astype(numpy.uint32)
    fitness_size = drv.pagelocked_zeros((FitnessValDim), numpy.uint32, "C", 0)
    fitness_size_h = drv.mem_alloc(fitness_size.nbytes)
    #fitness_hash = numpy.zeros(FitnessValDim).astype(numpy.uint32)
    fitness_hash = drv.pagelocked_zeros((FitnessValDim), numpy.uint32, "C", 0)
    fitness_hash_h = drv.mem_alloc(fitness_hash.nbytes)
    #drv.memcpy_htod(fitness_h, fitness)
    #print "Fitness values:"
    #print fitness

    #Set-up grids
    #grids = numpy.zeros((GenomeDim, DimGridX, DimGridY)).astype(numpy.uint8) #TEST
    grids = drv.pagelocked_zeros((GenomeDim, DimGridX, DimGridY), numpy.uint8, "C", 0)
    grids_h = drv.mem_alloc(GenomeDim*DimGridX*DimGridY) #TEST
    
    #drv.memcpy_htod(grids_h, grids)
    #print "Grids:"
    #print grids 

    #Set-up curand
    #curand = numpy.zeros(40*GenomeDim).astype(numpy.uint8);
    #curand_h = drv.mem_alloc(curand.nbytes)
    curand_h = drv.mem_alloc(40*GenomeDim)

    #SearchSpace control
    #SearchSpaceSize = 2**24
    #BlockDimY = SearchSpaceSize / (2**16)
    #BlockDimX = SearchSpaceSize / (BlockDimY)
    #print "SearchSpaceSize: ", SearchSpaceSize, " (", BlockDimX, ", ", BlockDimY,")"
   
    #Schedule kernel calls
    #MaxBlockDim = 100
    OffsetBlocks = (SearchSpaceSize) % (BlockDimX*BlockDimY*WarpSize)
    MaxBlockCycles = (SearchSpaceSize - OffsetBlocks)/(BlockDimX*BlockDimY*WarpSize)
    BlockCounter = 0
    print "Will do that many kernels a ", BlockDimX,"x", BlockDimY,"x ", WarpSize, ":", MaxBlockCycles
    #quit()

    #SET UP PROCESSING
    histo = {}
     
    #INITIALISATION
    CurandKernel(curand_h, block=(WarpSize,1,1), grid=(BlockDimX, BlockDimY))
    print "Finished Curand kernel, starting main kernel..."

    #FIRST GENERATION
    proc_time = time.time()
    print "Starting first generation..."
    start = drv.Event()
    stop = drv.Event()
    start.record()
    Kernel(dest_h, grids_h, fitness_size_h, fitness_hash_h, curand_h, numpy.int64(0), block=(WarpSize*BankSize,1,1), grid=(BlockDimX,BlockDimY))
    stop.record()
    stop.synchronize()
    print "Total kernel time taken: %fs"%(start.time_till(stop)*1e-3)    
    print "Copying..."
    drv.memcpy_dtoh(fitness_size, fitness_size_h)
    drv.memcpy_dtoh(fitness_hash, fitness_hash_h)
    drv.memcpy_dtoh(grids, grids_h)
    drv.memcpy_dtoh(dest, dest_h)

    #INTERMEDIATE GENERATION
    for i in range(MaxBlockCycles-1):
        print "Starting generation: ", i+1
        start = drv.Event()
        stop = drv.Event()
        start.record()
        Kernel(dest_h, grids_h, fitness_size_h, fitness_hash_h, curand_h, numpy.int64((i+1)*BlockDimX*BlockDimY*WarpSize), block=(WarpSize*BankSize,1,1), grid=(BlockDimX,BlockDimY))
        print "Processing..."
        for j in range(len(fitness_hash)):
#            if (fitness_hash[j]!=33) and (fitness_hash[j]!=44) and (fitness_hash[j]!=22):
            if fitness_hash[j] in histo: 
                histo[fitness_hash[j]] = (grids[j], dest[j*AlignedByteLengthGenome]+dest[j*AlignedByteLengthGenome+1]*2**8+dest[j*AlignedByteLengthGenome+2]*2**16+dest[j*AlignedByteLengthGenome+3]*2**24, histo[fitness_hash[j]][2]+1, fitness_size[j])
#(histo[fitness_hash[j]][0], histo[fitness_hash[j]][1], histo[fitness_hash[j]][2]+1, histo[fitness_hash[j]][3])
            else:
                histo[fitness_hash[j]] = (grids[j], dest[j*AlignedByteLengthGenome]+dest[j*AlignedByteLengthGenome+1]*2**8+dest[j*AlignedByteLengthGenome+2]*2**16+dest[j*AlignedByteLengthGenome+3]*2**24, 1, fitness_size[j])

        #DEBUG
        f = open("S28_tmp.s28", "w")
        for j in range(len(fitness_size)): 
            print >>f, "Size: ", fitness_size[j], " Hash:", fitness_hash[j], " Genome:", dest[j*AlignedByteLengthGenome], "|",dest[j*AlignedByteLengthGenome+1]," | ",dest[j*AlignedByteLengthGenome+2], " | ", dest[j*AlignedByteLengthGenome+3]
            print >>f, grids[j]
        f.close() 
        quit() #DEBUG  

if __name__ == '__main__':
    main()

