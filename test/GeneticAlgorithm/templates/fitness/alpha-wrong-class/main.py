import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
from jinja2 import Environment, PackageLoader

def main():

    #FourPermutations set-up
    FourPermutations = numpy.array([ [1,2,3,4],
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
                                  [4,1,3,2],]).astype(numpy.uint8)

    BankSize = 8 # Do not go beyond 8!

    #Define constants
    DimGridX = 19
    DimGridY = 19
    #SearchSpaceSize = 2**24
    #BlockDimY = SearchSpaceSize / (2**16)
    #BlockDimX = SearchSpaceSize / (BlockDimY)
    #print "SearchSpaceSize: ", SearchSpaceSize, " (", BlockDimX, ", ", BlockDimY,")"
    SearchSpaceSize = 100 * 100  * 32
    BlockDimX = 100
    BlockDimY = 100
    FitnessValDim = SearchSpaceSize
    GenomeDim = SearchSpaceSize

    #Create dictionary argument for rendering
    RenderArgs= {"safe_memory_mapping":1,
                 "aligned_byte_length_genome":4,
                 "bit_length_edge_type":3,
                 "curand_nr_threads_per_block":8*32,
                 "nr_tile_types":2,
                 "nr_edge_types":8,
                 "warpsize":32,
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
                 "banksize":BankSize
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
    dest = numpy.arange(GenomeDim*4).astype(numpy.uint8)
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

    dest_h = drv.mem_alloc(GenomeDim*4) #dest.nbytes)
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
    fitness_left = numpy.zeros(FitnessValDim).astype(numpy.float32)
    fitness_left_h = drv.mem_alloc(fitness_left.nbytes)
    fitness_bottom = numpy.zeros(FitnessValDim).astype(numpy.float32)
    fitness_bottom_h = drv.mem_alloc(fitness_bottom.nbytes)
    #drv.memcpy_htod(fitness_h, fitness)
    #print "Fitness values:"
    #print fitness

    #Set-up grids
    grids = numpy.zeros((10000*32, DimGridX, DimGridY)).astype(numpy.uint8) #TEST
    grids_h = drv.mem_alloc(GenomeDim*DimGridX*DimGridY) #TEST
    #drv.memcpy_htod(grids_h, grids)
    #print "Grids:"
    #print grids 

    #Set-up curand
    #curand = numpy.zeros(40*GenomeDim).astype(numpy.uint8);
    #curand_h = drv.mem_alloc(curand.nbytes)
    curand_h = drv.mem_alloc(40*GenomeDim)

    #Set-up four permutations
    FourPermutations_h = KernelSourceModule.get_global("c_ucFourPermutations") # Constant memory address
    drv.memcpy_htod(FourPermutations_h[0], FourPermutations)

    #SearchSpace control
    #SearchSpaceSize = 2**24
    #BlockDimY = SearchSpaceSize / (2**16)
    #BlockDimX = SearchSpaceSize / (BlockDimY)
    #print "SearchSpaceSize: ", SearchSpaceSize, " (", BlockDimX, ", ", BlockDimY,")"
   
    #Set-up timer
    start = drv.Event()
    stop = drv.Event()
    start.record()

    print "Start kernel:"

    #Call kernels
    CurandKernel(curand_h, block=(32,1,1), grid=(BlockDimX,BlockDimY))
    print "Finished Curand kernel, starting main kernel..."
    Kernel(dest_h, grids_h, fitness_left_h, fitness_bottom_h, curand_h, block=(32*BankSize,1,1), grid=(BlockDimX,BlockDimY))

    #End timer
    stop.record()
    stop.synchronize()
    print "Total kernel time taken: %fs"%(start.time_till(stop)*1e-3)
    #print "Mean time per generation: %fs"%(start.time_till(stop)*1e-3 / NrGenerations)
    pass

    #Output
    #drv.memcpy_dtoh(dest, dest_h)
    #print "Genomes after: "
    #print dest[0:4]
    
    drv.memcpy_dtoh(fitness_left, fitness_left_h)
    print "FitnessLeft after: "
    print fitness_left[0:500]

    drv.memcpy_dtoh(fitness_bottom, fitness_bottom_h)
    print "FitnessBottom after: "
    print fitness_bottom[0:500]

    drv.memcpy_dtoh(grids, grids_h)
    print "Grids[0] after: "
    print grids[160]
    print "Grids[31] after:"
    print grids[200]

if __name__ == '__main__':
    main()

