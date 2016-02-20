{% include "./alpha_core.cu" %}

__device__ float xAssembly::fEvaluateFitness(xThreadInfo __xThreadInfo){

        if(__xThreadInfo.WarpId()==0){
            this->data.corner_lower[__xThreadInfo.BankId()]=make_int2(0,0);
            this->data.corner_upper[__xThreadInfo.BankId()]=make_int2(m_fit_DimGridX-1,m_fit_DimGridY-1);
            this->data.assembly_size[__xThreadInfo.BankId()]=0; 
        } 
        __syncthreads();

        short offset = (m_fit_DimGridX*m_fit_DimGridY) % mBankSize;
        short myshare = (m_fit_DimGridX*m_fit_DimGridY - offset) / mBankSize;
        int off_x=0, off_y=0;

        //
        // 0. Calculate shape size
        //

        //Step0: Count how many tiles have been assembled - i.e. we do this as assembly process is still faulty
        {
        int sum_x=0, sum_y=0;
        for(int i=0;i<myshare;i++){
                off_x = (myshare*__xThreadInfo.WarpId()+i) % m_fit_DimGridX;
                off_y = (myshare*__xThreadInfo.WarpId()+i-off_x) / m_fit_DimGridX;
                if(this->data.grid.data.multi_d[off_x][off_y][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                     sum_x += 1;
                }
        }
        if(__xThreadInfo.WarpId()==mBankSize-1){
                for(int i=0;i<offset;i++){
                     if(this->data.grid.data.multi_d[off_x][off_y][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                         sum_x += 1;
                     }
                }
        }
        __syncthreads();
        atomicAdd(&this->data.assembly_size[__xThreadInfo.BankId()], sum_x);
        __syncthreads();
        }

        //
        // I. Calculate crop coordinates
        //
        {        
        int off_x=0, off_y=0;
        for(int i=0;i<myshare;i++){
                off_x = (myshare*__xThreadInfo.WarpId()+i) % m_fit_DimGridX;
                off_y = (myshare*__xThreadInfo.WarpId()+i-off_x) / m_fit_DimGridX;
                if(this->data.grid.data.multi_d[off_x][off_y][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                     //upper left corner: minimise x,y
                     atomicMin(&this->data.corner_upper[__xThreadInfo.BankId()].x, off_x);
                     atomicMin(&this->data.corner_upper[__xThreadInfo.BankId()].y, off_y);

                     //lower right corner: maximise x,y
                     atomicMax(&this->data.corner_lower[__xThreadInfo.BankId()].x, off_x);
                     atomicMax(&this->data.corner_lower[__xThreadInfo.BankId()].y, off_y);
                }
        }
        if(__xThreadInfo.WarpId()==mBankSize-1){
                for(int i=0;i<offset;i++){
                     if(this->data.grid.data.multi_d[off_x][off_y][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                          //upper left corner: minimise x,y
                          atomicMin(&this->data.corner_upper[__xThreadInfo.BankId()].x, off_x);
                          atomicMin(&this->data.corner_upper[__xThreadInfo.BankId()].y, off_y);

                          //lower right corner: maximise x,y
                          atomicMax(&this->data.corner_lower[__xThreadInfo.BankId()].x, off_x);
                          atomicMax(&this->data.corner_lower[__xThreadInfo.BankId()].y, off_y);
                     }
                }
        }
        }

        //DEBUG START
	//this->data.grid.set_xCell( __xThreadInfo, 0, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(),  this->data.corner_upper[__xThreadInfo.BankId()].x);
        //this->data.grid.set_xCell( __xThreadInfo, 1, 0, this->data.flags[__xThreadInfo.BankId()].get_ucRed(),  this->data.corner_upper[__xThreadInfo.BankId()].y);
        //this->data.grid.set_xCell( __xThreadInfo, 0, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(),  this->data.corner_lower[__xThreadInfo.BankId()].x);
        //this->data.grid.set_xCell( __xThreadInfo, 1, 1, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.corner_lower[__xThreadInfo.BankId()].y);
        //this->data.grid.set_xCell( __xThreadInfo, 0, 2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.assembly_size[__xThreadInfo.BankId()]);
        //this->data.grid.set_xCell( __xThreadInfo, 0, 2, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 9);
        //DEBUG END
        __syncthreads();

        //
        // II. Calculate 4 rotation hashes
        //

        if(__xThreadInfo.WarpId()==0){ //Rotation: 0 degrees
        	int2 up = this->data.corner_upper[__xThreadInfo.BankId()];
	        int2 low = this->data.corner_lower[__xThreadInfo.BankId()];
                int tmphash;
                jenkins_init(tmphash);
                for(int i=up.x;i<low.x+1;i++){
	                for(int j=up.y;j<low.y+1;j++){
                		if(this->data.grid.data.multi_d[i][j][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
					jenkins_add( (char) 1, tmphash);						
				} else {
                                        jenkins_add( (char) 0, tmphash);
                                }
			}
                }
                this->data.hash[__xThreadInfo.BankId()].x = jenkins_clean_up(tmphash);

        } else if(__xThreadInfo.WarpId()==1){ //Rotation: 90 degrees clockwise
                int2 up = this->data.corner_upper[__xThreadInfo.BankId()];
                int2 low = this->data.corner_lower[__xThreadInfo.BankId()];		
                int tmphash;
                jenkins_init(tmphash);
                for(int i=up.x;i<low.x+1;i++){
                        for(int j=low.y;j>up.y-1;j--){
                                if(this->data.grid.data.multi_d[i][j][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                                        jenkins_add( (char) 1, tmphash);
                                } else {
                                        jenkins_add( (char) 0, tmphash);
                                }
                        }
                }
                this->data.hash[__xThreadInfo.BankId()].y = jenkins_clean_up(tmphash);

        } else if(__xThreadInfo.WarpId()==2){ //Rotation: 180 degrees clockwise
                int2 up = this->data.corner_upper[__xThreadInfo.BankId()];
                int2 low = this->data.corner_lower[__xThreadInfo.BankId()];		
                int tmphash;
                jenkins_init(tmphash);
                for(int i=low.x;i>up.x-1;i--){
                        for(int j=low.y;j>up.y-1;j--){
                                if(this->data.grid.data.multi_d[i][j][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                                        jenkins_add( (char) 1, tmphash);
                                } else {
                                        jenkins_add( (char) 0, tmphash);
                                }
                        }
                }
                this->data.hash[__xThreadInfo.BankId()].z = jenkins_clean_up(tmphash);

        } else if(__xThreadInfo.WarpId()==3){ //Rotation: 270 degrees clockwise
                int2 up = this->data.corner_upper[__xThreadInfo.BankId()];
                int2 low = this->data.corner_lower[__xThreadInfo.BankId()];
                int tmphash;
                jenkins_init(tmphash);
                for(int i=low.x;i>up.x-1;i--){
                        for(int j=up.y;j<low.y+1;j++){
                                if(this->data.grid.data.multi_d[i][j][this->data.flags[__xThreadInfo.BankId()].get_ucRed()][__xThreadInfo.BankId()].get_xCell()!=mEMPTY_CELL){
                                        jenkins_add( (char) 1, tmphash);
                                } else {
                                        jenkins_add( (char) 0, tmphash);
                                }
                        }
                }
                this->data.hash[__xThreadInfo.BankId()].w = jenkins_clean_up(tmphash);	
        }
  
        //
        // III. Unify rotation hashes
        //
        __syncthreads();
        if(__xThreadInfo.WarpId()==0){
		union {
                	int4 integers;
                        char chars[16];
                } conv;
                int newhash=0;
                conv.integers = this->data.hash[__xThreadInfo.BankId()];
                for(int i=0;i<16;i++){
			jenkins_add(conv.chars[i], newhash);	
                }
                this->data.hash[__xThreadInfo.BankId()].x = jenkins_clean_up(newhash); 
        }

        /*if(__xThreadInfo.WarpId()==1){
		atomicXor(&this->data.hash[__xThreadInfo.BankId()].x, this->data.hash[__xThreadInfo.BankId()].y);
        } 
        if(__xThreadInfo.WarpId()==2){
                atomicXor(&this->data.hash[__xThreadInfo.BankId()].x, this->data.hash[__xThreadInfo.BankId()].z);
        }
        if(__xThreadInfo.WarpId()==3){
                atomicXor(&this->data.hash[__xThreadInfo.BankId()].x, this->data.hash[__xThreadInfo.BankId()].w);
        }*/

        __syncthreads();
	if(this->data.flags[__xThreadInfo.BankId()].get_bStericUND()){
            this->data.hash[__xThreadInfo.BankId()].x = 22;
        } if(this->data.flags[__xThreadInfo.BankId()].get_bTrivialUND()){
            this->data.hash[__xThreadInfo.BankId()].x = 33;
        } if(this->data.flags[__xThreadInfo.BankId()].get_bUnboundUND()){
            this->data.hash[__xThreadInfo.BankId()].x = 44;
        } 
        __syncthreads();

        //Additional GENERAL DEBUGGING - STUFF WHICH IS NOT TO BE AFFECTED BY FITNESS
/*        for(int i=0;i<mNrEdgeTypes;i++){
                for(int j=0;j<this->data.edgesort.get_xLength(__xThreadInfo, i);j++){
                       for(int k=0;k<4;k++){
                               this->data.grid.set_xCell( __xThreadInfo, j*4+k, i, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), this->data.edgesort.data.multi_d[i][j][k][__xThreadInfo.BankId()]);
                        }
                }
                this->data.grid.set_xCell( __xThreadInfo, this->data.edgesort.get_xLength(__xThreadInfo, i)*4, i, this->data.flags[__xThreadInfo.BankId()].get_ucRed(), 9);
        }*/

}

__global__ void SearchSpaceKernel(unsigned char *g_ucGenomes,  unsigned char *g_ucGrids, int *g_ucFitnessSize, int *g_ucFitnessHash,  curandState *states, long long int g_startval)
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
    
    union{
        long long int integer;
        char chars[8];
    } r_conv;

/*    if(r_xThreadInfo.WarpId() < mAlignedByteLengthGenome){
        r_conv.integer = g_startval + (blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId();
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[r_xThreadInfo.WarpId()] = r_conv.chars[r_xThreadInfo.WarpId()];
    }
*/
    /*if(r_xThreadInfo.FlatBlockId()%3==0){
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 33;
    } else if (r_xThreadInfo.FlatBlockId()%3==1) {
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 0;        
    } else if (r_xThreadInfo.FlatBlockId()%3==2) {
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 40;
    }

    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[1] = 0;//192; //0; //192; //6; //162; //160 //128; //0;//32;//16;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[2] = 0; //2; //138; //138;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[3] = 0;
    */

    /*int x = 3;
    if(r_xThreadInfo.BankId()%x==0){
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 40;    
    } else if (r_xThreadInfo.BankId()%x==1){
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 33;
    } else if (r_xThreadInfo.BankId()%x==2){
        s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 0;
    }*/

    /*union {
        short shorty;
        char charty[2];
    } convy;*/
 
    r_conv.integer = g_startval +  r_xThreadInfo.FlatBlockId()*mWarpSize + r_xThreadInfo.BankId();
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = r_conv.chars[0];//r_xThreadInfo.FlatBlockId()*mWarpSize + r_xThreadInfo.BankId();
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[1] = r_conv.chars[1];//0;//192; //0; //192; //6; //162; //160 //128; //0;//32;//16;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[2] = r_conv.chars[2]; //2; //138; //138;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[3] = r_conv.chars[3];

    /*if(r_xThreadInfo.WarpId()==0){
    int x= 7;
    if(r_xThreadInfo.BankId()%x==0){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 1;//r_xThreadInfo.BankId();//g_startval + (blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId();//41; // 40; //41; //132; // 40; //33;//4; //33; //40
    } else if (r_xThreadInfo.BankId()%x==1){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 2;
    } else if (r_xThreadInfo.BankId()%x==2){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 3;
    } else if (r_xThreadInfo.BankId()%x==3){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 4;
    } else if (r_xThreadInfo.BankId()%x==4){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 5;
    } else if (r_xThreadInfo.BankId()%x==5){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 6;
    } else if (r_xThreadInfo.BankId()%x==6){
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[0] = 7;
    }
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[1] = 0;//192; //0; //192; //6; //162; //160 //128; //0;//32;//16;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[2] = 0; //2; //138; //138;
    s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[3] = 0; 
    }*/
    __syncthreads();

    s_xAssembly.Assemble(r_xThreadInfo, &s_xGenomeSet);
    __syncthreads();
    s_xAssembly.fEvaluateFitness(r_xThreadInfo); //DEBUG

//    if(r_xThreadInfo.WarpId()==0){
        //g_ucFitnessLeft[r_xThreadInfo.GlobId(1)] = 3; //s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
//        g_ucFitnessLeft[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.left[r_xThreadInfo.BankId()];//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;//(float)  s_xAssembly.data.assembly_size[r_xThreadInfo.BankId()];//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
//s_xAssembly.data.left[r_xThreadInfo.BankId()];//(int) s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
// (blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId();//s_xAssembly.data.gravity[r_xThreadInfo.BankId()].x;
//        g_ucFitnessBottom[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.bottom[r_xThreadInfo.BankId()]; //s_xAssembly.data.gravity[r_xThreadInfo.BankId()].y;
//s_xAssembly.data.bottom[r_xThreadInfo.BankId()];//(int) s_xAssembly.data.gravity[r_xThreadInfo.BankId()].y;
    //g_ucFitnessBottom[462] = 7;
//    }

    //for(int i=0;i<4;i++){
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.length.multi_d[i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = s_xEdgeSort.data.multi_d[6][0][i][r_xThreadInfo.BankId()];
        //s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[i] = tex2D(t_ucInteractionMatrix, i, 1);
    //}
    //s_xGenomeSet.CopyToGlobal(r_xThreadInfo, g_ucGenomes);

    __syncthreads();

    if(r_xThreadInfo.WarpId()==0){
	g_ucFitnessSize[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.assembly_size[r_xThreadInfo.BankId()];
        g_ucFitnessHash[(blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId()] = s_xAssembly.data.hash[r_xThreadInfo.BankId()].x;
    }

    __syncthreads();
    //Copy to grid
    for(int i=0;i<m_fit_DimGridY;i++){
         for(int j=0;j<m_fit_DimGridX;j++){
             xCell TMP = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0);
             g_ucGrids[r_xThreadInfo.FlatBlockId()*m_fit_DimGridX*m_fit_DimGridY*32 + r_xThreadInfo.BankId()*m_fit_DimGridX*m_fit_DimGridY + j*m_fit_DimGridX + i] = s_xAssembly.data.grid.get_xCell(r_xThreadInfo, i, j, 0).get_xCell(); //Orient, Type, Cell
         }
    }

    /*if(r_xThreadInfo.WarpId() == 0){
        //s_xGenomeSet.CopyToGlobal(r_xThreadInfo, g_ucGenomes);
        for(int i=0;i<mAlignedByteLengthGenome;i++){
            
        }
    }*/

    __syncthreads();
    if(r_xThreadInfo.WarpId() < mAlignedByteLengthGenome){
        g_ucGenomes[ ((blockIdx.y*m_fit_DimBlockX + blockIdx.x)*32+r_xThreadInfo.BankId())*mAlignedByteLengthGenome +r_xThreadInfo.WarpId()] = s_xGenomeSet.data.multi_d[r_xThreadInfo.BankId()].data.one_d[r_xThreadInfo.WarpId()];
    }   
}

