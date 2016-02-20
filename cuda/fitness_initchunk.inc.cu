    /*-----------------------------------------------------------------*/
    /* Fitness - InitChunk --------------------------------------------*/                          
    /*-----------------------------------------------------------------*/
    /*
    Sets up EdgeSort Lists for all polyomino assemblies
    
    Requires definition of the following variables:
    unsigned char s_ucGenome[mByteLengthGenome]
    unsigned char *g_ucGenomes
    unsigned char s_ucEdgeSort[mNrEdgeTypes * mNrTileTypes * mNrTileOrientations * WarpSize]
    unsigned char s_ucListLength[mNrEdgeTypes]
    */
    /*-----------------------------------------------------------------*/
    
    /*ADAPTED, UNTESTED!*/
    
    //if(r_usFlatID < mByteLengthGenome){ //Copy genome from global memory
    for(int i = 0; i < mByteLengthGenome; i++){ //Copy genome from global memory
        //s_ucGenome[(int) r_usFlatID] = g_ucGenomes[ (int) (mBLOCK_ID * mByteLengthGenome + r_usFlatID) ];
        *fit_xMap( r_usBankID, (int) i, s_ucGenome ) = g_ucGenomes[ (int) (fit_xGlobalThreadAnchor(mByteLengthGenome) + i) ];
 //       if(threadIdx.x==0) printf("Genome[%d]:%d\n", i, *fit_xMap( r_usBankID, (int) i, s_ucGenome ) ); 
    }
    //__syncthreads();
    
    
    //if (r_usFlatID < mNrEdgeTypes){/*ThreadID < NrBondingEdges*/
    for(int k=0;k<mNrEdgeTypes;k++){
        //s_ucListLength[r_usFlatID] = 0;
        s_ucListLength[uiBankMapChar(k, r_usBankID)] = 0;
        bool r_bEdgeAdded = false;
        for(int i=0;i<mNrTileTypes;i++){/*Traverse all TileTypes*/ 
            for(int j=0;j<mNrTileOrientations;j++){/*Traverse all Orientations*/
    //            if(threadIdx.x==0) printf("Edge:%d::::Checking Tile %d at Orient %d -- Matrix: %f\n", k, i, j, tex2D( t_ucInteractionMatrix,
//                           fit_xGetEdgeTypeFromGenome(i, j, s_ucGenome, r_usBankID),
//                           k) );
                if( tex2D( t_ucInteractionMatrix,
                           fit_xGetEdgeTypeFromGenome(i, j, s_ucGenome, r_usBankID),
                           k ) > 0){ /*Does Edge j of Tile i bond to Tile ThreadID*/
                    //s_ucEdgeSort[r_usFlatID][s_ucListLength[r_usFlatID]][j] = i; /*Add Tile, Edge Orientation*/
                    *fit_xMap( r_usBankID, k, s_ucListLength[uiBankMapChar(k, r_usBankID)], j , s_ucEdgeSort) = i;
                    r_bEdgeAdded = true;
//	            if(threadIdx.x==0) printf("Edge was added!\n");
                } else {
                    //s_ucEdgeSort[r_usFlatID][s_ucListLength[r_usFlatID]][j] = 0; /*Empty entry*/
                    *fit_xMap( r_usBankID, k, s_ucListLength[uiBankMapChar(k, r_usBankID)], j , s_ucEdgeSort) = mEMPTY_CELL;
                }
            }
            if(r_bEdgeAdded){/*EdgeAdded?*/
                //s_ucListLength[r_usFlatID]++;
//                if(threadIdx.x==0) printf("Increasing ListLength...\n");
                s_ucListLength[uiBankMapChar(k, r_usBankID)]++;
//                if(threadIdx.x==0) printf("Increased ListLength to %d...\n", s_ucListLength[uiBankMapChar(k, r_usBankID)]);
                r_bEdgeAdded=0;
            } else {
                /*Do Nothing*/
            }
        }
        //__syncthreads();
    }// else {
      //  __syncthreads();
   // }

    //TEST START
    if(threadIdx.x==0){
        printf("Printing EdgeSort:\n");
        for(int i=0;i<mNrEdgeTypes;i++){
            printf("EdgeType %d (Length: %d): ", i, s_ucListLength[uiBankMapChar(i, r_usBankID)]);
	    for(int j=0;j<s_ucListLength[uiBankMapChar(i, r_usBankID)];j++){
                printf("[%d, %d, %d, %d],", *fit_xMap( r_usBankID, i, s_ucListLength[uiBankMapChar(j, r_usBankID)], 0 , s_ucEdgeSort),
                     *fit_xMap( r_usBankID, i, s_ucListLength[uiBankMapChar(j, r_usBankID)], 1 , s_ucEdgeSort),
                     *fit_xMap( r_usBankID, i, s_ucListLength[uiBankMapChar(j, r_usBankID)], 2 , s_ucEdgeSort),
                     *fit_xMap( r_usBankID, i, s_ucListLength[uiBankMapChar(j, r_usBankID)], 3 , s_ucEdgeSort) );           
            }
            printf("\n");
        }
    }
    //TEST STOP
    
//return; //TEST 
