/*-----------------------------------------------------------------*/
/* Fitness - FitnessChunk -----------------------------------------*/                          
/*-----------------------------------------------------------------*/
/*
Compares all fitnessgrids with assembled polyominoes and 

Requires definition of the following variables:
	bool s_bUNDCondition
	xMutex x_fit_Mutex
	unsigned char s_ucGrid[m_fit_NrAssemblyRedundancy * m_fit_DimGridX
			       * m_fit_DimGridY * warpSize]
        unsigned char *g_ucAssembledGrids
        float *g_fFitnessValues
*/
/*-----------------------------------------------------------------*/

    /*ADAPTED, UNTESTED!*/

    float r_fFitBuf = 0;  

    //if(!s_bUNDCondition){/*!(Was UND condition raised)?*/
    if(!s_xFlag[r_usBankID].UNDCondition){
        //return; //TEST
        s_fFitnessValue[r_usBankID] = 0;
        //__syncthreads();
        us2GRID2D r_xPixelIndexBuf;
        //return; //TEST

    //START TEMPLATE-GENERATED CODE
    {% for tex in textures %}
        for(int k=0; k<m_fit_NrPixelInSubgrid; k++){
            r_xPixelIndexBuf = fit_xGridPixelAnchor(r_usWarpID, k);  
            /*Note: we want entry in FitnessFunctionGrids to be 0xFF (i.e. non-empty tile)*/              
            //if( tex2D( t_ucFitnessFunctionGrids{{ tex }}, r_xPixelIndexBuf.x, r_xPixelIndexBuf.y ) == fit_xFFOrNIL( s_ucGrid[m_fit_CurrentGrid][r_xPixelIndexBuf.x][r_xPixelIndexBuf.y] + 1) ){
            if( tex2D( t_ucFitnessFunctionGrids{{ tex }}, r_xPixelIndexBuf.x, r_xPixelIndexBuf.y ) ==
                fit_xFFOrNIL( *fit_xMap(r_usBankID, s_xFlag[r_usBankID].RedID, r_xPixelIndexBuf.x, r_xPixelIndexBuf.y, s_ucGrid) + 1) ){
                //u_Lock(g_xMutexe[r_usBankID]); //Only if not using shared mem atomics, then can do simple atomicInc!
                s_fFitnessValue[r_usBankID]++; /*FitnessValue++*/
                //u_Unlock(g_xMutexe[r_usBankID]);
                break;
            }                       
        }
        //if( (mTHREAD_ID_X==0)&&(mTHREAD_ID_Y==0) ){
        if( r_usBankID == 0 ){
            if( r_fFitBuf < s_fFitnessValue[r_usBankID] ){
                r_fFitBuf = s_fFitnessValue[r_usBankID];
                s_fFitnessValue[r_usBankID] = 0; //UNSURE ABOUT THIS ONE...
            }
        }
        //__syncthreads();
    {% endfor %}        
    //END TEMPLATE-GENERATED CODE
    //return; //TEST
    } else {
        //if( (mTHREAD_ID_X==0)&&(mTHREAD_ID_Y==0) ){/*Thread 0 only: FitnessValue=0*/
        if(r_usBankID == 0){
            s_fFitnessValue[r_usBankID] = 0;
        }
    }   
 
    /*WriteToGlobal Memory: FitnessValue, (Grid)*/
    //if( (mTHREAD_ID_X==0)&&(mTHREAD_ID_Y==0) ){
    if(r_usBankID == 0){
        g_fFFValues[fit_xGlobalBlockAnchor(r_usBankID, 1)] = r_fFitBuf; 
    }



    //TEST START
    //if(mTHREAD_ID_X == 1){
    //*fit_xMap(r_usBankID, m_fit_CurrentGrid, 2, 2, s_ucGrid) = 22; 
    //*fit_xMap(r_usBankID, m_fit_CurrentGrid, 1, 2, s_ucGrid) = 12; 
    //*fit_xMap(r_usBankID, m_fit_CurrentGrid, 2, 1, s_ucGrid) = 21; 
    //*fit_xMap(r_usBankID, m_fit_CurrentGrid, 1, 1, s_ucGrid) = 11; 
    //*fit_xMap(r_usBankID, m_fit_CurrentGrid, 0, 0, s_ucGrid) = 255; 
    //s_ucGrid[0]=1;
    //s_ucGrid[1]=2;
    //s_ucGrid[2]=3;
    //s_ucGrid[3]=4;
    //}
    //TEST STOP
    
    //TEST START
//    (*fit_xMap(r_usBankID, m_fit_CurrentGrid, m_fit_DimGridX/2, m_fit_DimGridY/2, s_ucGrid)) = fit_xCell(3, 3);//First tile from edge
    //TEST STOP

    //TESTED - SEEMS TO WORK - NOT ANYMORE!
    us2GRID2D r_xPixelIndexBuf;
    for(int i=0;i<m_fit_NrPixelInSubgrid;i++){    
            //r_xPixelIndexBuf = fit_xSharedCellAnchor2D(fit_xGridPixelAnchor(i)); 
            r_xPixelIndexBuf = fit_xGridPixelAnchor(r_usWarpID, i);

            //g_ucAssembledGrids[mBLOCK_ID * m_fit_DimGridX * m_fit_DimGridY * WarpSize + m_fit_DimGridX * m_fit_DimGridY * r_usBankID + r_xPixelIndexBuf.x * m_fit_DimGridY + r_xPixelIndexBuf.y] = 
            g_ucAssembledGrids[ fit_xGlobalBlockAnchor( r_usBankID, m_fit_DimGridX * m_fit_DimGridY ) + r_xPixelIndexBuf.y * m_fit_DimGridX + r_xPixelIndexBuf.x] = 
                fit_xGetTileTypeFromCell(*fit_xMap(r_usBankID, m_fit_CurrentGrid, r_xPixelIndexBuf.x, r_xPixelIndexBuf.y, s_ucGrid));
                //uiBankMapChar(i, r_usBankID);//*fit_xMap(r_usBankID, m_fit_CurrentGrid, i % m_fit_DimSubgridX, (i - i%m_fit_DimSubgridX) / m_fit_DimSubgridY, s_ucGrid);
            //TEST START
            //g_ucAssembledGrids[mBLOCK_ID * m_fit_DimGridX * m_fit_DimGridY + r_xPixelIndexBuf.x * m_fit_DimGridY + r_xPixelIndexBuf.y] = i;
            //TEST STOP
    }
