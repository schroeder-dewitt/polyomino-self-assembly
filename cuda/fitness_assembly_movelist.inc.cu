//Single-thread only (just one thread per bank, i.e. 32 threads per block supported!)

uchar2 CurrentPos = make_uchar2(m_fit_DimGridX/2, m_fit_DimGridY/2);

s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x+1, CurrentPos.y), s_s2MoveListStorage, r_usBankID);
s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x-1, CurrentPos.y), s_s2MoveListStorage, r_usBankID);
s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x, CurrentPos.y+1), s_s2MoveListStorage, r_usBankID);
s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x, CurrentPos.y-1), s_s2MoveListStorage, r_usBankID);

us2GRID2D r_xPixelIndexBuf2;
for(int i=0;i<m_fit_NrPixelInSubgrid;i++){
    r_xPixelIndexBuf2 = fit_xGridPixelAnchor(r_usWarpID, i);
    *fit_xMap(r_usBankID, m_fit_CurrentGrid, r_xPixelIndexBuf2.x, r_xPixelIndexBuf2.y, s_ucGrid) = mEMPTY_CELL;
}

//for(int i = 1;i < 5/*m_fit_DimGridX*/; i++){
//    for(int j=1;j < 5/*m_fit_DimGridY*/; j++){
//        (*fit_xMap(r_usBankID, m_fit_CurrentGrid, 0, 0, s_ucGrid)) = mEMPTY_CELL;
//    }
//}
//return; //TEST

(*fit_xMap(r_usBankID, m_fit_CurrentGrid, m_fit_DimGridX/2, m_fit_DimGridY/2, s_ucGrid)) = fit_xCell(0, 0);//First tile from edge

unsigned int TEST_A = 0; //TEST

bool r_bWasTileAssembled = false;
unsigned int r_uiEdgePermutationIndex;
unsigned int r_uiSitePermutationIndex;
ucCELL r_xNeighbourCell;
while( (s_ucMoveList[r_usBankID].pos != 0 ) && (!s_xFlag[r_usBankID].UNDCondition) ){
    TEST_A++;//TEST
	r_bWasTileAssembled = false;
	//Remove entry at CurrentIndex from MoveList
    CurrentPos = s_ucMoveList[r_usBankID].pop(s_s2MoveListStorage, r_usBankID);//MoveList[CurrentIndex];
    if(threadIdx.x==0) printf("New position (%d, %d): \n", CurrentPos.x, CurrentPos.y);
	//If there is no tile currently at CurrentPos
	//if(grid[CurrentPos.x][CurrentPos.y].tile == mEMPTY_CELL){
    if( (*fit_xMap(r_usBankID, m_fit_CurrentGrid, CurrentPos.x, CurrentPos.y, s_ucGrid)) 
        == mEMPTY_CELL){
        //if(threadIdx.x==0) printf("Cell at current position is: %d\n", (*fit_xMap(r_usBankID, m_fit_CurrentGrid, CurrentPos.x, CurrentPos.y, s_ucGrid)));
		//Choose random permutation through Neighbouring Sites
		r_uiSitePermutationIndex = curand_uniform(&g_xCurandStates[fit_xGlobalThreadAnchor(1)])*24;
		//Cycle through random permutation until we have finished cycle or there is a bonding tile
                if(threadIdx.x==0) printf("( SitePermIndices<%d> [%d, %d, %d, %d] )\n", r_uiSitePermutationIndex,
                        c_ucFourPermutations[r_uiSitePermutationIndex][0],
			c_ucFourPermutations[r_uiSitePermutationIndex][1], c_ucFourPermutations[r_uiSitePermutationIndex][2],
			c_ucFourPermutations[r_uiSitePermutationIndex][3]);
		unsigned int r_uiSiteRotIndex;
		for(int i=0;i<4;i++){
			r_uiSiteRotIndex = c_ucFourPermutations[r_uiSitePermutationIndex][i] - 1;
                     //   if(threadIdx.x==0) printf("[Cycling neighbour cells, current SiteRotIndex %d]\n", r_uiSiteRotIndex);
			r_xNeighbourCell = fit_xGetNeighbourCell( r_usBankID, CurrentPos, r_uiSiteRotIndex, m_fit_CurrentGrid, s_ucGrid);
                        if(threadIdx.x==0) printf("At site SiteRotIndex %d, detect cell (%d, %d) [%d, %d, %d, %d]\n", r_uiSiteRotIndex,
                            fit_xGetTileTypeFromCell(r_xNeighbourCell), fit_xGetOrientationFromCell(r_xNeighbourCell),
                            fit_xGetEdgeTypeFromCell (r_xNeighbourCell, 0,  s_ucGenome, r_usBankID),
                            fit_xGetEdgeTypeFromCell (r_xNeighbourCell, 1,  s_ucGenome, r_usBankID),
                            fit_xGetEdgeTypeFromCell (r_xNeighbourCell, 2,  s_ucGenome, r_usBankID),
                            fit_xGetEdgeTypeFromCell (r_xNeighbourCell, 3,  s_ucGenome, r_usBankID));
			if(r_xNeighbourCell != mEMPTY_CELL){
                      //  if(threadIdx.x==0) printf("Detected neighbour cell at %d to be %d...\n", r_uiSiteRotIndex, r_xNeighbourCell);
				r_uiEdgePermutationIndex = curand_uniform( &g_xCurandStates[ fit_xGlobalThreadAnchor(1) ] ) * 24;
				for(int j=0;j<4;j++){ //Go through all Edge-Sort rotations!
					unsigned int r_uiEdgeRotIndex = c_ucFourPermutations[r_uiEdgePermutationIndex][j] - 1;
                                    //    if(threadIdx.x==0) printf("EdgeRotIndex %d.\n", r_uiEdgeRotIndex);
					//If there is a bonding tile, put it down and end cycle
                                        if(threadIdx.x==0) printf("EdgeRotIndex %d, SiteRotIndex %d, MirroredSiteRotIndex %d\n",
                                             r_uiEdgeRotIndex, r_uiSiteRotIndex, (r_uiSiteRotIndex+2)%4);
					r_uiSiteRotIndex= (r_uiSiteRotIndex+2)%4; //Rot index at which neighbouring edge found
					//Check if there is a bonding cell at CurrentPos r_uiSiteRotIndex, Neighbour Position neigh_r_uiSiteRotIndex
					ucEDGETYPE r_xBufEdgeType = fit_xGetEdgeTypeFromCell (r_xNeighbourCell, r_uiSiteRotIndex,  s_ucGenome, r_usBankID);
				//	if(threadIdx.x==0) printf("Site is neighboured in direction %d by edge %d\n", i, r_uiSiteRotIndex);
					if(threadIdx.x==0) printf("BufEdgeType:%d\n", r_xBufEdgeType);
					ucEDGETYPE r_xBufOrientation = fit_xGetOrientationFromCell(r_xNeighbourCell);
                                        if(threadIdx.x==0) printf("Orientation:%d\n", r_xBufOrientation);
					unsigned int r_uiBufMirroredRotIndex = fit_xGetUnrotatedEdgeCycleIndex(r_xBufOrientation, r_uiSiteRotIndex);
					if(threadIdx.x==0) printf("MirroredSiteRotIndex: %d\n", r_uiBufMirroredRotIndex);
					ucCELL r_xBondingCell = *fit_xMap (r_usBankID, r_xBufEdgeType, 0, r_uiEdgeRotIndex, s_ucEdgeSort);
                    			ucTILETYPE r_xBondingCellTileType = fit_xGetEdgeTypeFromCell( 
						//*fit_xMap (r_usBankID, r_xBufEdgeType, 0, r_uiSiteRotIndex, s_ucEdgeSort),
						r_xBondingCell,
                                                r_uiBufMirroredRotIndex, s_ucGenome, r_usBankID );
					if(threadIdx.x==0) printf("BondingCell:%d\n", 
						*fit_xMap (r_usBankID, r_xBufEdgeType, 0, r_uiSiteRotIndex, s_ucEdgeSort));
					if(threadIdx.x==0) printf("BondingCellTileType: %d\n", r_xBondingCellTileType);
					if( r_xBondingCell != mEMPTY_CELL){
						unsigned int r_uiBondingCellOrientation = (int) ( 4 - (r_uiEdgeRotIndex - r_uiSiteRotIndex) ) % (int) 4;
                                                (*fit_xMap(r_usBankID, m_fit_CurrentGrid, CurrentPos.x, CurrentPos.y, s_ucGrid))
                                                     = fit_xCell( r_xBondingCellTileType, r_uiBondingCellOrientation );
                                                if(threadIdx.x==0){
					              printf("Assembled Tile (%d, %d) at pos (%d, %d) !\n",
						            r_xBondingCellTileType, r_uiBondingCellOrientation, CurrentPos.x, CurrentPos.y); 
                                                }
                                                if(fit_bIsBorderCell(CurrentPos)){
                                                    s_xFlag[r_usBankID].UNDCondition = true;
                                  //                  if(threadIdx.x==0) printf("Border cell (%d, %d)! UND condition raised!\n", CurrentPos.x, CurrentPos.y);
                                                }
						r_bWasTileAssembled = true;
						break;
					} else {
//						printf("Did not find binding cell (WEIRD)... \n");
					}
				}
				if(r_bWasTileAssembled || s_xFlag[r_usBankID].UNDCondition) break;
			} else {
	//			if(threadIdx.x==0) printf("Neighbouring cell %d turned out to be empty...\n", r_uiSiteRotIndex);
			}
		}
	} else {
//		printf("Current position is NOT empty...\n");
	//ELSE If there is already a tile at CurrentPos
		//Cycle through standard permutation until we have finished cycle
			//If there is a bonding tile, compare with existing tile
				//If bonding tile is not the same as existing tile
					//Raise StericUND
	}
	//Cycle through all four neighbouring sites of CurrentPosif we assembled a grid element there (standard permutation)
	if(r_bWasTileAssembled && (!s_xFlag[r_usBankID].UNDCondition) ){
                if (threadIdx.x==0) printf("Seeing if we can add new tiles to movelist...\n");
		ucCELL r_xTempNeighbourCell;
		for(int j=0;j<4;j++){
		//If that position is empty
			r_xTempNeighbourCell = fit_xGetNeighbourCell(r_usBankID, CurrentPos, j, s_xFlag[r_usBankID].RedID, s_ucGrid); 
			if(threadIdx.x==0) printf("Scanned neigh cell at dir %d, has cell type %d.\n", j, r_xTempNeighbourCell);
			if(r_xTempNeighbourCell == mEMPTY_CELL){
                                if(threadIdx.x==0) printf("__Movelist Storage used: %d__\n", s_ucMoveList[r_usBankID].pos);
				//If neighbouring tile edge from CurrentPos != 0
				if(j==0) //NORTH
                                    s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x, CurrentPos.y-1), s_s2MoveListStorage, r_usBankID);
				else if(j==1) //EAST
                                    s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x+1, CurrentPos.y), s_s2MoveListStorage, r_usBankID);
				else if(j==2) //SOUTH
                                    s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x, CurrentPos.y+1), s_s2MoveListStorage, r_usBankID);
				else if(j==3) //WEST
                                    s_ucMoveList[r_usBankID].push(make_uchar2(CurrentPos.x-1, CurrentPos.y), s_s2MoveListStorage, r_usBankID);
                                if(threadIdx.x==0) printf("Added new entry to movelist in direction: %d\n", j);
			}
		}
	}
}
if(threadIdx.x==0) printf("(Thrd: [%d,%d,%d], Block:[%d,%d]) LOOPED %d TIMES!\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,  TEST_A);

