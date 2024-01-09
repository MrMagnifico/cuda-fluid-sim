#ifndef _UTILS_CUH_
#define _UTILS_CUH_


// Indicates if a cell is in the interior of a 2D block or on one of the edges/corners
enum CellLocationType2d { Interior = 0,
                          Top, Bottom, Left, Right,
                          TopLeft, TopRight, BottomLeft, BottomRight };


__device__ CellLocationType2d determineLocationType() {
    // Corners
    if      (threadIdx.x == 0U && threadIdx.y == 0U)                                { return TopLeft; }    
    else if (threadIdx.x == (blockDim.x - 1U) && threadIdx.y == 0U)                 { return TopRight; }
    else if (threadIdx.x == 0U && threadIdx.y == (blockDim.y - 1U))                 { return BottomLeft; }
    else if (threadIdx.x == (blockDim.x - 1U) && threadIdx.y == (blockDim.y - 1U))  { return BottomRight; }
    // Edges
    else if (threadIdx.x == 0U)                 { return Left; }
    else if (threadIdx.x == (blockDim.x - 1U))  { return Right; }
    else if (threadIdx.y == 0U)                 { return Top; }
    else if (threadIdx.y == (blockDim.y - 1U))  { return Bottom; }
    // If it's neither a corner or edge, it's an interior
    return Interior;
}


#endif
