#ifndef _FLUID_SIM_CUH_
#define _FLUID_SIM_CUH_

#include <render/config.h>

// Dictates how boundary edges will be dealt with
enum BoundaryStrategy { Conserve = 0,   // Use same sign as interior neighbour
                        Reverse };      // Reverse sign of interior neighbour

// Allows for templating dynamically sized shared memory
template<typename T>
__device__ T* shared_memory_proxy() {
    extern __shared__ unsigned char memory[]; // TODO: Possible source of error due to lack of __align__
    return reinterpret_cast<T*>(memory);
}

/**
 * Add contribution of source cells to density field
 * 
 * @param densities Field of densities
 * @param sources Field of density-producing sources. Each entry represents how much density is produced for each time step
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param sim_params Simulation parameters
*/
template<typename T>
__global__ void add_sources(T* densities, T* sources, unsigned int num_cells, SimulationParams sim_params);

/**
 * Compute diffusion of densities across field of cells
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param sim_params Simulation parameters
*/
template<typename T>
__global__ void diffuse(T* old_field, T* new_field, uint2 field_extents, unsigned int num_cells, BoundaryStrategy bs, SimulationParams sim_params);


/**
 * Compute effect of vector field on density field
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param velocity_field Field describing velocity vectors at each cell of grid (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param sim_params Simulation parameters
*/
template<typename FieldT, typename FieldCudaT, typename VelocityT>
__global__ void advect(FieldT* old_field, FieldT* new_field, VelocityT* velocity_field,
                       uint2 field_extents, unsigned int num_cells,
                       BoundaryStrategy bs, SimulationParams sim_params);


/**
 * Handle the boundaries of a field. Can handle both global and shared memory fields
 * by passing offset and axial neighbour indices corresponding to shared memory.
 * 
 * @param field Field to be operated on
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param tidX Global x-axis coordinate of thread in grid
 * @param tidY Global y-axis coordinate of thread in grid
 * @param offset Index of thread's field cell
 * @param leftIdx Index of thread's left neighbour's cell
 * @param rightIdx Index of thread's right neighbour's cell
 * @param upIdx Index of thread's top neighbour's cell
 * @param downIdx Index of thread's bottom neighbour's cell
*/
template<typename T>
__device__ void handle_boundary(T* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx);


#endif
