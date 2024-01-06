#ifndef _FLUID_SIM_CUH_
#define _FLUID_SIM_CUH_

enum BoundaryStrategy { Conserve = 0, ReverseVertical, ReverseHorizontal };

/**
 * Add contribution of source cells to density field
 * 
 * @param densities Field of densities
 * @param sources Field of density-producing sources. Each entry represents how much density is produced for each time step
 * @param time_step Magnitude of simulation step
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
*/
__global__ void add_sources(float3* densities, float3* sources, float time_step, unsigned int num_cells);

/**
 * Compute diffusion of densities across field of cells
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param b ???
 * @param time_step Magnitude of simulation step
 * @param diffusion_rate Rate at which density diffuses through cells
 * @param sim_steps Number of Gauss-Seidel relaxation steps to use for iteration 
*/
__global__ void diffuse(float3* old_field, float3* new_field, uint2 field_extents, unsigned int num_cells,
                        float time_step, float diffusion_rate, unsigned int sim_steps);

__device__ void handle_boundary(float3* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx);


#endif
