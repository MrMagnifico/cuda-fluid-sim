#ifndef _FLUID_SIM_CUH_
#define _FLUID_SIM_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

// Dictates how boundary edges will be dealt with
enum BoundaryStrategy { Conserve = 0,           // Use same sign as interior neighbour
                        ReverseVertical,        // Reverse sign of interior neighbour if cell lies on top/bottom edge
                        ReverseHorizontal };    // Reverse sign of interior neighbour if cell lies on left/right edge     

/**
 * Add contribution of source cells to density field
 * 
 * @param densities Field of densities
 * @param sources Field of density-producing sources. Each entry represents how much density is produced for each time step
 * @param time_step Magnitude of simulation step
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
*/
__global__ void add_sources(glm::vec3* densities, glm::vec3* sources, float time_step, unsigned int num_cells);

/**
 * Compute diffusion of densities across field of cells
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param time_step Magnitude of simulation step
 * @param diffusion_rate Rate at which density diffuses through cells
 * @param sim_steps Number of Gauss-Seidel relaxation steps to use for iteration 
*/
__global__ void diffuse(glm::vec3* old_field, glm::vec3* new_field, uint2 field_extents, unsigned int num_cells,
                        float time_step, float diffusion_rate, unsigned int sim_steps);


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
__device__ void handle_boundary(glm::vec3* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx);


#endif
