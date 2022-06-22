#include "pfem2particle.h"

#include <iostream>
#include <fstream>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_poly.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_handler.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include "omp.h"

using namespace dealii;

pfem2Particle::pfem2Particle(const Point<2> & location,const Point<2> & reference_location,const unsigned id)
	: id (id)
{
    this->location[0] = location[0];
	this->location[1] = location[1];
	this->reference_location[0] = reference_location[0];
	this->reference_location[1] = reference_location[1];
	this->velocity[0] = 0.0;
	this->velocity[1] = 0.0;
}

pfem2Particle::pfem2Particle()
	: pfem2Particle(Point<2>(), Point<2>(), 0)
{
}

void pfem2Particle::set_location (const Point<2> &new_location)
{
	location[0] = new_location[0];
	location[1] = new_location[1];
}

const Point<2> pfem2Particle::get_location () const
{
    return Point<2>(location[0], location[1]);
}

void pfem2Particle::set_reference_location (const Point<2> &new_reference_location)
{
    reference_location[0] = new_reference_location[0];
	reference_location[1] = new_reference_location[1];
}

const Point<2> pfem2Particle::get_reference_location () const
{
    return Point<2>(reference_location[0], reference_location[1]);
}

unsigned int pfem2Particle::get_id () const
{
    return id;
}

void pfem2Particle::set_cell_dofs(const typename DoFHandler<2>::active_cell_iterator &cell)
{
	for(unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i) cell_dofs[i] = cell->vertex_dof_index(i, 0);
}

void pfem2Particle::set_tria_position(const int &new_position)
{
	tria_position = new_position;
}

int pfem2Particle::get_tria_position() const
{
	return tria_position;
}

const Tensor<1,2> pfem2Particle::get_velocity() const
{
	return Tensor<1,2>({velocity[0],velocity[1]});
}

const Tensor<1,2> pfem2Particle::get_velocity_ext() const
{
	return Tensor<1,2>({velocity_ext[0],velocity_ext[1]});
}

double pfem2Particle::get_velocity_component(int component) const
{
	return velocity[component];
}

void pfem2Particle::set_velocity (const Tensor<1,2> &new_velocity)
{
	velocity[0] = new_velocity[0];
	velocity[1] = new_velocity[1];
}

void pfem2Particle::set_velocity_component (const double value, int component)
{
	velocity[component] = value;
}

void pfem2Particle::set_velocity_ext (const Tensor<1,2> &new_ext_velocity)
{
	velocity_ext[0] = new_ext_velocity[0];
	velocity_ext[1] = new_ext_velocity[1];
}

Triangulation<2>::cell_iterator pfem2Particle::get_surrounding_cell(const Triangulation<2> &triangulation) const
{
	const typename Triangulation<2>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position);
	
	return cell;
}

DoFHandler<2>::cell_iterator pfem2Particle::get_surrounding_cell(const Triangulation<2> &triangulation, const DoFHandler<2> &dof_handler) const
{
	const typename DoFHandler<2>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position, &dof_handler);
	
	return cell;
}

unsigned int pfem2Particle::find_closest_vertex_of_cell(const typename Triangulation<2>::active_cell_iterator &cell, const Mapping<2> &mapping)
{
	//transformation of local particle coordinates transformation is required as the global particle coordinates have already been updated by the time this function is called
	const Point<2> old_position = mapping.transform_unit_to_real_cell(cell, get_reference_location());
	
	Tensor<1,2> velocity_normalized = get_velocity_ext() / get_velocity_ext().norm();
	Tensor<1,2> particle_to_vertex = cell->vertex(0) - old_position;
    particle_to_vertex /= particle_to_vertex.norm();
    
    double maximum_angle = velocity_normalized * particle_to_vertex;
    unsigned int closest_vertex = 0;
    
    for (unsigned int v = 1; v < GeometryInfo<2>::vertices_per_cell; ++v){
		particle_to_vertex = cell->vertex(v) - old_position;
		particle_to_vertex /= particle_to_vertex.norm();
		const double v_angle = velocity_normalized * particle_to_vertex;
		
		if (v_angle > maximum_angle){
			closest_vertex = v;
			maximum_angle = v_angle;
		}
	}
	
	return closest_vertex;
}

std::size_t pfem2Particle::serialized_size_in_bytes() const
{
	std::size_t size = sizeof(id) + sizeof(location) + sizeof(reference_location) + sizeof(tria_position)
		+ sizeof(velocity) + sizeof(velocity_ext);

	return size;
}

pfem2ParticleHandler::pfem2ParticleHandler(const parallel::distributed::Triangulation<2> &tria, const Mapping<2> &coordMapping)
	: triangulation(&tria, typeid(*this).name())
	, mapping(&coordMapping, typeid(*this).name())
	, particles()
    {}
    
pfem2ParticleHandler::~pfem2ParticleHandler()
{
	clear_particles();
}

void pfem2ParticleHandler::initialize_maps()
{
	vertex_to_cells = std::vector<std::set<typename Triangulation<2>::active_cell_iterator>>(GridTools::vertex_to_cell_map(*triangulation));
    vertex_to_cell_centers = std::vector<std::vector<Tensor<1,2>>>(GridTools::vertex_to_cell_centers_directions(*triangulation,vertex_to_cells));	  
}

void pfem2ParticleHandler::clear_particles()
{
	particles.clear();
}

std::vector<pfem2Particle>::iterator pfem2ParticleHandler::remove_particle(std::vector<pfem2Particle>::iterator particleIndex)
{
	return particles.erase(particleIndex);
}

void pfem2ParticleHandler::insert_particle(pfem2Particle &particle, const typename DoFHandler<2>::active_cell_iterator &cell)
{
	particle.set_tria_position(cell->index());
	particle.set_cell_dofs(cell);
	particles.push_back(particle);
}

unsigned int pfem2ParticleHandler::n_global_particles() const
{
	return particles.size();
}

bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,2> &particle_direction, const std::vector<Tensor<1,2> > &center_directions)
{
	const double scalar_product_a = center_directions[a] * particle_direction;
    const double scalar_product_b = center_directions[b] * particle_direction;

    return scalar_product_a > scalar_product_b;
}

void pfem2ParticleHandler::sort_particles_into_subdomains_and_cells(const DoFHandler<2> &dof_handler)
{
	//std::cout << "Sorting particles" << std::endl;

	for(auto it = begin(); it != end(); ){
		const typename Triangulation<2>::cell_iterator cell = (*it).get_surrounding_cell(*triangulation);
		
        bool found_cell = false;
		try{
			const Point<2> p_unit = mapping->transform_real_to_unit_cell(cell, (*it).get_location());
		
			if(GeometryInfo<2>::is_inside_unit_cell(p_unit)){
				(*it).set_reference_location(p_unit);
				found_cell = true;
				++it;
			}
		} catch(typename Mapping<2>::ExcTransformationFailed &){
#ifdef VERBOSE_OUTPUT
			std::cout << "Transformation failed for particle with global coordinates " << (*it).get_location() << " (checked cell index #" << cell->index() << ")" << std::endl;
#endif // VERBOSE_OUTPUT
		}
		
		if(!found_cell){
			std::vector<unsigned int> neighbor_permutation;

			Point<2> current_reference_position;
			typename Triangulation<2>::active_cell_iterator current_cell = (*it).get_surrounding_cell(*triangulation);

			const unsigned int closest_vertex = (*it).find_closest_vertex_of_cell(current_cell, *mapping);
			Tensor<1,2> vertex_to_particle = (*it).get_location() - current_cell->vertex(closest_vertex);
			vertex_to_particle /= vertex_to_particle.norm();

			const unsigned int closest_vertex_index = current_cell->vertex_index(closest_vertex);
			const unsigned int n_neighbor_cells = vertex_to_cells[closest_vertex_index].size();

			neighbor_permutation.resize(n_neighbor_cells);
			for (unsigned int i=0; i<n_neighbor_cells; ++i) neighbor_permutation[i] = i;

			std::sort(neighbor_permutation.begin(), neighbor_permutation.end(),
				std::bind(&compare_particle_association, std::placeholders::_1, std::placeholders::_2, std::cref(vertex_to_particle), std::cref(vertex_to_cell_centers[closest_vertex_index])));
			
			for (unsigned int i=0; i<n_neighbor_cells; ++i){
				typename std::set<typename Triangulation<2>::active_cell_iterator>::const_iterator cell = vertex_to_cells[closest_vertex_index].begin();
				std::advance(cell, neighbor_permutation[i]);
              
				try{
					const Point<2> p_unit = mapping->transform_real_to_unit_cell(*cell, (*it).get_location());
					if (GeometryInfo<2>::is_inside_unit_cell(p_unit)){
						current_cell = *cell;
						(*it).set_reference_location(p_unit);
						(*it).set_tria_position(current_cell->index());
					
						const typename DoFHandler<2>::cell_iterator dofCell(triangulation, triangulation->n_levels() - 1, current_cell->index(), &dof_handler);
						(*it).set_cell_dofs(dofCell);
					
						found_cell = true;
					
						break; 
					}
                } catch(typename Mapping<2>::ExcTransformationFailed &)
                { }
            }
          
			if (!found_cell){
				*it = std::move(particles.back());
				particles.pop_back();
			} else ++it;
		}
	}
}

std::vector<pfem2Particle>::iterator pfem2ParticleHandler::begin()
{
	return particles.begin();
}

std::vector<pfem2Particle>::iterator pfem2ParticleHandler::end()
{
	return particles.end();
}

pfem2Solver::pfem2Solver()
	: mpi_communicator (MPI_COMM_WORLD),
	tria(mpi_communicator,Triangulation<2>::maximum_smoothing),
	particle_handler(tria, mapping),
	feV (1),
	feP (1),
	fe(FE_Q<2>(1), 1),
	dof_handlerV (tria),
	dof_handlerP (tria),
	quadrature_formula(2),
	face_quadrature_formula(2),
	feV_values (feV, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feP_values (feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feV_face_values (feV, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	feP_face_values (feP, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	dofs_per_cellV (feV.dofs_per_cell),
	dofs_per_cellP (feP.dofs_per_cell),
	local_dof_indicesV (dofs_per_cellV),
	local_dof_indicesP (dofs_per_cellP),
	n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
	pcout (std::cout,(this_mpi_process == 0)),
	n_q_points (quadrature_formula.size()),
	n_face_q_points (face_quadrature_formula.size()),
	quantities({0,0})
{
	setCudaConstants();
}

pfem2Solver::~pfem2Solver()
{

}

void pfem2Solver::seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell)
{
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	double shapeValue;
	
	for(unsigned int i = 0; i < quantities[0]; ++i){
		for(unsigned int j = 0; j < quantities[1]; ++j){
			pfem2Particle particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
						
			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				shapeValue = fe.shape_value(vertex, particle.get_reference_location());

				particle.set_velocity_component(particle.get_velocity_component(0) + shapeValue * locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)), 0);
				particle.set_velocity_component(particle.get_velocity_component(1) + shapeValue * locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)), 1);
			}//vertex
			
			particle_handler.insert_particle(particle, cell);
		}
	}
}

bool pfem2Solver::check_cells_for_empty_parts ()
{
	bool res = false;

	std::map<int, std::map<std::vector<unsigned int>, unsigned int>> particlesInCellParts;
	std::vector<std::vector<pfem2Particle>::iterator> particles_to_be_deleted;

	//определение, в каких частях ячейки лежат частицы
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	unsigned int num_x, num_y;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ){
		num_x = (*particleIndex).get_reference_location()(0)/hx;
		num_y = (*particleIndex).get_reference_location()(1)/hy;
		
		if(particlesInCellParts[(*particleIndex).get_tria_position()][{num_x,num_y}] > MAX_PARTICLES_PER_CELL_PART){
			*particleIndex = std::move(particle_handler.particles.back());
			particle_handler.particles.pop_back();
			res = true;
		} else {
			particlesInCellParts[(*particleIndex).get_tria_position()][{num_x,num_y}]++;
			++particleIndex;
		}
	}
	
	double shapeValue;
	
	//проверка каждой части ячейки на количество частиц: при 0 - подсевание 1 частицы в центр
	for(auto cellInfo = particlesInCellParts.begin(); cellInfo != particlesInCellParts.end(); ++cellInfo){
		const DoFHandler<2>::cell_iterator cell(&tria, tria.n_levels() - 1, (*cellInfo).first, &dof_handlerV);

		for(unsigned int i = 0; i < quantities[0]; i++)
			for(unsigned int j = 0; j < quantities[1]; j++)
				if((*cellInfo).second[{i,j}] == 0){
					pfem2Particle particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
					
					for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
						shapeValue = fe.shape_value(vertex, particle.get_reference_location());

						particle.set_velocity_component(particle.get_velocity_component(0) + shapeValue * locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)), 0);
						particle.set_velocity_component(particle.get_velocity_component(1) + shapeValue * locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)), 1);
					}//vertex

					particle_handler.insert_particle(particle, cell);
					
					res = true;
				}
	}
	
	//удаление лишних частиц
	//for(std::vector<std::vector<pfem2Particle>::iterator>::reverse_iterator it = particles_to_be_deleted.rbegin(); it != particles_to_be_deleted.rend(); ++it) particle_handler.remove_particle(*it);
		
	//if(!particles_to_be_deleted.empty()) res = true;
	
	return res;
}

void pfem2Solver::seed_particles(const std::vector < unsigned int > & quantities)
{
	TimerOutput::Scope timer_section(*timer, "Particles' seeding");
	
	if(quantities.size() < 2){ return; }
	
	this->quantities = quantities;
	
	typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels()-1), endc = dof_handlerV.end(tria.n_levels()-1);
	for (; cell != endc; ++cell)
		if (cell->is_locally_owned()) seed_particles_into_cell(cell);
	
	std::cout << "Created and placed " << particleCount << " particles" << std::endl;
	std::cout << "Particle handler contains " << particle_handler.n_global_particles() << " particles" << std::endl;
}

__global__ void correct_particle_velocities_cuda (const unsigned int Numparticles, pfem2Particle *particles, const double *deltaVx, const double *deltaVy)
 {
	int i = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	
	if (i < Numparticles){ 
		double shapeValue;
	    double vel_in_partVx = 0.0;
	    double vel_in_partVy = 0.0;
		
		double *velocity = (double*)((char*)&particles[0] + i * particleSize + velocityPos);
		double *refLocation = (double*)((char*)&particles[0] + i * particleSize + refLocationPos);
		int *cellDofs = (int*)((char*)&particles[0] + i * particleSize + cellDoFsPos);
				
		shapeValue = (1 - refLocation[0]) * (1 - refLocation[1]);
		vel_in_partVx += shapeValue * deltaVx[cellDofs[0]];
		vel_in_partVy += shapeValue * deltaVy[cellDofs[0]];
		shapeValue = refLocation[0] * (1 - refLocation[1]);
		vel_in_partVx += shapeValue * deltaVx[cellDofs[1]];
		vel_in_partVy += shapeValue * deltaVy[cellDofs[1]];
		shapeValue = (1 - refLocation[0]) * refLocation[1];
		vel_in_partVx += shapeValue * deltaVx[cellDofs[2]];
		vel_in_partVy += shapeValue * deltaVy[cellDofs[2]];
		shapeValue = refLocation[0] * refLocation[1];
		vel_in_partVx += shapeValue * deltaVx[cellDofs[3]];
		vel_in_partVy += shapeValue * deltaVy[cellDofs[3]];
		
		velocity[0] += vel_in_partVx;
		velocity[1] += vel_in_partVy;
	
		*(((char*)&particles[0] + i * particleSize + velocityPos)) = *velocity;
	}	
}

void pfem2Solver::correct_particles_velocities()
{
	TimerOutput::Scope timer_section(*timer, "Particles' velocities correction");
	
	cudaError_t err = cudaSuccess;
 	
 	//Mesh node velocity difference vectors
 	size_t meshVectorSize = dof_handlerV.n_dofs() * sizeof(double);
 	
 	//Vx
 	double *hostDeltaVx = (double *)malloc(meshVectorSize);
	if (hostDeltaVx == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    for(unsigned int i = 0; i < dof_handlerV.n_dofs(); ++i) hostDeltaVx[i] = locally_relevant_solutionVx[i] - locally_relevant_old_solutionVx[i];
   	
	double *deviceDeltaVx = NULL;
    err = cudaMalloc((void **)&deviceDeltaVx, meshVectorSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector deviceDeltaVx (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(deviceDeltaVx, hostDeltaVx, meshVectorSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector hostDeltaVx from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Vy
 	double *hostDeltaVy = (double *)malloc(meshVectorSize);
    if (hostDeltaVy == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
 	for(unsigned int i = 0; i < dof_handlerV.n_dofs(); ++i) hostDeltaVy[i] = locally_relevant_solutionVy[i] - locally_relevant_old_solutionVy[i];
 	
	double *deviceDeltaVy = NULL;
    err = cudaMalloc((void **)&deviceDeltaVy, meshVectorSize);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector deviceDeltaVy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(deviceDeltaVy, hostDeltaVy, meshVectorSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector hostDeltaVy from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	dim3 grid_dim, block_dim;
		
	constexpr unsigned int particles_per_block = CUDAWrappers::warp_size;
	const double apply_n_blocks = std::ceil(static_cast<double>(particle_handler.n_global_particles()) / static_cast<double>(particles_per_block));
	const unsigned int apply_x_n_blocks = std::round(std::sqrt(apply_n_blocks));
	const unsigned int apply_y_n_blocks = std::ceil(apply_n_blocks / static_cast<double>(apply_x_n_blocks));

	grid_dim = dim3(apply_x_n_blocks, apply_y_n_blocks);
	block_dim = dim3(particles_per_block);
	
	pfem2Particle *deviceParticles = NULL;
	err = cudaMalloc((void **)&deviceParticles, particle_handler.n_global_particles() * sizeof(pfem2Particle));  
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector deviceParticles (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(deviceParticles, particle_handler.particles.data(), particle_handler.n_global_particles() * sizeof(pfem2Particle), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector hostParticles from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	correct_particle_velocities_cuda<<<grid_dim, block_dim>>>(particle_handler.n_global_particles(), deviceParticles, deviceDeltaVx, deviceDeltaVy);
	AssertCudaKernel ();
  
	err = cudaGetLastError();

	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch move_particles_cuda kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	timer->enter_subsection("Memory copy");
	err = cudaMemcpy(particle_handler.particles.data(), deviceParticles, particle_handler.n_global_particles() * sizeof(pfem2Particle), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector deviceParticles from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer->leave_subsection();
	
	cudaFree(deviceParticles);
	cudaFree(deviceDeltaVx);
	cudaFree(deviceDeltaVy);
    free(hostDeltaVx);
    free(hostDeltaVy);
	
	//std::cout << "Finished correcting particles' velocities" << std::endl;	
}

__global__ void move_particles_cuda ( const unsigned int Numparticles, const double time_step,
                                       pfem2Particle *particles, const double *solutionVx, const double *solutionVy)
 {
	int i = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	
	if (i < Numparticles){ 
		double shapeValue;
	    double vel_in_partVx = 0.0;
	    double vel_in_partVy = 0.0;
		
		double *location = (double*)((char*)&particles[0] + i * particleSize + locationPos);
		double *refLocation = (double*)((char*)&particles[0] + i * particleSize + refLocationPos);
		int *cellDofs = (int*)((char*)&particles[0] + i * particleSize + cellDoFsPos);
		//printf("Particle %d has coordinates (%f, %f)\n", i, location[0], location[1]);
		//printf("Particle %d has local coordinates (%f, %f)\n", i, refLocation[0], refLocation[1]);
		//printf("Particle %d has cell with dofs (%d, %d, %d, %d)\n", i, cellDofs[0], cellDofs[1], cellDofs[2], cellDofs[3]);
				
		shapeValue = (1 - refLocation[0]) * (1 - refLocation[1]);
		vel_in_partVx += shapeValue * solutionVx[cellDofs[0]];
		vel_in_partVy += shapeValue * solutionVy[cellDofs[0]];
		shapeValue = refLocation[0] * (1 - refLocation[1]);
		vel_in_partVx += shapeValue * solutionVx[cellDofs[1]];
		vel_in_partVy += shapeValue * solutionVy[cellDofs[1]];
		shapeValue = (1 - refLocation[0]) * refLocation[1];
		vel_in_partVx += shapeValue * solutionVx[cellDofs[2]];
		vel_in_partVy += shapeValue * solutionVy[cellDofs[2]];
		shapeValue = refLocation[0] * refLocation[1];
		vel_in_partVx += shapeValue * solutionVx[cellDofs[3]];
		vel_in_partVy += shapeValue * solutionVy[cellDofs[3]];
		
		location[0] += time_step * vel_in_partVx;
		location[1] += time_step * vel_in_partVy;
		*((char*)&particles[0] + i * particleSize + locationPos) = *location;
	
		*((double*)((char*)&particles[0] + i * particleSize + velocityExtPos)) = vel_in_partVx;
		*((double*)((char*)&particles[0] + i * particleSize + velocityExtPos + sizeof(double))) = vel_in_partVy;
	}	
}
                                     
void pfem2Solver::move_particles() //перенос частиц
{
	TimerOutput::Scope timer_section(*timer, "Particles' movement");	
  
	double min_time_step = time_step / PARTICLES_MOVEMENT_STEPS;

 	cudaError_t err = cudaSuccess;
 	
 	//Mesh node velocity vectors
 	size_t meshVectorSize = dof_handlerV.n_dofs() * sizeof(double);
 	
 	//Vx
 	double *hostSolutionVx = (double *)malloc(meshVectorSize);
	if (hostSolutionVx == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    for(unsigned int i = 0; i < dof_handlerV.n_dofs(); ++i) hostSolutionVx[i] = locally_relevant_solutionVx[i];
   	
	double *deviceSolutionVx = NULL;
    err = cudaMalloc((void **)&deviceSolutionVx, meshVectorSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector deviceSolutionVx (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(deviceSolutionVx, hostSolutionVx, meshVectorSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector hostSolutionVx from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Vy
 	double *hostSolutionVy = (double *)malloc(meshVectorSize);
    if (hostSolutionVy == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
 	for(unsigned int i = 0; i < dof_handlerV.n_dofs(); ++i) hostSolutionVy[i] = locally_relevant_solutionVy[i];
 	
	double *deviceSolutionVy = NULL;
    err = cudaMalloc((void **)&deviceSolutionVy, meshVectorSize);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector deviceSolutionVy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(deviceSolutionVy, hostSolutionVy, meshVectorSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector hostSolutionVy from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 	
	for (int np_m = 0; np_m < PARTICLES_MOVEMENT_STEPS; ++np_m) {
		timer->enter_subsection("Variables preparation");
		
		dim3 grid_dim, block_dim;
		
		constexpr unsigned int particles_per_block = CUDAWrappers::warp_size;
		const double apply_n_blocks = std::ceil(static_cast<double>(particle_handler.n_global_particles()) / static_cast<double>(particles_per_block));
		const unsigned int apply_x_n_blocks = std::round(std::sqrt(apply_n_blocks));
		const unsigned int apply_y_n_blocks = std::ceil(apply_n_blocks / static_cast<double>(apply_x_n_blocks));

		grid_dim = dim3(apply_x_n_blocks, apply_y_n_blocks);
		block_dim = dim3(particles_per_block);
		
		pfem2Particle *deviceParticles = NULL;
		err = cudaMalloc((void **)&deviceParticles, particle_handler.n_global_particles() * sizeof(pfem2Particle));  
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector deviceParticles (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		err = cudaMemcpy(deviceParticles, particle_handler.particles.data(), particle_handler.n_global_particles() * sizeof(pfem2Particle), cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to copy vector hostParticles from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		timer->leave_subsection();

        timer->enter_subsection("CUDA step");
		move_particles_cuda<<<grid_dim, block_dim>>>(particle_handler.n_global_particles(), min_time_step, deviceParticles, deviceSolutionVx, deviceSolutionVy);
		AssertCudaKernel ();
	  
		err = cudaGetLastError();

		if (err != cudaSuccess){
			fprintf(stderr, "Failed to launch move_particles_cuda kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		timer->leave_subsection();

		timer->enter_subsection("Memory copy");
	    err = cudaMemcpy(particle_handler.particles.data(), deviceParticles, particle_handler.n_global_particles() * sizeof(pfem2Particle), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
			fprintf(stderr, "Failed to copy vector deviceParticles from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		timer->leave_subsection();
	   
		timer->enter_subsection("Particles' sorting");
		particle_handler.sort_particles_into_subdomains_and_cells(dof_handlerV);
		timer->leave_subsection();
		
		timer->enter_subsection("Memory release");
		cudaFree(deviceParticles);
		timer->leave_subsection();
	}//np_m

	//проверка наличия пустых ячеек (без частиц) и размещение в них частиц
	timer->enter_subsection("Checking cells for empty parts");
	check_cells_for_empty_parts();
	timer->leave_subsection();

	//std::cout << "Finished moving particles" << std::endl;

	cudaFree(deviceSolutionVx);
	cudaFree(deviceSolutionVy);
    free(hostSolutionVx);
    free(hostSolutionVy);
}

__global__ void distribute_particle_velocities_cuda ( const unsigned int Numparticles, const pfem2Particle *particles, double *solutionVx, double *solutionVy, double *nodeWeights)
 {
	int i = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	
	if (i < Numparticles){ 
		double shapeValue;
		
		double *velocity = (double*)((char*)&particles[0] + i * particleSize + velocityPos);
		double *refLocation = (double*)((char*)&particles[0] + i * particleSize + refLocationPos);
		int *cellDofs = (int*)((char*)&particles[0] + i * particleSize + cellDoFsPos);
				
		shapeValue = (1 - refLocation[0]) * (1 - refLocation[1]);
		atomicAdd(&solutionVx[cellDofs[0]], shapeValue * velocity[0]);
		atomicAdd(&solutionVy[cellDofs[0]], shapeValue * velocity[1]);
		atomicAdd(&nodeWeights[cellDofs[0]], shapeValue);
		
		shapeValue = refLocation[0] * (1 - refLocation[1]);
		atomicAdd(&solutionVx[cellDofs[1]], shapeValue * velocity[0]);
		atomicAdd(&solutionVy[cellDofs[1]], shapeValue * velocity[1]);
		atomicAdd(&nodeWeights[cellDofs[1]], shapeValue);

		shapeValue = (1 - refLocation[0]) * refLocation[1];
		atomicAdd(&solutionVx[cellDofs[2]], shapeValue * velocity[0]);
		atomicAdd(&solutionVy[cellDofs[2]], shapeValue * velocity[1]);
		atomicAdd(&nodeWeights[cellDofs[2]], shapeValue);
		
		shapeValue = refLocation[0] * refLocation[1];
		atomicAdd(&solutionVx[cellDofs[3]], shapeValue * velocity[0]);
		atomicAdd(&solutionVy[cellDofs[3]], shapeValue * velocity[1]);
		atomicAdd(&nodeWeights[cellDofs[3]], shapeValue);
	}	
}

__global__ void calculate_node_velocities_cuda ( const unsigned int dof_count, double *solutionVx, double *solutionVy, const double *nodeWeights)
{
	int i = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	
	if(i < dof_count){
		solutionVx[i] /= nodeWeights[i];
		solutionVy[i] /= nodeWeights[i];
			
		//printf("Velocity for i=%d equals (%f, %f) with weight=%f\n", i, solutionVx[i], solutionVy[i], nodeWeights[i]);
	}
}

void pfem2Solver::distribute_particle_velocities_to_grid() //перенос скоростей частиц на узлы сетки
{	
	TimerOutput::Scope timer_section(*timer, "Distribution of particles' velocities to grid nodes");
		
	cudaError_t err = cudaSuccess;
	
	dim3 grid_dim, block_dim;
		
	constexpr unsigned int particles_per_block = CUDAWrappers::warp_size;
	const double apply_n_blocks = std::ceil(static_cast<double>(particle_handler.n_global_particles()) / static_cast<double>(particles_per_block));
	const unsigned int apply_x_n_blocks = std::round(std::sqrt(apply_n_blocks));
	const unsigned int apply_y_n_blocks = std::ceil(apply_n_blocks / static_cast<double>(apply_x_n_blocks));

	grid_dim = dim3(apply_x_n_blocks, apply_y_n_blocks);
	block_dim = dim3(particles_per_block);
	
	pfem2Particle *deviceParticles = NULL;
	err = cudaMalloc((void **)&deviceParticles, particle_handler.n_global_particles() * sizeof(pfem2Particle));  
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector deviceParticles (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(deviceParticles, particle_handler.particles.data(), particle_handler.n_global_particles() * sizeof(pfem2Particle), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector hostParticles from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	size_t meshVectorSize = dof_handlerV.n_dofs() * sizeof(double);
	
	double *deviceSolutionVx = NULL;
	err = cudaMalloc((void **)&deviceSolutionVx, meshVectorSize);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector deviceSolutionVx (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *deviceSolutionVy = NULL;
	err = cudaMalloc((void **)&deviceSolutionVy, meshVectorSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector deviceSolutionVy (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *deviceWeights = NULL;
	err = cudaMalloc((void **)&deviceWeights, meshVectorSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector deviceSolutionVy (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	distribute_particle_velocities_cuda<<<grid_dim, block_dim>>>(particle_handler.n_global_particles(), deviceParticles, deviceSolutionVx, deviceSolutionVy, deviceWeights);
	calculate_node_velocities_cuda<<<grid_dim, block_dim>>>(dof_handlerV.n_dofs(), deviceSolutionVx, deviceSolutionVy, deviceWeights);
	AssertCudaKernel ();
	  
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch move_particles_cuda kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	double *hostSolutionVx = (double *)malloc(meshVectorSize);
	if (hostSolutionVx == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(hostSolutionVx, deviceSolutionVx, meshVectorSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector deviceSolutionVx from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *hostSolutionVy = (double *)malloc(meshVectorSize);
	if (hostSolutionVy == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(hostSolutionVy, deviceSolutionVy, meshVectorSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector deviceSolutionVy from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	for(unsigned int i = 0; i < dof_handlerV.n_dofs(); ++i){
		locally_relevant_solutionVx[i] = hostSolutionVx[i];
		locally_relevant_solutionVy[i] = hostSolutionVy[i];
	}
	
	for(std::map<unsigned int, unsigned int>::iterator it = wallsAndBodyDoFs.begin(); it != wallsAndBodyDoFs.end(); ++it){
		locally_relevant_solutionVx(it->first) = 0.0;
		locally_relevant_solutionVy(it->first) = 0.0;
	}
	
	locally_relevant_solutionVx.compress (VectorOperation::insert);
	locally_relevant_solutionVy.compress (VectorOperation::insert);

	cudaFree(deviceWeights);
	cudaFree(deviceParticles);
	cudaFree(deviceSolutionVx);
	cudaFree(deviceSolutionVy);
    free(hostSolutionVx);
    free(hostSolutionVy);

	//std::cout << "Finished distributing particles' velocities to grid" << std::endl;	 
}

void pfem2Solver::calculate_loads(types::boundary_id patch_id, std::ofstream *out){
	TimerOutput::Scope timer_section(*timer, "Loads calculation");
	
	double Fx_nu(0.0), Fx_p(0.0), Fy_nu(0.0), Fy_p(0.0), point_valueP, dVtdn, Cx_nu, Cx_p, Cy_nu, Cy_p;
		
	for(const auto &cell : dof_handlerP.active_cell_iterators())
		if(cell->is_locally_owned())
			for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == patch_id) {
					feP_face_values.reinit (cell, face_number);

					for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point) {
						point_valueP = 0.0;
						dVtdn = 0.0;

						for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
							point_valueP += locally_relevant_solutionP(cell->vertex_dof_index(vertex,0)) * feP_face_values.shape_value(vertex, q_point);
							dVtdn += (locally_relevant_solutionVx(cell->vertex_dof_index(vertex,0)) * feP_face_values.normal_vector(q_point)[1] - locally_relevant_solutionVy(cell->vertex_dof_index(vertex,0)) * feP_face_values.normal_vector(q_point)[0]) *
									(feP_face_values.shape_grad(vertex, q_point)[0] * feP_face_values.normal_vector(q_point)[0] + feP_face_values.shape_grad(vertex, q_point)[1] * feP_face_values.normal_vector(q_point)[1]);
						}//vertex

						Fx_nu += mu * dVtdn * feP_face_values.normal_vector(q_point)[1] * feP_face_values.JxW (q_point);
						Fx_p -= point_valueP * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW (q_point);
						Fy_nu -= mu * dVtdn * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW (q_point);
						Fy_p -= point_valueP * feP_face_values.normal_vector(q_point)[1] * feP_face_values.JxW (q_point);
					}//q_index
				}//if

	Cx_nu = 2.0 * Fx_nu / (rho * uMean * uMean * diam);
	Cx_p = 2.0 * Fx_p / (rho * uMean * uMean * diam);
	Cy_nu = 2.0 * Fy_nu / (rho * uMean * uMean * diam);
	Cy_p = 2.0 * Fy_p / (rho * uMean * uMean * diam);

	//pressure difference
	double pressureDifference = 0.0;
	if(xaDoF != -100) pressureDifference += locally_relevant_solutionP(xaDoF);
	if(xeDoF != -100) pressureDifference -= locally_relevant_solutionP(xeDoF);

	const double local_coeffs[5] = { Cx_nu, Cx_p, Cy_nu, Cy_p, pressureDifference };
	double global_coeffs[5];
	
	Utilities::MPI::sum(local_coeffs, mpi_communicator, global_coeffs);
		
	if (this_mpi_process == 0){
		double Cx = global_coeffs[0] + global_coeffs[1];
		double Cy = global_coeffs[2] + global_coeffs[3];
		*out << time << "," << Cx << "," << Cy << "," << global_coeffs[4] << "," << global_coeffs[0] << "," << global_coeffs[1] << "," << global_coeffs[2] << "," << global_coeffs[3] << std::endl;
	}
	
	//std::cout << "Calculating loads finished" << std::endl;
}

void pfem2Solver::setCudaConstants()
{
	cudaError_t err = cudaSuccess;
	
	size_t pfem2ParticleSize = sizeof(pfem2Particle);
	err = cudaMemcpyToSymbol(particleSize, &pfem2ParticleSize,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t locationOffset = offsetof(pfem2Particle, location);
	err = cudaMemcpyToSymbol(locationPos, &locationOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t refLocationOffset = offsetof(pfem2Particle, reference_location);
	err = cudaMemcpyToSymbol(refLocationPos, &refLocationOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t velocityOffset = offsetof(pfem2Particle, velocity);
	err = cudaMemcpyToSymbol(velocityPos, &velocityOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t velocityExtOffset = offsetof(pfem2Particle, velocity_ext);
	err = cudaMemcpyToSymbol(velocityExtPos, &velocityExtOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t cellDofsOffset = offsetof(pfem2Particle, cell_dofs);
	err = cudaMemcpyToSymbol(cellDoFsPos, &cellDofsOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
}
