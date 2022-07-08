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
	: location (location),
    reference_location (reference_location),
    id (id),
	velocity({0,0})
{

}

pfem2Particle::pfem2Particle(const void*& data)
{
	const types::particle_index* id_data = static_cast<const types::particle_index*> (data);
	id = *id_data++;
	tria_position = *id_data++;
	const double* pdata = reinterpret_cast<const double*> (id_data);

	for (unsigned int i = 0; i < 2; ++i) location(i) = *pdata++;

	for (unsigned int i = 0; i < 2; ++i) reference_location(i) = *pdata++;

	for (unsigned int i = 0; i < 2; ++i) velocity[i] = *pdata++;
	for (unsigned int i = 0; i < 2; ++i) velocity_ext[i] = *pdata++;

	data = static_cast<const void*> (pdata);
}


void pfem2Particle::set_location (const Point<2> &new_location)
{
	location = new_location;
}

const Point<2> & pfem2Particle::get_location () const
{
    return location;
}

void pfem2Particle::set_reference_location (const Point<2> &new_reference_location)
{
    reference_location = new_reference_location;
}

const Point<2> & pfem2Particle::get_reference_location () const
{
    return reference_location;
}

unsigned int pfem2Particle::get_id () const
{
    return id;
}

void pfem2Particle::set_tria_position(const int &new_position)
{
	tria_position = new_position;
}

const Tensor<1,2> & pfem2Particle::get_velocity() const
{
	return velocity;
}

const Tensor<1,2> & pfem2Particle::get_velocity_ext() const
{
	return velocity_ext;
}

double pfem2Particle::get_velocity_component(int component) const
{
	return velocity[component];
}

void pfem2Particle::set_velocity (const Tensor<1,2> &new_velocity)
{
	velocity = new_velocity;
}

void pfem2Particle::set_velocity_component (const double value, int component)
{
	velocity[component] = value;
}

void pfem2Particle::set_velocity_ext (const Tensor<1,2> &new_ext_velocity)
{
	velocity_ext = new_ext_velocity;
}

void pfem2Particle::set_map_position (const std::unordered_multimap<int, pfem2Particle*>::iterator &new_position)
{
	map_position = new_position;
}

const std::unordered_multimap<int, pfem2Particle*>::iterator & pfem2Particle::get_map_position () const
{
	return map_position;
}

Triangulation<2>::cell_iterator pfem2Particle::get_surrounding_cell(const Triangulation<2> &triangulation) const
{
	const typename Triangulation<2>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position);
	
	return cell;
}

unsigned int pfem2Particle::find_closest_vertex_of_cell(const typename Triangulation<2>::active_cell_iterator &cell, const Mapping<2> &mapping)
{
	//transformation of local particle coordinates transformation is required as the global particle coordinates have already been updated by the time this function is called
	const Point<2> old_position = mapping.transform_unit_to_real_cell(cell, reference_location);
	
	Tensor<1,2> velocity_normalized = velocity_ext / velocity_ext.norm();
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

void pfem2Particle::write_data(void*& data) const
{
	unsigned int* id_data = static_cast<unsigned int*> (data);
	*id_data = id;
	++id_data;

	*id_data = tria_position;
	++id_data;

	double* pdata = reinterpret_cast<double*> (id_data);

	// Write location data
	for (unsigned int i = 0; i < 2; ++i, ++pdata) *pdata = location(i);

	// Write reference location data
	for (unsigned int i = 0; i < 2; ++i, ++pdata) *pdata = reference_location(i);

	// Write velocity
	for (unsigned int i = 0; i < 2; ++i, ++pdata) *pdata = velocity[i];

	// Write streamline velocity
	for (unsigned int i = 0; i < 2; ++i, ++pdata) *pdata = velocity_ext[i];

	data = static_cast<void*> (pdata);
}


pfem2ParticleHandler::pfem2ParticleHandler(const parallel::distributed::Triangulation<2> &tria, const Mapping<2> &coordMapping)
	: triangulation(&tria, typeid(*this).name())
	, mapping(&coordMapping, typeid(*this).name())
	, particles()
	, global_number_of_particles(0)
    , global_max_particles_per_cell(0)
    {}
    
pfem2ParticleHandler::~pfem2ParticleHandler()
{
	clear_particles();
}


void pfem2ParticleHandler::initialize_maps()
{
	vertex_to_cells = std::vector<std::set<typename Triangulation<2>::active_cell_iterator>>(GridTools::vertex_to_cell_map(*triangulation));
	vertex_to_cell_centers = std::vector<std::vector<Tensor<1, 2>>>(GridTools::vertex_to_cell_centers_directions(*triangulation, vertex_to_cells));
}


void pfem2ParticleHandler::clear()
{
	clear_particles();
	global_number_of_particles    = 0;
    global_max_particles_per_cell = 0;
}

void pfem2ParticleHandler::clear_particles()
{
	//for(auto particleIndex = particles.begin(); particleIndex != particles.end(); ++particleIndex) delete (*particleIndex).second;
	particles.clear();
}

void pfem2ParticleHandler::remove_particle(const pfem2Particle *particle)
{
	particles.erase(particle->get_map_position());
	delete particle;
}

void pfem2ParticleHandler::insert_particle(pfem2Particle *particle,
										   const typename Triangulation<2>::active_cell_iterator &cell)
{
	typename std::unordered_multimap<int, pfem2Particle*>::iterator it = particles.insert(std::make_pair(cell->index(), particle));
	particle->set_map_position(it);
	particle->set_tria_position(cell->index());
}

unsigned int pfem2ParticleHandler::n_global_particles() const
{
	return particles.size();
}

unsigned int pfem2ParticleHandler::n_global_max_particles_per_cell() const
{
	return global_max_particles_per_cell;
}

unsigned int pfem2ParticleHandler::n_locally_owned_particles() const
{
	return particles.size();
}

unsigned int pfem2ParticleHandler::n_particles_in_cell(const typename Triangulation<2>::active_cell_iterator &cell) const
{
	return particles.count(cell->index());
}

bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,2> &particle_direction, const std::vector<Tensor<1,2> > &center_directions)
{
	const double scalar_product_a = center_directions[a] * particle_direction;
    const double scalar_product_b = center_directions[b] * particle_direction;

    return scalar_product_a > scalar_product_b;
}

void pfem2ParticleHandler::sort_particles_into_subdomains_and_cells()
{
#ifdef VERBOSE_OUTPUT
	std::cout << "Started sorting particles..." << std::endl;
	double start = omp_get_wtime();
#endif // VERBOSE_OUTPUT
	
	std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator> particles_out_of_cell;
	particles_out_of_cell.reserve(particles.size());
	
	for(auto it = begin(); it != end(); ++it){
		const typename Triangulation<2>::cell_iterator cell = (*it).second->get_surrounding_cell(*triangulation);
		
		try{
			const Point<2> p_unit = mapping->transform_real_to_unit_cell(cell, (*it).second->get_location());
		
			if(GeometryInfo<2>::is_inside_unit_cell(p_unit)) (*it).second->set_reference_location(p_unit);
			else particles_out_of_cell.push_back(it);
		} catch(typename Mapping<2>::ExcTransformationFailed &){
#ifdef VERBOSE_OUTPUT
			std::cout << "Transformation failed for particle with global coordinates " << (*it).second->get_location() << " (checked cell index #" << cell->index() << ")" << std::endl;
#endif // VERBOSE_OUTPUT
			
			particles_out_of_cell.push_back(it);
		}
	}

#ifdef VERBOSE_OUTPUT	
	double checkingPositionsEnd = omp_get_wtime();
	std::cout << "Finished sorting out gone particles" << std::endl;
#endif // VERBOSE_OUTPUT
	
	std::vector<std::pair<int, pfem2Particle*>> sorted_particles;

	std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator> particles_to_be_deleted;
	std::map<unsigned int, std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator>> moved_particles;
	std::map<unsigned int, std::vector<Triangulation<2>::active_cell_iterator>> moved_cells;

	//typedef typename std::vector<std::pair<int, pfem2Particle*>>::size_type vector_size;
	typedef typename std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator>::size_type vector_size;

	sorted_particles.reserve(static_cast<vector_size> (particles_out_of_cell.size() * 1.25));
	particles_to_be_deleted.reserve(static_cast<vector_size> (particles_out_of_cell.size() * 0.25));

	const std::set<unsigned int> ghost_owners = triangulation->ghost_owners();

	for (auto ghost_domain_id = ghost_owners.begin(); ghost_domain_id != ghost_owners.end(); ++ghost_domain_id) {
		moved_particles[*ghost_domain_id].reserve(static_cast<vector_size> (particles_out_of_cell.size() * 0.25));
		moved_cells[*ghost_domain_id].reserve(static_cast<vector_size> (particles_out_of_cell.size() * 0.25));
	}

#ifdef VERBOSE_OUTPUT
	double prepareToSortClock;
	double closestVertexClocks = 0;
	double neighboursSortingClocks = 0;
	double neighboursCheckClocks = 0;
	double globalCellSearchClocks = 0;
	double particleFinalizationClocks = 0;
	
	double catchClocks = 0;
#endif // VERBOSE_OUTPUT
			
	{
      //const std::vector<std::set<typename Triangulation<2>::active_cell_iterator>> vertex_to_cells(GridTools::vertex_to_cell_map(*triangulation));
      //const std::vector<std::vector<Tensor<1,2>>> vertex_to_cell_centers(GridTools::vertex_to_cell_centers_directions(*triangulation,vertex_to_cells));
	  std::vector<unsigned int> neighbor_permutation;

#ifdef VERBOSE_OUTPUT
	  std::cout << "Finished building vertex to cell and cell centers map" << std::endl;
	  prepareToSortClock = omp_get_wtime();
	  int numOutOfMesh = 0;
#endif // VERBOSE_OUTPUT

      for (auto it = particles_out_of_cell.begin(); it != particles_out_of_cell.end(); ++it){
#ifdef VERBOSE_OUTPUT
      	  double particleStart = omp_get_wtime();
#endif // VERBOSE_OUTPUT
		  
		  Point<2> current_reference_position;
          bool found_cell = false;

		  auto particle = (*it);

          typename Triangulation<2>::active_cell_iterator current_cell = (*particle).second->get_surrounding_cell(*triangulation);

          const unsigned int closest_vertex = (*particle).second->find_closest_vertex_of_cell(current_cell, *mapping);
          Tensor<1,2> vertex_to_particle = (*particle).second->get_location() - current_cell->vertex(closest_vertex);
          vertex_to_particle /= vertex_to_particle.norm();

#ifdef VERBOSE_OUTPUT
		  double closestVertexEnd = omp_get_wtime();
		  closestVertexClocks += closestVertexEnd - particleStart;
#endif // VERBOSE_OUTPUT

          const unsigned int closest_vertex_index = current_cell->vertex_index(closest_vertex);
          const unsigned int n_neighbor_cells = vertex_to_cells[closest_vertex_index].size();

		  neighbor_permutation.resize(n_neighbor_cells);
		  
          for (unsigned int i=0; i<n_neighbor_cells; ++i) neighbor_permutation[i] = i;

          std::sort(neighbor_permutation.begin(), neighbor_permutation.end(),
			std::bind(&compare_particle_association, std::placeholders::_1, std::placeholders::_2, std::cref(vertex_to_particle), std::cref(vertex_to_cell_centers[closest_vertex_index])));

#ifdef VERBOSE_OUTPUT
		  double neighboursSortingEnd = omp_get_wtime();
		  neighboursSortingClocks += neighboursSortingEnd - closestVertexEnd;
#endif // VERBOSE_OUTPUT
			
		  for (unsigned int i=0; i<n_neighbor_cells; ++i){
		      typename std::set<typename Triangulation<2>::active_cell_iterator>::const_iterator cell = vertex_to_cells[closest_vertex_index].begin();

              std::advance(cell,neighbor_permutation[i]);
              
              try{
				  const Point<2> p_unit = mapping->transform_real_to_unit_cell(*cell, (*particle).second->get_location());
				  if (GeometryInfo<2>::is_inside_unit_cell(p_unit)){
					current_cell = *cell;
					current_reference_position = p_unit;
					(*particle).second->set_tria_position(current_cell->index());
					found_cell = true;
					
#ifdef VERBOSE_OUTPUT
					//std::cout << "Particle found in a neighbour cell" << std::endl;
#endif // VERBOSE_OUTPUT
					
					break; 
				  }
              } catch(typename Mapping<2>::ExcTransformationFailed &)
                { }
            }

#ifdef VERBOSE_OUTPUT
          double neighboursCheckEnd = omp_get_wtime();
          neighboursCheckClocks += neighboursCheckEnd - neighboursSortingEnd;
          
          double t1,t2;
#endif // VERBOSE_OUTPUT
          
          if (!found_cell){			  
#ifdef VERBOSE_OUTPUT
              ++numOutOfMesh;
#endif // VERBOSE_OUTPUT
			  particles_to_be_deleted.push_back(particle);
			  continue;

                      
/*                      
			  t1 = omp_get_wtime();          
			  try {
                  const std::pair<const typename Triangulation<2>::active_cell_iterator, Point<2> > current_cell_and_position =
                          GridTools::find_active_cell_around_point<2> (*mapping, *triangulation, (*particle).second->get_location());
              
                  current_cell = current_cell_and_position.first;
                  current_reference_position = current_cell_and_position.second;
                  //std::cout << "Particle found in a far away cell" << std::endl;
              } catch (GridTools::ExcPointNotFound<2> &){				  
				  particles_to_be_deleted.push_back(particle);
				  //std::cout << "Particle not found" << std::endl;
				  ++numOutOfMesh;
				  t2 = omp_get_wtime();
				  
				  catchClocks += t2-t1;
				  continue;
			  }
*/
          }
                    
#ifdef VERBOSE_OUTPUT
          double globalCellSearchEnd = omp_get_wtime();
          globalCellSearchClocks += globalCellSearchEnd - neighboursCheckEnd;
#endif // VERBOSE_OUTPUT
          
          (*particle).second->set_reference_location(current_reference_position);
		  if (current_cell->is_locally_owned()) sorted_particles.push_back(std::make_pair(current_cell->index(), (*particle).second));
		  else {
			  moved_particles[current_cell->subdomain_id()].push_back(particle);
			  moved_cells[current_cell->subdomain_id()].push_back(current_cell);
			  particles_to_be_deleted.push_back(particle);
		  }
                   
#ifdef VERBOSE_OUTPUT
          double particleFinalizationEnd = omp_get_wtime();
          particleFinalizationClocks += particleFinalizationEnd - globalCellSearchEnd;
#endif // VERBOSE_OUTPUT
	  }                                                        
	  
#ifdef VERBOSE_OUTPUT	  
	  std::cout << "N out of mesh = " << numOutOfMesh << std::endl;
#endif // VERBOSE_OUTPUT
	}
	
#ifdef VERBOSE_OUTPUT	
	std::cout << "Finished processing gone particles" << std::endl;
	
	double presortingStart = omp_get_wtime();
#endif // VERBOSE_OUTPUT
	
	std::unordered_multimap<int,pfem2Particle*> sorted_particles_map;

#ifdef DEAL_II_WITH_MPI
	if (dealii::Utilities::MPI::n_mpi_processes(triangulation->get_communicator()) > 1)
		send_recv_particles(moved_particles, sorted_particles_map, moved_cells);
#endif

	sorted_particles_map.insert(sorted_particles.begin(), sorted_particles.end());
	
#ifdef VERBOSE_OUTPUT
	double presortingEnd = omp_get_wtime();
#endif // VERBOSE_OUTPUT
		
	for (unsigned int i = 0; i < particles_to_be_deleted.size(); ++i) {
		auto particle = particles_to_be_deleted[i];

		delete (*particle).second;
	}

	for (unsigned int i = 0; i < particles_out_of_cell.size(); ++i) {
		auto particle = particles_out_of_cell[i];
		particles.erase(particle);
	}
	
#ifdef VERBOSE_OUTPUT
	double particlesDeleteClock = omp_get_wtime();
#endif // VERBOSE_OUTPUT
	
	//for (unsigned int i=0; i<moved_particles.size(); ++i) particles.erase(moved_particles[i]);

#ifdef VERBOSE_OUTPUT
	double movedParticlesEraseClock = omp_get_wtime();
#endif // VERBOSE_OUTPUT
	
	particles.insert(sorted_particles_map.begin(),sorted_particles_map.end());
	
#ifdef VERBOSE_OUTPUT
	double end = omp_get_wtime();
	
	std::cout << "Checking positions time: " << (checkingPositionsEnd-start) << " sec. (" << (checkingPositionsEnd-start)/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Preparation for sorting time: " << (prepareToSortClock-checkingPositionsEnd) << " sec. (" << (prepareToSortClock-checkingPositionsEnd)/(end-start)*100 << "% of total)" << std::endl;
	
	std::cout << "Getting the closest vertex time: " << closestVertexClocks << " sec. (" << closestVertexClocks/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Neighbours' sorting time: " << neighboursSortingClocks << " sec. (" << neighboursSortingClocks/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Neighbours' checking time: " << neighboursCheckClocks << " sec. (" << neighboursCheckClocks/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Global cell search time: " << globalCellSearchClocks << " sec. (" << globalCellSearchClocks/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Exception catch time: " << catchClocks << " sec. (" << catchClocks/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Finalization for particle time: " << particleFinalizationClocks << " sec. (" << particleFinalizationClocks/(end-start)*100 << "% of total)" << std::endl;

	std::cout << "Pre-sorting time: " << (presortingEnd-presortingStart) << " sec. (" << (presortingEnd-presortingStart)/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Particles delete time: " << (particlesDeleteClock-presortingEnd) << " sec. (" << (particlesDeleteClock-presortingEnd)/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Moved particles erase time: " << (movedParticlesEraseClock-particlesDeleteClock) << " sec. (" << (movedParticlesEraseClock-particlesDeleteClock)/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Pre-sorted particles insert time: " << (end-movedParticlesEraseClock) << " sec. (" << (end-movedParticlesEraseClock)/(end-start)*100 << "% of total)" << std::endl;
	std::cout << "Total sorting time: " << (end - start) << " sec." << std::endl;
	
	std::cout << "Finished sorting particles" << std::endl;
#endif // VERBOSE_OUTPUT
}


#ifdef DEAL_II_WITH_MPI
void pfem2ParticleHandler::send_recv_particles(const std::map<unsigned int, std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator>>& particles_to_send,
	std::unordered_multimap<int, pfem2Particle*>& received_particles,
	const std::map<unsigned int, std::vector<typename Triangulation<2>::active_cell_iterator>>& send_cells)
{
	// Determine the communication pattern
	const std::set<unsigned int> ghost_owners = triangulation->ghost_owners();
	const std::vector<unsigned int> neighbors(ghost_owners.begin(), ghost_owners.end());
	const unsigned int n_neighbors = neighbors.size();

	if (send_cells.size() != 0)
		Assert(particles_to_send.size() == send_cells.size(), ExcInternalError());

	// If we do not know the subdomain this particle needs to be send to, throw an error
	Assert(particles_to_send.find(numbers::artificial_subdomain_id) == particles_to_send.end(), ExcInternalError());

	// TODO: Implement the shipping of particles to processes that are not ghost owners of the local domain
	for (auto send_particles = particles_to_send.begin(); send_particles != particles_to_send.end(); ++send_particles)
		Assert(ghost_owners.find(send_particles->first) != ghost_owners.end(), ExcNotImplemented());

	unsigned int n_send_particles = 0;
	for (auto send_particles = particles_to_send.begin(); send_particles != particles_to_send.end(); ++send_particles)
		n_send_particles += send_particles->second.size();

	const unsigned int cellid_size = sizeof(CellId::binary_type);

	// Containers for the amount and offsets of data we will send to other processors and the data itself.
	std::vector<unsigned int> n_send_data(n_neighbors, 0);
	std::vector<unsigned int> send_offsets(n_neighbors, 0);
	std::vector<char> send_data;

	// Only serialize things if there are particles to be send.
	// We can not return early even if no particles are send, because we might receive particles from other processes
	if (n_send_particles) {
		// Allocate space for sending particle data
		auto firstParticle = begin();
		const unsigned int particle_size = firstParticle->second->serialized_size_in_bytes() + cellid_size;
		send_data.resize(n_send_particles * particle_size);
		void* data = static_cast<void*> (&send_data.front());

		// Serialize the data sorted by receiving process
		for (unsigned int i = 0; i < n_neighbors; ++i) {
			send_offsets[i] = reinterpret_cast<std::size_t> (data) - reinterpret_cast<std::size_t> (&send_data.front());

			for (unsigned int j = 0; j < particles_to_send.at(neighbors[i]).size(); ++j) {
				auto particleIndex = particles_to_send.at(neighbors[i])[j];

				// If no target cells are given, use the iterator information
				typename Triangulation<2>::active_cell_iterator cell;
				if (!send_cells.size()) cell = particleIndex->second->get_surrounding_cell(*triangulation);
				else cell = send_cells.at(neighbors[i])[j];

				const CellId::binary_type cellid = cell->id().template to_binary<2>();
				memcpy(data, &cellid, cellid_size);
				data = static_cast<char*>(data) + cellid_size;

				particleIndex->second->write_data(data);
			}
			n_send_data[i] = reinterpret_cast<std::size_t> (data) - send_offsets[i] - reinterpret_cast<std::size_t> (&send_data.front());
		}
	}

	// Containers for the data we will receive from other processors
	std::vector<unsigned int> n_recv_data(n_neighbors);
	std::vector<unsigned int> recv_offsets(n_neighbors);

	// Notify other processors how many particles we will send
	{
		std::vector<MPI_Request> n_requests(2 * n_neighbors);
		for (unsigned int i = 0; i < n_neighbors; ++i) {
			const int ierr = MPI_Irecv(&(n_recv_data[i]), 1, MPI_INT, neighbors[i], 0, triangulation->get_communicator(), &(n_requests[2 * i]));
			AssertThrowMPI(ierr);
		}

		for (unsigned int i = 0; i < n_neighbors; ++i) {
			const int ierr = MPI_Isend(&(n_send_data[i]), 1, MPI_INT, neighbors[i], 0, triangulation->get_communicator(), &(n_requests[2 * i + 1]));
			AssertThrowMPI(ierr);
		}

		const int ierr = MPI_Waitall(2 * n_neighbors, &n_requests[0], MPI_STATUSES_IGNORE);
		AssertThrowMPI(ierr);
	}

	// Determine how many particles and data we will receive
	unsigned int total_recv_data = 0;
	for (unsigned int neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id) {
		recv_offsets[neighbor_id] = total_recv_data;
		total_recv_data += n_recv_data[neighbor_id];
	}

	// Set up the space for the received particle data
	std::vector<char> recv_data(total_recv_data);

	// Exchange the particle data between domains
	{
		std::vector<MPI_Request> requests(2 * n_neighbors);
		unsigned int send_ops = 0;
		unsigned int recv_ops = 0;

		for (unsigned int i = 0; i < n_neighbors; ++i)
			if (n_recv_data[i] > 0) {
				const int ierr = MPI_Irecv(&(recv_data[recv_offsets[i]]), n_recv_data[i], MPI_CHAR, neighbors[i], 1, triangulation->get_communicator(), &(requests[send_ops]));
				AssertThrowMPI(ierr);
				send_ops++;
			}

		for (unsigned int i = 0; i < n_neighbors; ++i)
			if (n_send_data[i] > 0) {
				const int ierr = MPI_Isend(&(send_data[send_offsets[i]]), n_send_data[i], MPI_CHAR, neighbors[i], 1, triangulation->get_communicator(), &(requests[send_ops + recv_ops]));
				AssertThrowMPI(ierr);
				recv_ops++;
			}

		const int ierr = MPI_Waitall(send_ops + recv_ops, &requests[0], MPI_STATUSES_IGNORE);
		AssertThrowMPI(ierr);
	}

	// Put the received particles into the domain if they are in the triangulation
	const void* recv_data_it = static_cast<const void*> (recv_data.data());

	while (reinterpret_cast<std::size_t> (recv_data_it) - reinterpret_cast<std::size_t> (recv_data.data()) < total_recv_data) {
		CellId::binary_type binary_cellid;
		memcpy(&binary_cellid, recv_data_it, cellid_size);
		const CellId id(binary_cellid);
		recv_data_it = static_cast<const char*> (recv_data_it) + cellid_size;

		const typename Triangulation<2>::active_cell_iterator cell = triangulation->create_cell_iterator(id);

		pfem2Particle* newParticle = new pfem2Particle(recv_data_it);
		typename std::unordered_multimap<int, pfem2Particle*>::iterator recv_particle = received_particles.insert(std::make_pair(cell->index(), newParticle));
		newParticle->set_map_position(recv_particle);
	}

	AssertThrow(recv_data_it == recv_data.data() + recv_data.size(),
		ExcMessage("The amount of data that was read into new particles does not match the amount of data sent around."));
}
#endif


std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::begin()
{
	return particles.begin();
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::end()
{
	return particles.end();
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::particles_in_cell_begin(const typename Triangulation<2>::active_cell_iterator &cell)
{
	return particles.equal_range(cell->index()).first;
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::particles_in_cell_end(const typename Triangulation<2>::active_cell_iterator &cell)
{
	return particles.equal_range(cell->index()).second;
}

pfem2Solver::pfem2Solver()
	: mpi_communicator(MPI_COMM_WORLD),
	tria(mpi_communicator, Triangulation<2>::maximum_smoothing),
	particle_handler(tria, mapping),
	feV (1),
	feP (1),
	feU (1),
	fe(FE_Q<2>(1), 1),
	fe2d(FE_Q<2>(1), 2),
	dof_handlerV (tria),
	dof_handlerP (tria),
	dof_handlerU (tria),
	quadrature_formula(2),
	face_quadrature_formula(2),
	feV_values(feV, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feP_values(feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feU_values(fe2d, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feV_face_values(feV, face_quadrature_formula, update_values | update_quadrature_points | update_gradients | update_normal_vectors | update_JxW_values),
	feP_face_values(feP, face_quadrature_formula, update_values | update_quadrature_points | update_gradients | update_normal_vectors | update_JxW_values),
	dofs_per_cellV(feV.dofs_per_cell),
	dofs_per_cellP(feP.dofs_per_cell),
	dofs_per_cellU(fe2d.dofs_per_cell),
	local_dof_indicesV(dofs_per_cellV),
	local_dof_indicesP(dofs_per_cellP),
	local_dof_indicesU(dofs_per_cellU),
	n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
	this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
	pcout(std::cout, (this_mpi_process == 0)),
	n_q_points(quadrature_formula.size()),
	n_face_q_points(face_quadrature_formula.size()),
	quantities({0,0})
{

}

pfem2Solver::~pfem2Solver()
{
	dof_handlerU.clear(); //?
}

void pfem2Solver::seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell)
{
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	double shapeValue;
	
	for(unsigned int i = 0; i < quantities[0]; ++i){
		for(unsigned int j = 0; j < quantities[1]; ++j){
			pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
			particle_handler.insert_particle(particle, cell);
			
			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				shapeValue = fe.shape_value(vertex, particle->get_reference_location());

				particle->set_velocity_component(particle->get_velocity_component(0) + shapeValue * locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)), 0);
				particle->set_velocity_component(particle->get_velocity_component(1) + shapeValue * locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)), 1);
			}//vertex
		}
	}
}

void pfem2Solver::check_empty_cells()
{
	for (auto cell = dof_handlerV.begin(tria.n_levels()-1); cell != dof_handlerV.end(tria.n_levels()-1); ++cell)
		if(cell->is_locally_owned()) check_cell_for_empty_parts(cell);
}

bool pfem2Solver::check_cell_for_empty_parts (const typename DoFHandler<2>::cell_iterator &cell)
{
	bool res = false;
	
	std::map<std::vector<unsigned int>, unsigned int> particlesInParts;
	std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator> particles_to_be_deleted;
	
	//определение, в каких частях ячейки лежат частицы
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	unsigned int num_x, num_y;
	for(auto particleIndex = particle_handler.particles_in_cell_begin(cell); particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex){
		num_x = (*particleIndex).second->get_reference_location()(0)/hx;
		num_y = (*particleIndex).second->get_reference_location()(1)/hy;

		particlesInParts[{num_x,num_y}]++;
		if(particlesInParts[{num_x,num_y}] > MAX_PARTICLES_PER_CELL_PART) particles_to_be_deleted.push_back(particleIndex);
	}
	
	double shapeValue;
	
	//проверка каждой части ячейки на количество частиц: при 0 - подсевание 1 частицы в центр
	for(unsigned int i = 0; i < quantities[0]; i++){
		for(unsigned int j = 0; j < quantities[1]; j++){			
			if(!particlesInParts[{i,j}]){
				pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
				particle_handler.insert_particle(particle, cell);
				
				for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
					shapeValue = fe.shape_value(vertex, particle->get_reference_location());

					particle->set_velocity_component(particle->get_velocity_component(0) + shapeValue * locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)), 0);
					particle->set_velocity_component(particle->get_velocity_component(1) + shapeValue * locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)), 1);
				}//vertex
				
				res = true;
			}
		}
	}
	
	//удаление лишних частиц
	for(unsigned int i = 0; i < particles_to_be_deleted.size(); ++i){
		auto it = particles_to_be_deleted.at(i);
		(*it).second->set_map_position(it);
		particle_handler.remove_particle((*it).second);
	}
		
	if(!particles_to_be_deleted.empty()) res = true;
	
	return res;
}

void pfem2Solver::seed_particles(const std::vector < unsigned int > & quantities)
{
	TimerOutput::Scope timer_section(*timer, "Particles' seeding");
	
	if (quantities.size() < 2) { return; }
	
	this->quantities = quantities;
	
	typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels() - 1), endc = dof_handlerV.end(tria.n_levels() - 1);
	for (; cell != endc; ++cell)
		if (cell->is_locally_owned()) seed_particles_into_cell(cell);
	
	std::cout << "Created and placed " << particleCount << " particles" << std::endl;
	std::cout << "Particle handler contains " << particle_handler.n_global_particles() << " particles" << std::endl;
}

void pfem2Solver::correct_particles_velocities()
{
	TimerOutput::Scope timer_section(*timer, "Particles' velocities correction");
	
	double shapeValue;
			
	typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels() - 1), endc = dof_handlerV.end(tria.n_levels() - 1);
	for (; cell != endc; ++cell)
		if (cell->is_locally_owned())
			for (auto particleIndex = particle_handler.particles_in_cell_begin(cell);
				particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex)
				for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex) {
					shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());

					(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(0) + shapeValue * (locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)) - locally_relevant_old_solutionVx(cell->vertex_dof_index(vertex, 0))), 0);
					(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(1) + shapeValue * (locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)) - locally_relevant_old_solutionVy(cell->vertex_dof_index(vertex, 0))), 1);
				}//vertex

	
	//std::cout << "Finished correcting particles' velocities" << std::endl;	
}

void pfem2Solver::move_particles() //перенос частиц
{
	TimerOutput::Scope timer_section(*timer, "Particles' movement");	
	
	Tensor<1,2> vel_in_part;
	
	double shapeValue;
	double min_time_step = time_step / PARTICLES_MOVEMENT_STEPS;
	
	for (int np_m = 0; np_m < PARTICLES_MOVEMENT_STEPS; ++np_m) {
		//РАЗДЕЛИТЬ НА VX И VY!!!!!!!!
		typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels() - 1), endc = dof_handlerV.end(tria.n_levels() - 1);

		for (; cell != endc; ++cell)
			if (cell->is_locally_owned())
				for (auto particleIndex = particle_handler.particles_in_cell_begin(cell);
					particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex) {

						vel_in_part = Tensor<1, 2>({ 0,0 });

					for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex) {
						shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());
						vel_in_part[0] += shapeValue * locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0));
						vel_in_part[1] += shapeValue * locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0));
					}//vertex

					(*particleIndex).second->set_velocity_ext(vel_in_part);

					vel_in_part[0] *= min_time_step;
					vel_in_part[1] *= min_time_step;

					(*particleIndex).second->set_location((*particleIndex).second->get_location() + vel_in_part);
				}//particle

		particle_handler.sort_particles_into_subdomains_and_cells();
	}//np_m
	
		//проверка наличия пустых ячеек (без частиц) и размещение в них частиц
	typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels() - 1), endc = dof_handlerV.end(tria.n_levels() - 1);
	for (; cell != endc; ++cell)
		if (cell->is_locally_owned()) check_cell_for_empty_parts(cell);
	
	//std::cout << "Finished moving particles" << std::endl;
}

void pfem2Solver::distribute_particle_velocities_to_grid() //перенос скоростей частиц на узлы сетки
{
	TimerOutput::Scope timer_section(*timer, "Distribution of particles' velocities to grid nodes");

	TrilinosWrappers::MPI::Vector node_velocityX, node_velocityY, node_weights;

	double shapeValue;

	node_velocityX.reinit(locally_owned_dofsV, mpi_communicator);
	node_velocityY.reinit(locally_owned_dofsV, mpi_communicator);
	node_weights.reinit(locally_owned_dofsV, mpi_communicator);

	Vector<double> local_Vx(dofs_per_cellV);
	Vector<double> local_Vy(dofs_per_cellV);
	Vector<double> local_weights(dofs_per_cellV);

	node_velocityX = 0.0;
	node_velocityY = 0.0;
	node_weights = 0.0;

	typename DoFHandler<2>::cell_iterator cell = dof_handlerV.begin(tria.n_levels() - 1), endc = dof_handlerV.end(tria.n_levels() - 1);
	for (; cell != endc; ++cell)
		if (cell->is_locally_owned()) {
			local_Vx = 0.0;
			local_Vy = 0.0;
			local_weights = 0.0;

			for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex)
				for (auto particleIndex = particle_handler.particles_in_cell_begin(cell);
					particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex) {
				shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());

				local_Vx(vertex) += shapeValue * (*particleIndex).second->get_velocity_component(0);
				local_Vy(vertex) += shapeValue * (*particleIndex).second->get_velocity_component(1);

				local_weights(vertex) += shapeValue;
			}//particle

			cell->get_dof_indices(local_dof_indicesV);
			cell->distribute_local_to_global(local_Vx, node_velocityX);
			cell->distribute_local_to_global(local_Vy, node_velocityY);
			cell->distribute_local_to_global(local_weights, node_weights);
		}

	node_velocityX.compress(VectorOperation::add);
	node_velocityY.compress(VectorOperation::add);
	node_weights.compress(VectorOperation::add);

	for (unsigned int i = node_velocityX.local_range().first; i < node_velocityX.local_range().second; ++i) {
		node_velocityX(i) /= node_weights(i);
		node_velocityY(i) /= node_weights(i);
	}

	node_velocityX.compress(VectorOperation::insert);
	node_velocityY.compress(VectorOperation::insert);

	locally_relevant_solutionVx = node_velocityX;
	locally_relevant_solutionVy = node_velocityY;

	for (std::map<unsigned int, boundaryDoF>::iterator it = wallsAndBodyDoFs.begin(); it != wallsAndBodyDoFs.end(); ++it) {
		std::set<typename Triangulation<2>::active_cell_iterator> adjacent_cells = particle_handler.vertex_to_cells[it->first];

		double vxValue = locally_relevant_solutionVx(it->second.dofScalarIndex);
		double vyValue = locally_relevant_solutionVy(it->second.dofScalarIndex);
		
		double vxValueBC = 0.0, vyValueBC = 0.0;
		if(it->second.boundaryId == 3) vyValueBC = body_velVy_;

		if ((it->second.boundaryId == 3 && std::fabs(vxValue - vxValueBC) > 1e-5) || std::fabs(vyValue - vyValueBC) > 1e-5)
			for (auto cell : adjacent_cells)
				if (cell->is_locally_owned()) {
					int vertexNo = -1;
					for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex)
						if (cell->vertex_index(vertex) == it->first) {
							vertexNo = vertex;
							break;
						}

					if (vertexNo == -1) continue;

					for (auto particleIndex = particle_handler.particles_in_cell_begin(cell);
						particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex) {
						shapeValue = fe.shape_value(vertexNo, (*particleIndex).second->get_reference_location());

						if(it->second.boundaryId == 3) (*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(0) + shapeValue * (vxValueBC - vxValue), 0);
						(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(1) + shapeValue * (vyValueBC - vyValue), 1);
					}//particle
				}

		if(it->second.boundaryId == 3) locally_relevant_solutionVx(it->second.dofScalarIndex) = vxValueBC;
		locally_relevant_solutionVy(it->second.dofScalarIndex) = vyValueBC;
	}

	locally_relevant_solutionVx.compress(VectorOperation::insert);
	locally_relevant_solutionVy.compress(VectorOperation::insert);

	//std::cout << "Finished distributing particles' velocities to grid" << std::endl;	 
}

void pfem2Solver::calculate_vorticity() {
	TimerOutput::Scope timer_section(*timer, "Vorticity calculation");
	
	TrilinosWrappers::MPI::Vector node_vorticity, node_weights;

	unsigned int iDoFindex;
	double shapeValue;
	double qPointVorticity;

	node_vorticity.reinit(locally_owned_dofsV, mpi_communicator);
	node_weights.reinit(locally_owned_dofsV, mpi_communicator);

	Vector<double> local_vorticity(dofs_per_cellV);
	Vector<double> local_weights(dofs_per_cellV);

	node_vorticity = 0.0;
	node_weights = 0.0;
	
	for (const auto& cell : dof_handlerV.active_cell_iterators())
		if (cell->is_locally_owned()){
			feV_values.reinit(cell);
			local_vorticity = 0.0;
			local_weights = 0.0;
			
			for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
				qPointVorticity = 0.0;
				for (unsigned int i = 0; i < dofs_per_cellV; ++i){
					iDoFindex = cell->vertex_dof_index(i, 0);
					const Tensor<1, 2> Ni_vel_grad = feV_values.shape_grad(i, q_index);
					qPointVorticity += Ni_vel_grad[0] * locally_relevant_solutionVy(iDoFindex) - Ni_vel_grad[1] * locally_relevant_solutionVx(iDoFindex);
				}
				
				for (unsigned int vertex = 0; vertex < dofs_per_cellV; ++vertex){
					shapeValue = feV_values.shape_value(vertex, q_index);
					local_vorticity(vertex) += shapeValue * qPointVorticity;
					local_weights(vertex) += shapeValue;
				}
			}
			
			cell->get_dof_indices(local_dof_indicesV);
			cell->distribute_local_to_global(local_vorticity, node_vorticity);
			cell->distribute_local_to_global(local_weights, node_weights);
		}
		
	node_vorticity.compress(VectorOperation::add);
	node_weights.compress(VectorOperation::add);

	for (unsigned int i = node_vorticity.local_range().first; i < node_vorticity.local_range().second; ++i) node_vorticity(i) /= node_weights(i);

	locally_relevant_vorticity = node_vorticity;
}

void pfem2Solver::calculate_loads(types::boundary_id patch_id, std::ofstream* out) {
	TimerOutput::Scope timer_section(*timer, "Loads calculation");

	double Fx_nu(0.0), Fx_p(0.0), Fy_nu(0.0), Fy_p(0.0), point_valueP, dVtdn, Cx, Cy;

	for (const auto& cell : dof_handlerP.active_cell_iterators())
		if (cell->is_locally_owned())
			for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == patch_id) {
					feP_face_values.reinit(cell, face_number);

					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
						point_valueP = 0.0;
						dVtdn = 0.0;

						for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex) {
							point_valueP += locally_relevant_solutionP(cell->vertex_dof_index(vertex, 0)) * feP_face_values.shape_value(vertex, q_point);
							dVtdn += (locally_relevant_solutionVx(cell->vertex_dof_index(vertex, 0)) * feP_face_values.normal_vector(q_point)[1] - locally_relevant_solutionVy(cell->vertex_dof_index(vertex, 0)) * feP_face_values.normal_vector(q_point)[0]) *
								(feP_face_values.shape_grad(vertex, q_point)[0] * feP_face_values.normal_vector(q_point)[0] + feP_face_values.shape_grad(vertex, q_point)[1] * feP_face_values.normal_vector(q_point)[1]);
						}//vertex

						Fx_nu += mu * dVtdn * feP_face_values.normal_vector(q_point)[1] * feP_face_values.JxW(q_point);
						Fx_p -= point_valueP * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
						Fy_nu -= mu * dVtdn * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
						Fy_p -= point_valueP * feP_face_values.normal_vector(q_point)[1] * feP_face_values.JxW(q_point);
					}//q_index
				}//if

	//pressure difference
	double pressureDifference = 0.0;
	if (xaDoF != -100) pressureDifference += locally_relevant_solutionP(xaDoF);
	if (xeDoF != -100) pressureDifference -= locally_relevant_solutionP(xeDoF);

	const double local_coeffs[5] = { Fx_nu, Fx_p, Fy_nu, Fy_p, pressureDifference };
	double global_coeffs[5];

	Utilities::MPI::sum(local_coeffs, mpi_communicator, global_coeffs);

	double Fx = global_coeffs[0]+global_coeffs[1];
	double Fy = global_coeffs[2]+global_coeffs[3];

	if (this_mpi_process == 0) {
		//Cx = 2.0 * Fx / (rho * uMean * uMean * diam);
		//Cy = 2.0 * Fy / (rho * uMean * uMean * diam);
		*out << time << "," << Fx << "," << Fy;
	}

	force_Fy = -Fy;
	//std::cout << "Calculating loads finished" << std::endl;
}


void pfem2Solver::initialize_moving_mesh_nodes()
{
	TimerOutput::Scope timer_section(*timer, "Moving mesh initialization");
	
	//1. Определение соответствия номеров узлов сетки степеням свободны скалярных полей (Vx, Vy, P) - для заполнения соответствующего поля в структуре узла подвижной сетки
	//(обход степеней свободы возможен только через ячейки и для каждого узла выполнять такой поиск сложно, даже с учетом однократного вызова этой процедуры)
	std::map<int, int> scalarFieldDoFs;
	for(auto cell = dof_handlerP.begin(); cell != dof_handlerP.end(); ++cell)
		if(cell->is_locally_owned())
			for(unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
				if(!scalarFieldDoFs.count(cell->vertex_index(i))) scalarFieldDoFs[cell->vertex_index(i)] = cell->vertex_dof_index(i,0);

	//2. Формирование контейнера из структур для узлов подвижной сетки
	for(auto cell = dof_handlerU.begin(); cell != dof_handlerU.end(); ++cell)
		if(cell->is_locally_owned())
			for(unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
				if(!movingNodes.count(cell->vertex_index(i))){
					meshNode node;
					node.coords = cell->vertex(i);
					node.cellIndex = cell->index();
					node.dofUindexX = cell->vertex_dof_index(i,0);
					node.dofUindexY = cell->vertex_dof_index(i,1);
					node.dofScalarIndex = scalarFieldDoFs[cell->vertex_index(i)];
					node.isBoundaryNode = false;
					
					movingNodes[cell->vertex_index(i)] = node;
				}
			
	//Определение, какие из узлов подвижной сетки лежат на границе
	for(auto cell = dof_handlerU.begin(); cell != dof_handlerU.end(); ++cell)
		if(cell->is_locally_owned() && cell->at_boundary())
			for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary())
					for (unsigned int i=0; i < GeometryInfo<2>::vertices_per_face; ++i) movingNodes[cell->face(face_number)->vertex_index(i)].isBoundaryNode = true;
			
	//3. Формирование контейнера указателей на ячейки сетки со связью со степенями свободы
	for(auto cell = dof_handlerP.begin(); cell != dof_handlerP.end(); ++cell)
		if(cell->is_locally_owned() || cell->is_ghost()) dofHandlerCells[cell->index()] = cell;
}

void pfem2Solver::reinterpolate_fields()
{
	TimerOutput::Scope timer_section(*timer, "Fields' reinterpolation");
	
	TrilinosWrappers::MPI::Vector newSolutionVx, newSolutionVy, newSolutionP, nodeWeights;
	
	newSolutionVx.reinit(locally_owned_dofsV, mpi_communicator);
	newSolutionVy.reinit(locally_owned_dofsV, mpi_communicator);
	newSolutionP.reinit(locally_owned_dofsP, mpi_communicator);
	nodeWeights.reinit(locally_owned_dofsV, mpi_communicator);
	
	newSolutionVx = 0.0;
	newSolutionVy = 0.0;
	newSolutionP = 0.0;
	nodeWeights = 0.0;
	
	//вектор ячеек, смежных каждому узлу сетки (количество элементов вектора = количеству узлов сетки)
	const std::vector<std::set<typename Triangulation<2>::active_cell_iterator>> vertex_to_cells(GridTools::vertex_to_cell_map(tria));
	double shapeValue;

	for(std::map<int, meshNode>::iterator it = movingNodes.begin(); it != movingNodes.end(); ++it){
		//для граничных узлов поиск ячейки, вычисление локальных координат и переинтерполяция полей не выполняются, значения полей остаются неизменными (определяются ГУ)
		if(it->second.isBoundaryNode){
			newSolutionVx[it->second.dofScalarIndex] += locally_relevant_solutionVx[it->second.dofScalarIndex];
			newSolutionVy[it->second.dofScalarIndex] += locally_relevant_solutionVy[it->second.dofScalarIndex];
			newSolutionP[it->second.dofScalarIndex] += locally_relevant_solutionP[it->second.dofScalarIndex];
			nodeWeights[it->second.dofScalarIndex] += 1;
			
			continue;
		}
		
		typename Triangulation<2>::active_cell_iterator cell(&tria, tria.n_levels() - 1, it->second.cellIndex);
		
		//временная переменная для новых координат (обновлять в структуре для узла нельзя, т.к. иначе затем при трансформации сетки не получится найти
		//для произвольной точки (подставляет сама функция GridTools::transform) узел сетки, чтобы взять соответствующие перемещения, поскольку в Triangulation положения еще не обновлены
		Point<2> newCoords = it->second.coords;
		newCoords[0] += locally_relevant_solutionU[it->second.dofUindexX];
		newCoords[1] += locally_relevant_solutionU[it->second.dofUindexY];
		
		bool cellFound = false;
		Point<2> localCoords;
		
		try {
			localCoords = mapping.transform_real_to_unit_cell(cell, newCoords);
		} catch(typename Mapping<2>::ExcTransformationFailed &){
			//std::cout << "Transformation failed for process no. " << this_mpi_process << ", dof no. " << it->second.dofScalarIndex << std::endl;
		}
		
		if(GeometryInfo<2>::is_inside_unit_cell(localCoords, 1e-5)) cellFound = true;
		else {
			//ячейки, которым принадлежит текущий узел (при смещении он попасть в одну из них)
			const std::set<typename Triangulation<2>::active_cell_iterator> adjacentCells = vertex_to_cells.at(it->first);
			for(auto adjIt = adjacentCells.begin(); adjIt != adjacentCells.end(); ++adjIt){
				if(*adjIt == cell) continue;	//одной из ячеек всегда будет та, которая уже проверена ранее
				
				try {
					localCoords = mapping.transform_real_to_unit_cell(*adjIt, newCoords);
				} catch(typename Mapping<2>::ExcTransformationFailed &){
					//std::cout << "Transformation failed for process no. " << this_mpi_process << ", dof no. " << it->second.dofScalarIndex << std::endl;
				}
				
				if(GeometryInfo<2>::is_inside_unit_cell(localCoords, 1e-5)){
					cell = *adjIt;
					cellFound = true;
					break;
				}
			}
		}
		
		if(!cellFound){
			//значения полей остаются прежними
			newSolutionVx[it->second.dofScalarIndex] += locally_relevant_solutionVx[it->second.dofScalarIndex];
			newSolutionVy[it->second.dofScalarIndex] += locally_relevant_solutionVy[it->second.dofScalarIndex];
			newSolutionP[it->second.dofScalarIndex] += locally_relevant_solutionP[it->second.dofScalarIndex];
			nodeWeights[it->second.dofScalarIndex] += 1;
		
			continue;
		}
		
		//интерполяция полей в новых координатах текущего узла по найденной ячейке
		DoFHandler<2>::active_cell_iterator dofCell = dofHandlerCells[cell->index()];
		
		for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; ++vertex){
			shapeValue = fe.shape_value(vertex, localCoords);

			newSolutionVx[it->second.dofScalarIndex] += shapeValue * locally_relevant_solutionVx[dofCell->vertex_dof_index(vertex,0)];
			newSolutionVy[it->second.dofScalarIndex] += shapeValue * locally_relevant_solutionVy[dofCell->vertex_dof_index(vertex,0)];
			newSolutionP[it->second.dofScalarIndex] += shapeValue * locally_relevant_solutionP[dofCell->vertex_dof_index(vertex,0)];
		}//vertex
		nodeWeights[it->second.dofScalarIndex] += 1;
	}

	newSolutionVx.compress(VectorOperation::add);
	newSolutionVy.compress(VectorOperation::add);
	newSolutionP.compress(VectorOperation::add);
	nodeWeights.compress(VectorOperation::add);
	
	for (unsigned int i = newSolutionVx.local_range().first; i < newSolutionVx.local_range().second; ++i) {
		newSolutionVx(i) /= nodeWeights(i);
		newSolutionVy(i) /= nodeWeights(i);
		newSolutionP(i) /= nodeWeights(i);
	}

	locally_relevant_solutionVx = newSolutionVx;
	locally_relevant_solutionVy = newSolutionVy;
	locally_relevant_solutionP = newSolutionP;
}

void pfem2Solver::transform_grid()
{
	TimerOutput::Scope timer_section(*timer, "Grid transformation");
	
	std::set<unsigned int> updated_vertices;

	for(const auto &cell : dof_handlerV.active_cell_iterators())
		if(cell->is_locally_owned())
			for(unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
				if(!updated_vertices.count(cell->vertex_index(i))){
					meshNode &mNode = movingNodes[cell->vertex_index(i)];
					
					cell->vertex(i) = Point<2>(cell->vertex(i)[0] + locally_relevant_solutionU[mNode.dofUindexX], cell->vertex(i)[1] + locally_relevant_solutionU[mNode.dofUindexY]);
					
					mNode.coords[0] += locally_relevant_solutionU[mNode.dofUindexX];
					mNode.coords[1] += locally_relevant_solutionU[mNode.dofUindexY];
					
					updated_vertices.insert(cell->vertex_index(i));
				}
				
	tria.signals.mesh_movement();
}
