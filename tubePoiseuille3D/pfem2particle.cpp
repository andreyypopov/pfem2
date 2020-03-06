#include "pfem2particle.h"

#include <iostream>
#include <fstream>

#include <deal.II/base/std_cxx14/memory.h>

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

#include "omp.h"

pfem2Particle::pfem2Particle(const Point<3> & location,const Point<3> & reference_location,const unsigned id)
	: location (location),
    reference_location (reference_location),
    id (id),
	velocity({0,0,0})
{

}

void pfem2Particle::set_location (const Point<3> &new_location)
{
	location = new_location;
}

const Point<3> & pfem2Particle::get_location () const
{
    return location;
}

void pfem2Particle::set_reference_location (const Point<3> &new_reference_location)
{
    reference_location = new_reference_location;
}

const Point<3> & pfem2Particle::get_reference_location () const
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

const Tensor<1,3> & pfem2Particle::get_velocity() const
{
	return velocity;
}

const Tensor<1,3> & pfem2Particle::get_velocity_ext() const
{
	return velocity_ext;
}

double pfem2Particle::get_velocity_component(int component) const
{
	return velocity[component];
}

void pfem2Particle::set_velocity (const Tensor<1,3> &new_velocity)
{
	velocity = new_velocity;
}

void pfem2Particle::set_velocity_component (const double value, int component)
{
	velocity[component] = value;
}

void pfem2Particle::set_velocity_ext (const Tensor<1,3> &new_ext_velocity)
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

Triangulation<3>::cell_iterator pfem2Particle::get_surrounding_cell(const Triangulation<3> &triangulation) const
{
	const typename Triangulation<3>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position);
	
	return cell;
}

unsigned int pfem2Particle::find_closest_vertex_of_cell(const typename Triangulation<3>::active_cell_iterator &cell, const Mapping<3> &mapping)
{
	//transformation of local particle coordinates transformation is required as the global particle coordinates have already been updated by the time this function is called
	const Point<3> old_position = mapping.transform_unit_to_real_cell(cell, reference_location);
	
	Tensor<1,3> velocity_normalized = velocity_ext / velocity_ext.norm();
	Tensor<1,3> particle_to_vertex = cell->vertex(0) - old_position;
    particle_to_vertex /= particle_to_vertex.norm();
    
    double maximum_angle = velocity_normalized * particle_to_vertex;
    unsigned int closest_vertex = 0;
    
    for (unsigned int v = 1; v < GeometryInfo<3>::vertices_per_cell; ++v){
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

pfem2ParticleHandler::pfem2ParticleHandler(const parallel::distributed::Triangulation<3> &tria, const Mapping<3> &coordMapping)
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

void pfem2ParticleHandler::clear()
{
	clear_particles();
	global_number_of_particles    = 0;
    global_max_particles_per_cell = 0;
}

void pfem2ParticleHandler::clear_particles()
{
	for(auto particleIndex = particles.begin(); particleIndex != particles.end(); ++particleIndex) delete (*particleIndex).second;
	particles.clear();
}

void pfem2ParticleHandler::remove_particle(const pfem2Particle *particle)
{
	particles.erase(particle->get_map_position());
	delete particle;
}

void pfem2ParticleHandler::insert_particle(pfem2Particle *particle,
										   const typename Triangulation<3>::active_cell_iterator &cell)
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

unsigned int pfem2ParticleHandler::n_particles_in_cell(const typename Triangulation<3>::active_cell_iterator &cell) const
{
	return particles.count(cell->index());
}

bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,3> &particle_direction, const std::vector<Tensor<1,3> > &center_directions)
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
		const typename Triangulation<3>::cell_iterator cell = (*it).second->get_surrounding_cell(*triangulation);
		
		try{
			const Point<3> p_unit = mapping->transform_real_to_unit_cell(cell, (*it).second->get_location());
		
			if(GeometryInfo<3>::is_inside_unit_cell(p_unit)) (*it).second->set_reference_location(p_unit);
			else particles_out_of_cell.push_back(it);
		} catch(typename Mapping<3>::ExcTransformationFailed &){
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
	std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator> moved_particles, particles_to_be_deleted;
	
	typedef typename std::vector<std::pair<int, pfem2Particle*>>::size_type vector_size;
	typedef typename std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator>::size_type vector_size2;
	
	sorted_particles.reserve(static_cast<vector_size> (particles_out_of_cell.size()*1.25));
	moved_particles.reserve(static_cast<vector_size2> (particles_out_of_cell.size()*1.25));
	particles_to_be_deleted.reserve(static_cast<vector_size2> (particles_out_of_cell.size()*1.25));

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
      const std::vector<std::set<typename Triangulation<3>::active_cell_iterator>> vertex_to_cells(GridTools::vertex_to_cell_map(*triangulation));
      const std::vector<std::vector<Tensor<1,3>>> vertex_to_cell_centers(GridTools::vertex_to_cell_centers_directions(*triangulation,vertex_to_cells));
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
		  
		  Point<3> current_reference_position;
          bool found_cell = false;

		  auto particle = (*it);

          typename Triangulation<3>::active_cell_iterator current_cell = (*particle).second->get_surrounding_cell(*triangulation);

          const unsigned int closest_vertex = (*particle).second->find_closest_vertex_of_cell(current_cell, *mapping);
          Tensor<1,3> vertex_to_particle = (*particle).second->get_location() - current_cell->vertex(closest_vertex);
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
		      typename std::set<typename Triangulation<3>::active_cell_iterator>::const_iterator cell = vertex_to_cells[closest_vertex_index].begin();

              std::advance(cell,neighbor_permutation[i]);
              
              try{
				  const Point<3> p_unit = mapping->transform_real_to_unit_cell(*cell, (*particle).second->get_location());
				  if (GeometryInfo<3>::is_inside_unit_cell(p_unit)){
					current_cell = *cell;
					current_reference_position = p_unit;
					(*particle).second->set_tria_position(current_cell->index());
					found_cell = true;
					
#ifdef VERBOSE_OUTPUT
					//std::cout << "Particle found in a neighbour cell" << std::endl;
#endif // VERBOSE_OUTPUT
					
					break; 
				  }
              } catch(typename Mapping<3>::ExcTransformationFailed &)
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
                  const std::pair<const typename Triangulation<3>::active_cell_iterator, Point<3> > current_cell_and_position =
                          GridTools::find_active_cell_around_point<3> (*mapping, *triangulation, (*particle).second->get_location());
              
                  current_cell = current_cell_and_position.first;
                  current_reference_position = current_cell_and_position.second;
                  //std::cout << "Particle found in a far away cell" << std::endl;
              } catch (GridTools::ExcPointNotFound<3> &){				  
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
          sorted_particles.push_back(std::make_pair(current_cell->index(), (*particle).second));
          moved_particles.push_back(particle);
                   
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
	sorted_particles_map.insert(sorted_particles.begin(), sorted_particles.end());
	
#ifdef VERBOSE_OUTPUT
	double presortingEnd = omp_get_wtime();
#endif // VERBOSE_OUTPUT
		
	for (unsigned int i=0; i<particles_to_be_deleted.size(); ++i){
		auto particle = particles_to_be_deleted[i];
		delete (*particle).second;
		particles.erase(particle);
	}
	
#ifdef VERBOSE_OUTPUT
	double particlesDeleteClock = omp_get_wtime();
#endif // VERBOSE_OUTPUT
	
	for (unsigned int i=0; i<moved_particles.size(); ++i) particles.erase(moved_particles[i]);

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

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::begin()
{
	return particles.begin();
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::end()
{
	return particles.end();
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::particles_in_cell_begin(const typename Triangulation<3>::active_cell_iterator &cell)
{
	return particles.equal_range(cell->index()).first;
}

std::unordered_multimap<int, pfem2Particle*>::iterator pfem2ParticleHandler::particles_in_cell_end(const typename Triangulation<3>::active_cell_iterator &cell)
{
	return particles.equal_range(cell->index()).second;
}

pfem2Solver::pfem2Solver()
	: tria(MPI_COMM_WORLD,Triangulation<3>::maximum_smoothing),
	particle_handler(tria, mapping),
	feVx (1),
	feVy (1),
	feVz (1),
	feP (1),
	fe(FE_Q<3>(1), 1),
	dof_handlerVx (tria),
	dof_handlerVy (tria),
	dof_handlerVz (tria),
	dof_handlerP (tria),
	quantities({0,0,0})
{
	
}

pfem2Solver::~pfem2Solver()
{
	
}

void pfem2Solver::seed_particles_into_cell (const typename DoFHandler<3>::cell_iterator &cell)
{
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	double hz = 1.0/quantities[2];
	
	double shapeValue;
	
	for(unsigned int i = 0; i < quantities[0]; ++i)
		for(unsigned int j = 0; j < quantities[1]; ++j)
			for(unsigned int k = 0; k < quantities[2]; ++k){
				pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<3>((i + 1.0/2)*hx, (j+1.0/2)*hy, (k+1.0/2)*hz)), Point<3>((i + 1.0/2)*hx, (j+1.0/2)*hy, (k+1.0/2)*hz), ++particleCount);
				particle_handler.insert_particle(particle, cell);
			
				for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex){
					shapeValue = fe.shape_value(vertex, particle->get_reference_location());

					particle->set_velocity_component(particle->get_velocity_component(0) + shapeValue * solutionVx(cell->vertex_dof_index(vertex,0)), 0);
					particle->set_velocity_component(particle->get_velocity_component(1) + shapeValue * solutionVy(cell->vertex_dof_index(vertex,0)), 1);
					particle->set_velocity_component(particle->get_velocity_component(2) + shapeValue * solutionVz(cell->vertex_dof_index(vertex,0)), 2);
				}//vertex
			}
}

bool pfem2Solver::check_cell_for_empty_parts (const typename DoFHandler<3>::cell_iterator &cell)
{
	bool res = false;
	
	std::map<std::vector<unsigned int>, unsigned int> particlesInParts;
	std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator> particles_to_be_deleted;
	
	//определение, в каких частях ячейки лежат частицы
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	double hz = 1.0/quantities[2];
	
	unsigned int num_x, num_y, num_z;
	for(auto particleIndex = particle_handler.particles_in_cell_begin(cell); particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex){
		num_x = (*particleIndex).second->get_reference_location()(0)/hx;
		num_y = (*particleIndex).second->get_reference_location()(1)/hy;
		num_z = (*particleIndex).second->get_reference_location()(2)/hz;

		particlesInParts[{num_x,num_y,num_z}]++;
		if(particlesInParts[{num_x,num_y,num_z}] > MAX_PARTICLES_PER_CELL_PART) particles_to_be_deleted.push_back(particleIndex);
	}
	
	double shapeValue;
	
	//проверка каждой части ячейки на количество частиц: при 0 - подсевание 1 частицы в центр
	for(unsigned int i = 0; i < quantities[0]; i++)
		for(unsigned int j = 0; j < quantities[1]; j++)
			for(unsigned int k = 0; k < quantities[2]; ++k){
				if(!particlesInParts[{i,j,k}]){
					pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<3>((i + 1.0/2)*hx, (j+1.0/2)*hy, (k+1.0/2)*hz)), Point<3>((i + 1.0/2)*hx, (j+1.0/2)*hy, (k+1.0/2)*hz), ++particleCount);
					particle_handler.insert_particle(particle, cell);
				
					for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex){
						shapeValue = fe.shape_value(vertex, particle->get_reference_location());

						particle->set_velocity_component(particle->get_velocity_component(0) + shapeValue * solutionVx(cell->vertex_dof_index(vertex,0)), 0);
						particle->set_velocity_component(particle->get_velocity_component(1) + shapeValue * solutionVy(cell->vertex_dof_index(vertex,0)), 1);
						particle->set_velocity_component(particle->get_velocity_component(2) + shapeValue * solutionVz(cell->vertex_dof_index(vertex,0)), 2);
					}//vertex	
				
					res = true;
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
	
	if(quantities.size() < 3) return;
	
	this->quantities = quantities;
	
	typename DoFHandler<3>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell) seed_particles_into_cell(cell);
	
	std::cout << "Created and placed " << particleCount << " particles" << std::endl;
	std::cout << "Particle handler contains " << particle_handler.n_global_particles() << " particles" << std::endl;
}

void pfem2Solver::correct_particles_velocities()
{
	TimerOutput::Scope timer_section(*timer, "Particles' velocities correction");
	
	double shapeValue;
			
	typename DoFHandler<3>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell)
		for(auto particleIndex = particle_handler.particles_in_cell_begin(cell); 
		                                   particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex) {
		
			for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex){
				shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());

				(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(0) + shapeValue * ( solutionVx(cell->vertex_dof_index(vertex,0)) - old_solutionVx(cell->vertex_dof_index(vertex,0)) ), 0);
				(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(1) + shapeValue * ( solutionVy(cell->vertex_dof_index(vertex,0)) - old_solutionVy(cell->vertex_dof_index(vertex,0)) ), 1);
				(*particleIndex).second->set_velocity_component((*particleIndex).second->get_velocity_component(2) + shapeValue * ( solutionVz(cell->vertex_dof_index(vertex,0)) - old_solutionVz(cell->vertex_dof_index(vertex,0)) ), 2);
			}//vertex
		}//particle
	
	//std::cout << "Finished correcting particles' velocities" << std::endl;	
}

void pfem2Solver::move_particles() //перенос частиц
{
	TimerOutput::Scope timer_section(*timer, "Particles' movement");	
	
	Tensor<1,3> vel_in_part;
	
	double shapeValue;
	double min_time_step = time_step / PARTICLES_MOVEMENT_STEPS;
	
	for (int np_m = 0; np_m < PARTICLES_MOVEMENT_STEPS; ++np_m) {
		typename DoFHandler<3>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
		
		for (; cell != endc; ++cell)
			for( auto particleIndex = particle_handler.particles_in_cell_begin(cell); 
		                                   particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex ) {

				vel_in_part = Tensor<1,3> ({0,0});
				
				for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex){
					shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());
					vel_in_part[0] += shapeValue * solutionVx(cell->vertex_dof_index(vertex,0));
					vel_in_part[1] += shapeValue * solutionVy(cell->vertex_dof_index(vertex,0));
					vel_in_part[2] += shapeValue * solutionVz(cell->vertex_dof_index(vertex,0));
				}//vertex
				
				vel_in_part[0] *= min_time_step;
				vel_in_part[1] *= min_time_step;
				vel_in_part[2] *= min_time_step;
				
				(*particleIndex).second->set_location((*particleIndex).second->get_location() + vel_in_part);
				(*particleIndex).second->set_velocity_ext(vel_in_part);
			}//particle
		
		particle_handler.sort_particles_into_subdomains_and_cells();
	}//np_m
	
	//проверка наличия пустых ячеек (без частиц) и размещение в них частиц
	typename DoFHandler<3>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	
	for (; cell != endc; ++cell) check_cell_for_empty_parts(cell);
	
	//std::cout << "Finished moving particles" << std::endl;
}

void pfem2Solver::distribute_particle_velocities_to_grid() //перенос скоростей частиц на узлы сетки
{	
	TimerOutput::Scope timer_section(*timer, "Distribution of particles' velocities to grid nodes");
		
	Vector<double> node_velocityX, node_velocityY, node_velocityZ;
	Vector<double> node_weights;
	
	double shapeValue;
	
	node_velocityX.reinit (tria.n_vertices(), 0.0);
	node_velocityY.reinit (tria.n_vertices(), 0.0);
	node_velocityZ.reinit (tria.n_vertices(), 0.0);
	node_weights.reinit (tria.n_vertices(), 0.0);
	
	typename DoFHandler<3>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell)	
		for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex)				
			for (auto particleIndex = particle_handler.particles_in_cell_begin(cell); 
	                                   particleIndex != particle_handler.particles_in_cell_end(cell); ++particleIndex ){
										   
				shapeValue = fe.shape_value(vertex, (*particleIndex).second->get_reference_location());
										   
				node_velocityX[cell->vertex_dof_index(vertex,0)] += shapeValue * (*particleIndex).second->get_velocity_component(0);
				node_velocityY[cell->vertex_dof_index(vertex,0)] += shapeValue * (*particleIndex).second->get_velocity_component(1);
				node_velocityZ[cell->vertex_dof_index(vertex,0)] += shapeValue * (*particleIndex).second->get_velocity_component(2);
				
				node_weights[cell->vertex_dof_index(vertex,0)] += shapeValue;			
			}//particle
	
	for (unsigned int i=0; i<tria.n_vertices(); ++i) {
		node_velocityX[i] /= node_weights[i];
		node_velocityY[i] /= node_weights[i];
		node_velocityZ[i] /= node_weights[i];
	}//i
		
	solutionVx = node_velocityX;
	solutionVy = node_velocityY;
	solutionVz = node_velocityZ;
	
	//std::cout << "Finished distributing particles' velocities to grid" << std::endl;	 
}

void pfem2Solver::calculate_loads(types::boundary_id patch_id, std::ofstream *out){
	TimerOutput::Scope timer_section(*timer, "Loads calculation");
	
	DoFHandler<3>::active_cell_iterator cell = dof_handlerP.begin_active(), endc = dof_handlerP.end();
	QGauss<2> face_quadrature_formula(2);
	FEFaceValues<3> feP_face_values (feP, face_quadrature_formula, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
			
	double Fx = 0.0, Fy = 0.0, point_valueP;//, Cx, Cy;
		
	for (; cell != endc; ++cell) {
		for (unsigned int face_number=0; face_number < GeometryInfo<3>::faces_per_cell; ++face_number) {
			if ( (cell->face(face_number)->at_boundary()) &&  (cell->face(face_number)->boundary_id() == patch_id) ) {
				feP_face_values.reinit (cell, face_number);
							
				for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point) {
					point_valueP = 0.0;
					
					for (unsigned int vertex=0; vertex<GeometryInfo<3>::vertices_per_cell; ++vertex){
						point_valueP += solutionP(cell->vertex_dof_index(vertex,0)) * feP_face_values.shape_value(vertex, q_point);
					}//vertex
										
					Fx += point_valueP * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW (q_point);
					Fy += point_valueP * feP_face_values.normal_vector(q_point)[1] * feP_face_values.JxW (q_point);
					
				}//q_index
			}//if
		}//face_number
	}//cell

	//Cx = 2.0 * Fx / (0.1 * 0.1);
	//Cy = 2.0 * Fy / (0.1 * 0.1);
		
	//pressure at the point of flow deceleration
	double p_point = 0.0;
	if(probeDoFnumbers.size() == 1) p_point = solutionP(probeDoFnumbers.front());
	else {
		for(std::vector<unsigned int>::iterator it = probeDoFnumbers.begin(); it != probeDoFnumbers.end(); ++it) p_point += solutionP(*it);
		p_point /= probeDoFnumbers.size();
	}
			
	*out << time << ";" << Fx << ";" << Fy << ";" << p_point /*<< ";" << Cx << ";" << Cy << ";"*/ << std::endl;
	
	//std::cout << "Calculating loads finished" << std::endl;
}
