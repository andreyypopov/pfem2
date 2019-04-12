#include "pfem2particle.h"

#include <iostream>
#include <fstream>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_poly.h>

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

#include <map>

pfem2Particle::pfem2Particle(const Point<2> & location,const Point<2> & reference_location,const types::particle_index id)
	: Particle<2>(location, reference_location, id),
	velocity({0,0})
{

}

const Tensor<1,2> & pfem2Particle::getVelocity() const
{
	return velocity;
}
	
void pfem2Particle::setVelocity (const Tensor<1,2> &new_velocity)
{
	velocity = new_velocity;
}

pfem2Solver::pfem2Solver()
	: tria(MPI_COMM_WORLD,Triangulation<2>::maximum_smoothing),
	particle_handler(tria,mapping),
	feVx (1),
	feVy (1),
	feP (1),
	dof_handlerVx (tria),
	dof_handlerVy (tria),
	dof_handlerP (tria),
	quantities({0,0})
{
	
}

pfem2Solver::~pfem2Solver()
{
	
}

void pfem2Solver::seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell)
{
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	FESystem<2> fe(FE_Q<2>(1), 1);
	double shapeValue;
	
	for(unsigned int i = 0; i < quantities[0]; ++i){
		for(unsigned int j = 0; j < quantities[1]; ++j){
			pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
			particle_handler.insert_particle(*particle, cell);
			
			velocity_x[particle->get_id()] = 0.0;
			velocity_y[particle->get_id()] = 0.0;
			
			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				shapeValue = fe.shape_value(vertex, particle->get_reference_location());

				velocity_x[particle->get_id()] += shapeValue * solutionVx(cell->vertex_dof_index(vertex,0));
				velocity_y[particle->get_id()] += shapeValue * solutionVy(cell->vertex_dof_index(vertex,0));
			}//vertex
		}
	}
}

bool pfem2Solver::check_cell_for_empty_parts (const typename DoFHandler<2>::cell_iterator &cell)
{
	bool res = false;
	
	std::map<std::vector<unsigned int>, unsigned int> particlesInParts;
	
	//определение, в каких частях ячейки лежат частицы
	double hx = 1.0/quantities[0];
	double hy = 1.0/quantities[1];
	
	unsigned int num_x, num_y;
	for(ParticleIterator<2> particleIndex = particle_handler.particles_in_cell(cell).begin(); particleIndex != particle_handler.particles_in_cell(cell).end(); ++particleIndex){
		num_x = particleIndex->get_reference_location()(0)/hx;
		num_y = particleIndex->get_reference_location()(1)/hy;

		particlesInParts[{num_x,num_y}]++;
	}
	
	FESystem<2> fe(FE_Q<2>(1), 1);
	double shapeValue;
	
	//проверка каждой части ячейки на количество частиц: при 0 - подсевание 1 частицы в центр, при > max разрешенного значения - удаление лишних частиц
	for(unsigned int i = 0; i < quantities[0]; i++){
		for(unsigned int j = 0; j < quantities[1]; j++){
			if(!particlesInParts[{i,j}]){
				pfem2Particle* particle = new pfem2Particle(mapping.transform_unit_to_real_cell(cell, Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy)), Point<2>((i + 1.0/2)*hx, (j+1.0/2)*hy), ++particleCount);
				particle_handler.insert_particle(*particle, cell);
				
				velocity_x[particle->get_id()] = 0.0;
				velocity_y[particle->get_id()] = 0.0;
				
				for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
					shapeValue = fe.shape_value(vertex, particle->get_reference_location());

					velocity_x[particle->get_id()] += shapeValue * solutionVx(cell->vertex_dof_index(vertex,0));
					velocity_y[particle->get_id()] += shapeValue * solutionVy(cell->vertex_dof_index(vertex,0));
				}//vertex
				
				res = true;
			}/* else if(particlesInParts[{i,j}] > MAX_PARTICLES_PER_CELL_PART){		WORKS INCORRECTLY
				for(ParticleIterator<2> particleIndex = particle_handler.particles_in_cell(cell).begin(); particleIndex != particle_handler.particles_in_cell(cell).end(); ++particleIndex){
					num_x = particleIndex->get_reference_location()(0)/hx;
					num_y = particleIndex->get_reference_location()(1)/hy;
					
					if(num_x == i && num_y == j){
						particle_handler.remove_particle(particleIndex);
						if(--particlesInParts[{i,j}] == MAX_PARTICLES_PER_CELL_PART) break;						
					}
				}
				
				res = true;
			}*/
		}
	}
	
	return res;
}

void pfem2Solver::seed_particles(const std::vector < unsigned int > & quantities)
{
	TimerOutput::Scope timer_section(*timer, "Particles' seeding");
	
	if(quantities.size() < 2){ return; }
	
	this->quantities = quantities;
	
	typename DoFHandler<2>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell) {
		seed_particles_into_cell(cell);
	}
	
	particle_handler.update_cached_numbers();
	
	std::cout << "Created and placed " << particleCount << " particles" << std::endl;
	std::cout << "Particle handler contains " << particle_handler.n_global_particles() << " particles" << std::endl;
}

void pfem2Solver::correct_particles_velocities()
{
	TimerOutput::Scope timer_section(*timer, "Particles' velocities correction");
	
	FESystem<2>   fe(FE_Q<2>(1), 1);
	
	double shapeValue;
			
	typename DoFHandler<2>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell) {
		for(ParticleIterator<2> particleIndex = particle_handler.particles_in_cell(cell).begin(); 
		                                   particleIndex != particle_handler.particles_in_cell(cell).end(); ++particleIndex) {
		
			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				shapeValue = fe.shape_value(vertex, particleIndex->get_reference_location());

				velocity_x[particleIndex->get_id()] += shapeValue * ( solutionVx(cell->vertex_dof_index(vertex,0)) - old_solutionVx(cell->vertex_dof_index(vertex,0)) );
				velocity_y[particleIndex->get_id()] += shapeValue * ( solutionVy(cell->vertex_dof_index(vertex,0)) - old_solutionVy(cell->vertex_dof_index(vertex,0)) );
			}//vertex
		}//particle
	}//cell
	
	//std::cout << "Finished correcting particles' velocities" << std::endl;	
}

void pfem2Solver::move_particles() //перенос частиц
{
	TimerOutput::Scope timer_section(*timer, "Particles' movement");	
	
	FESystem<2>   fe(FE_Q<2>(1), 1);

	Point<2> vel_in_part;
	
	double shapeValue;
	double min_time_step = time_step / PARTICLES_MOVEMENT_STEPS;
	
	for (int np_m = 0; np_m < PARTICLES_MOVEMENT_STEPS; ++np_m) {
		//РАЗДЕЛИТЬ НА VX И VY!!!!!!!!
		typename DoFHandler<2>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
		
		for (; cell != endc; ++cell) {
			for( ParticleIterator<2> particleIndex = particle_handler.particles_in_cell(cell).begin(); 
		                                   particleIndex != particle_handler.particles_in_cell(cell).end(); ++particleIndex ) {

				vel_in_part = Point<2> (0,0);
				
				for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
					shapeValue = fe.shape_value(vertex, particleIndex->get_reference_location());
					vel_in_part[0] += shapeValue * solutionVx(cell->vertex_dof_index(vertex,0));
					vel_in_part[1] += shapeValue * solutionVy(cell->vertex_dof_index(vertex,0));
				}//vertex
				
				vel_in_part[0] *= min_time_step;
				vel_in_part[1] *= min_time_step;
				
				particleIndex->set_location(particleIndex->get_location() + vel_in_part);
			}//particle
		}//cell
		
		particle_handler.sort_particles_into_subdomains_and_cells();
	}//np_m
	
	//проверка наличия пустых ячеек (без частиц) и размещение в них частиц
	typename DoFHandler<2>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	
	bool cell_modified = false;
	for (; cell != endc; ++cell) {
		bool res = check_cell_for_empty_parts(cell);
		if(res) cell_modified = true;
	}
	
	if (cell_modified) particle_handler.update_cached_numbers();

	//std::cout << "Finished moving particles" << std::endl;
}

void pfem2Solver::distribute_particle_velocities_to_grid() //перенос скоростей частиц на узлы сетки
{	
	TimerOutput::Scope timer_section(*timer, "Distribution of particles' velocities to grid nodes");
		
	FESystem<2>   fe(FE_Q<2>(1), 1);
	
	Vector<double> node_velocityX, node_velocityY;
	Vector<double> node_weights;
	
	double shapeValue;
	
	node_velocityX.reinit (tria.n_vertices(), 0.0);
	node_velocityY.reinit (tria.n_vertices(), 0.0);
	node_weights.reinit (tria.n_vertices(), 0.0);
	
	typename DoFHandler<2>::cell_iterator cell = dof_handlerVx.begin(tria.n_levels()-1), endc = dof_handlerVx.end(tria.n_levels()-1);
	for (; cell != endc; ++cell) {
	
		for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				
			for (ParticleIterator<2> particleIndex = particle_handler.particles_in_cell(cell).begin(); 
	                                   particleIndex != particle_handler.particles_in_cell(cell).end(); ++particleIndex ){
										   
				shapeValue = fe.shape_value(vertex, particleIndex->get_reference_location());
										   
				node_velocityX[cell->vertex_dof_index(vertex,0)] += shapeValue * velocity_x[particleIndex->get_id()];
				node_velocityY[cell->vertex_dof_index(vertex,0)] += shapeValue * velocity_y[particleIndex->get_id()];
				
				node_weights[cell->vertex_dof_index(vertex,0)] += shapeValue;
			
			}//particle
		}//vertex
	}//cell
	
	for (unsigned int i=0; i<tria.n_vertices(); ++i) {
		node_velocityX[i] /= node_weights[i];
		node_velocityY[i] /= node_weights[i];
	}//i
		
	solutionVx = node_velocityX;
	solutionVy = node_velocityY;
	
	//std::cout << "Finished distributing particles' velocities to grid" << std::endl;	 
}

void pfem2Solver::calculate_loads(types::boundary_id patch_id, std::ofstream *out){
	TimerOutput::Scope timer_section(*timer, "Loads calculation");
	
	DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active(), endc = dof_handlerP.end();
	QGauss<1> face_quadrature_formula(2);
	FEFaceValues<2> feP_face_values (feP, face_quadrature_formula,
                                    update_values    | update_normal_vectors |
                                    update_quadrature_points  | update_JxW_values);
    const unsigned int   n_face_q_points = face_quadrature_formula.size();
			
	double Fx = 0.0, Fy = 0.0, point_valueP;//, Cx, Cy;
		
	for (; cell != endc; ++cell) {
		for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number) {
			if ( (cell->face(face_number)->at_boundary()) &&  (cell->face(face_number)->boundary_id() == patch_id) ) {
				feP_face_values.reinit (cell, face_number);
							
				for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point) {
					point_valueP = 0.0;
					
					for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
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
			
	*out << time << ";" << Fx << ";" << Fy /*<< ";" << Cx << ";" << Cy << ";"*/ << std::endl;
	
	//std::cout << "Calculating loads finished" << std::endl;
}
