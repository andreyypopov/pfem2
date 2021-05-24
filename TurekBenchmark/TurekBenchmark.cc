#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/base/parameter_handler.h>

#include "pfem2particle.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string>

using namespace dealii;

class parabolicBC : public Function<2>
{
public:
	parabolicBC() : Function<2>() {}
	
	virtual double value (const Point<2> &p, const unsigned int component = 0) const;
};

double parabolicBC::value(const Point<2> &p, const unsigned int) const
{
	return 4 * 1.5 * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
}

class velocityPredictionBC : public Function<2>
{
public:
	velocityPredictionBC(double timestep_,double rho_, Functions::FEFieldFunction<2, DoFHandler<2>, TrilinosWrappers::MPI::Vector>& pressureFieldFun_) 
	: Function<2>()
	, timestep(timestep_)
	, rho(rho_)
	, index(0)
	, bcType(0)
	, pressureFieldFun(&pressureFieldFun_)
		 {}
	
	virtual double value (const Point<2> &p, const unsigned int component = 0) const;
	
	void updateIndex(const unsigned int index_){ index = index_; }
	void updateBcType(const unsigned int bcType_){ bcType = bcType_; }
	
	const double timestep, rho;
	unsigned int index, bcType;
	Functions::FEFieldFunction<2, DoFHandler<2>, TrilinosWrappers::MPI::Vector> *pressureFieldFun;
};

double velocityPredictionBC::value(const Point<2> &p, const unsigned int) const
{
	double val(0.0);
	if(index == 0 && bcType == 1) val = 4 * 1.5 * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
	
	double grad;
	bool pointFound = true;
	try {
		grad = timestep / rho * pressureFieldFun->gradient(p)[index];
	} catch (const VectorTools::ExcPointNotAvailableHere &){
		pointFound = false;
	}
	
	if(pointFound) val += grad;
	
	return val;
}

class TurekBenchmark : public pfem2Solver
{
public:
	TurekBenchmark();

	static void declare_parameters (ParameterHandler &prm);
	void get_parameters (ParameterHandler &prm);
	void build_grid ();
	void setup_system();
	void solveVx(bool correction = false);
	void solveVy(bool correction = false);
	void solveP();
	void output_results(bool predictionCorrection = false);
	void run();
	
	void fem_step();
	void check_balance(std::ofstream *out);

	FullMatrix<double> local_matrixVx, local_matrixVy, local_matrixP;
	
	Vector<double> local_rhsVx, local_rhsVy, local_rhsP;
	
	TrilinosWrappers::SparseMatrix system_mVx, system_mVy, system_mP;
	TrilinosWrappers::MPI::Vector system_rVx, system_rVy, system_rP;
	
private:
	double final_time_, accuracy_;
	int num_of_part_x_, num_of_part_y_;
	double velX_inlet_, velX_wall_, velX_cyl_,
		   velY_inlet_, velY_wall_, velY_cyl_,
		   press_outlet_;
	int num_of_iter_, num_of_particles_x_, num_of_particles_y_, num_of_data_;
	std::string mesh_file_;  
};

TurekBenchmark::TurekBenchmark()
	: pfem2Solver(),
	local_matrixVx (dofs_per_cellV, dofs_per_cellV),
	local_matrixVy (dofs_per_cellV, dofs_per_cellV),
	local_matrixP (dofs_per_cellP, dofs_per_cellP),
	local_rhsVx (dofs_per_cellV),
	local_rhsVy (dofs_per_cellV),
	local_rhsP (dofs_per_cellP)
{
	diam = 0.1;
	uMean = 1.0;	//Re = 100;
	//uMean = 0.2;	//Re = 20;
}

void TurekBenchmark::declare_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		prm.declare_entry ("Dynamic viscosity", "1.0");
		prm.declare_entry ("Density", "1.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Time parameters");
	{
		prm.declare_entry ("Initial time", "0.0");
		prm.declare_entry ("Time step", "0.05");
		prm.declare_entry ("Number of time step", "1");
		prm.declare_entry ("Final Time", "100.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Solver parameters");
	{
		prm.declare_entry ("Accuracy", "1e-16");
		prm.declare_entry ("Number of iterations", "1000");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Boundary Values");
	{
		//prm.declare_entry ("Neuman value on the outlet boundary", "0.0");
		prm.declare_entry ("VelocityX value on the inlet boundary", "0.0");
		prm.declare_entry ("VelocityX value on the wall boundary", "0.0");
		prm.declare_entry ("VelocityX value on the cylinder boundary", "0.0");
		
		prm.declare_entry ("VelocityY value on the inlet boundary", "0.0");
		prm.declare_entry ("VelocityY value on the wall boundary", "0.0");
		prm.declare_entry ("VelocityY value on the cylinder boundary", "0.0");
		
		prm.declare_entry ("Pressure value on the outlet boundary", "0.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Particles");
	{
		prm.declare_entry ("Number of particles in the x direction", "0");
		prm.declare_entry ("Number of particles in the y direction", "0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("The criterion of record in the file");
	{
		prm.declare_entry ("Number of data to be recorded", "0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("The file with mesh");
	{
		prm.declare_entry ("Name of the file", "N");
	}
	prm.leave_subsection();
}

void TurekBenchmark::get_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		mu = prm.get_double ("Dynamic viscosity");
		rho = prm.get_double ("Density");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Time parameters");
	{
		time = prm.get_double("Initial time");
		time_step = prm.get_double("Time step");
		timestep_number = prm.get_integer("Number of time step");
		final_time_ = prm.get_double("Final Time");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Solver parameters");
	{
		accuracy_ = prm.get_double("Accuracy");
		num_of_iter_ = prm.get_integer("Number of iterations");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Boundary Values");
	{
		//neum_val_outlet_ = prm.get_double ("Neuman value on the outlet boundary");
		velX_inlet_ = prm.get_double ("VelocityX value on the inlet boundary");
		velX_wall_ = prm.get_double ("VelocityX value on the wall boundary");
		velX_cyl_ = prm.get_double ("VelocityX value on the cylinder boundary");
		
		velY_inlet_ = prm.get_double ("VelocityY value on the inlet boundary");
		velY_wall_ = prm.get_double ("VelocityY value on the wall boundary");
		velY_cyl_ = prm.get_double ("VelocityY value on the cylinder boundary");
		
		press_outlet_ = prm.get_double ("Pressure value on the outlet boundary");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Particles");
	{
		num_of_particles_x_ = prm.get_integer("Number of particles in the x direction");
		num_of_particles_y_ = prm.get_integer("Number of particles in the y direction");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("The criterion of record in the file");
	{
		num_of_data_ = prm.get_integer("Number of data to be recorded");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("The file with mesh");
	{
		mesh_file_ = prm.get("Name of the file");
	}
	prm.leave_subsection();	
}

/*!
 * \brief Построение сетки
 * 
 * Используется объект tria
 */
void TurekBenchmark::build_grid ()
{
	TimerOutput::Scope timer_section(*timer, "Mesh construction"); 
	
	GridIn<2> gridin;
	gridin.attach_triangulation(tria);
	std::ifstream f(mesh_file_);
	gridin.read_unv(f);
	f.close();
	
	pcout << "The mesh contains " << tria.n_active_cells() << " cells" << std::endl;
}

void TurekBenchmark::setup_system()
{
	TimerOutput::Scope timer_section(*timer, "System setup");

	dof_handlerV.distribute_dofs (feV);
	pcout << "Number of degrees of freedom V: " << dof_handlerV.n_dofs() << " * 2 = " << 2 * dof_handlerV.n_dofs() << std::endl;
			  
	locally_owned_dofsV = dof_handlerV.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handlerV, locally_relevant_dofsV);

	//Vx
    locally_relevant_solutionVx.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);
    locally_relevant_old_solutionVx.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);
    locally_relevant_predictionVx.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);
    
    system_rVx.reinit (locally_owned_dofsV, mpi_communicator);
    
    constraintsVx.clear ();
    constraintsVx.reinit (locally_relevant_dofsV);
    DoFTools::make_hanging_node_constraints(dof_handlerV, constraintsVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, parabolicBC(), constraintsVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, ConstantFunction<2>(0.0), constraintsVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, ConstantFunction<2>(0.0), constraintsVx);
    constraintsVx.close ();
    
    constraintsPredVx.clear ();
    constraintsPredVx.reinit (locally_relevant_dofsV);
    DoFTools::make_hanging_node_constraints(dof_handlerV, constraintsPredVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, ConstantFunction<2>(0.0), constraintsPredVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, ConstantFunction<2>(0.0), constraintsPredVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, ConstantFunction<2>(0.0), constraintsPredVx);
    constraintsPredVx.close ();
    
    DynamicSparsityPattern dspVx(locally_relevant_dofsV);
    DoFTools::make_sparsity_pattern (dof_handlerV, dspVx, constraintsVx, false);
    SparsityTools::distribute_sparsity_pattern (dspVx, dof_handlerV.locally_owned_dofs(), mpi_communicator, locally_relevant_dofsV);
    system_mVx.reinit (locally_owned_dofsV, locally_owned_dofsV, dspVx, mpi_communicator);
    
    //Vy
    locally_relevant_solutionVy.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);
    locally_relevant_old_solutionVy.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);
    locally_relevant_predictionVy.reinit (locally_owned_dofsV, locally_relevant_dofsV, mpi_communicator);

	system_rVy.reinit (locally_owned_dofsV, mpi_communicator);
    
    constraintsVy.clear ();
    constraintsVy.reinit (locally_relevant_dofsV);
    DoFTools::make_hanging_node_constraints(dof_handlerV, constraintsVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, ConstantFunction<2>(0.0), constraintsVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, ConstantFunction<2>(0.0), constraintsVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, ConstantFunction<2>(0.0), constraintsVy);
    constraintsVy.close ();
    
    constraintsPredVy.clear ();
    constraintsPredVy.reinit (locally_relevant_dofsV);
    DoFTools::make_hanging_node_constraints(dof_handlerV, constraintsPredVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, ConstantFunction<2>(0.0), constraintsPredVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, ConstantFunction<2>(0.0), constraintsPredVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, ConstantFunction<2>(0.0), constraintsPredVy);
    constraintsPredVy.close ();
    
    DynamicSparsityPattern dspVy(locally_relevant_dofsV);
    DoFTools::make_sparsity_pattern (dof_handlerV, dspVy, constraintsVy, false);
    SparsityTools::distribute_sparsity_pattern (dspVy, dof_handlerV.locally_owned_dofs(), mpi_communicator, locally_relevant_dofsV);
    system_mVy.reinit (locally_owned_dofsV, locally_owned_dofsV, dspVy, mpi_communicator);
    
    //P
	dof_handlerP.distribute_dofs (feP);
    pcout << "Number of degrees of freedom P: " << dof_handlerP.n_dofs() << std::endl;
    
    locally_owned_dofsP = dof_handlerP.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handlerP, locally_relevant_dofsP);
    locally_relevant_solutionP.reinit (locally_owned_dofsP, locally_relevant_dofsP, mpi_communicator);
    locally_relevant_old_solutionP.reinit (locally_owned_dofsP, locally_relevant_dofsP, mpi_communicator);
    
    system_rP.reinit (locally_owned_dofsP, mpi_communicator);
    
    constraintsP.clear ();
    constraintsP.reinit (locally_relevant_dofsP);
    DoFTools::make_hanging_node_constraints(dof_handlerP, constraintsP);
    VectorTools::interpolate_boundary_values (dof_handlerP, 2, ConstantFunction<2>(0.0), constraintsP);
    constraintsP.close ();
    
    DynamicSparsityPattern dspP(locally_relevant_dofsP);
    DoFTools::make_sparsity_pattern (dof_handlerP, dspP, constraintsP, false);
    SparsityTools::distribute_sparsity_pattern (dspP, dof_handlerP.locally_owned_dofs(), mpi_communicator, locally_relevant_dofsP);
    system_mP.reinit (locally_owned_dofsP, locally_owned_dofsP, dspP, mpi_communicator);
    
    //determine the DoF numbers for points for pressure difference measurement
	Point<2> xa(0.15, 0.2), xe(0.25, 0.2);
	DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
	DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();

	xaDoF = -100;
	xeDoF = -100;
	
	std::set<int> usedDoFs;

	for (; cell != endc; ++cell)
		if(cell->is_locally_owned()){
			if(xaDoF != -100 && xeDoF != -100) break;
	
			for(unsigned int i = 0; i < 4; i++){
				if(cell->vertex(i).distance(xa) < 1e-3)	xaDoF = cell->vertex_dof_index(i, 0);
				else if(cell->vertex(i).distance(xe) < 1e-3) xeDoF = cell->vertex_dof_index(i, 0);
			
				if(xaDoF != -100 && xeDoF != -100) break;
			}
			
			for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number){
				if(usedDoFs.count(cell->face(face_number)->vertex_dof_index(0,0)) || usedDoFs.count(cell->face(face_number)->vertex_dof_index(1,0))) continue;
				
				if (std::fabs(cell->face(face_number)->center()[0] - 0.08) < 1e-5) sectionFaces1.push_back(std::make_pair(cell, face_number));
				else if (std::fabs(cell->face(face_number)->center()[0] - 0.4) < 1e-5) sectionFaces2.push_back(std::make_pair(cell, face_number));
				else if (std::fabs(cell->face(face_number)->center()[0] - 1.00) < 1e-5) sectionFaces3.push_back(std::make_pair(cell, face_number));
				
				usedDoFs.insert(cell->face(face_number)->vertex_dof_index(0,0));
				usedDoFs.insert(cell->face(face_number)->vertex_dof_index(1,0));
			}
		}

	//std::cout << "Process " << this_mpi_process << " has " << sectionFaces1.size() << " DoFs" << std::endl;
}

void TurekBenchmark::fem_step()
{
	TimerOutput::Scope timer_section(*timer, "FEM Step");
	
	locally_relevant_old_solutionVx = locally_relevant_solutionVx;
	locally_relevant_old_solutionVy = locally_relevant_solutionVy;
	locally_relevant_old_solutionP = locally_relevant_solutionP;
	
	double aux, weight;
	unsigned int jDoFindex;
	
	TrilinosWrappers::MPI::Vector innerVx (locally_owned_dofsV, mpi_communicator), innerVy (locally_owned_dofsV, mpi_communicator);
	
	for(int nOuterCorr = 0; nOuterCorr < 3; ++nOuterCorr){
		innerVx = locally_relevant_solutionVx;
		innerVy = locally_relevant_solutionVy;
	
	//Vx
	system_mVx = 0.0;
	system_rVx = 0.0;
	system_mVy = 0.0;
	system_rVy = 0.0;

	TrilinosWrappers::MPI::Vector pressureField = locally_relevant_solutionP;
#ifdef SCHEMEB
	pressureField -= locally_relevant_old_solutionP;
#endif
	Functions::FEFieldFunction<2, DoFHandler<2>, TrilinosWrappers::MPI::Vector> pressureFieldFunction(dof_handlerP, pressureField);
	velocityPredictionBC velBC(time_step, rho, pressureFieldFunction);
	
	constraintsPredVx.clear ();
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, velBC, constraintsPredVx);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, velBC, constraintsPredVx);
    
    velBC.updateBcType(1);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, velBC, constraintsPredVx);
    constraintsPredVx.close ();
    
    velBC.updateBcType(0);
    velBC.updateIndex(1);

	constraintsPredVy.clear ();
    VectorTools::interpolate_boundary_values (dof_handlerV, 3, velBC, constraintsPredVy);
    VectorTools::interpolate_boundary_values (dof_handlerV, 4, velBC, constraintsPredVy);
    
    velBC.updateBcType(1);
    VectorTools::interpolate_boundary_values (dof_handlerV, 1, velBC, constraintsPredVy);
    constraintsPredVy.close ();
	
	{
		for(const auto &cell : dof_handlerV.active_cell_iterators())
			if(cell->is_locally_owned()){
				feV_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVx = 0.0;
				local_rhsVx = 0.0;
				local_matrixVy = 0.0;
				local_rhsVy = 0.0;
			
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					weight = feV_values.JxW (q_index);
					
					for (unsigned int i=0; i<dofs_per_cellV; ++i) {
						const Tensor<0,2> Ni_vel = feV_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feV_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellV; ++j) {
							jDoFindex = cell->vertex_dof_index(j,0);
							
							const Tensor<0,2> Nj_vel = feV_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feV_values.shape_grad (j,q_index);
#ifdef SCHEMEB
							const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);
#endif
							aux = rho * Ni_vel * Nj_vel * weight;
							local_matrixVx(i,j) += aux;
							local_matrixVy(i,j) += aux;
							
							local_rhsVx(i) += aux * locally_relevant_old_solutionVx(jDoFindex);
							local_rhsVy(i) += aux * locally_relevant_old_solutionVy(jDoFindex); 
							
							aux = mu * time_step * weight;
							//implicit account for tau_ij
							local_matrixVx(i,j) += aux * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]);
							local_matrixVy(i,j) += aux * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]);
														
							//explicit account for tau_ij
							local_rhsVx(i) -= aux * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * innerVy(jDoFindex);
							local_rhsVy(i) -= aux * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * innerVx(jDoFindex);
#ifdef SCHEMEB
							local_rhsVx(i) -= time_step * Ni_vel * Nj_p_grad[0] * locally_relevant_old_solutionP(jDoFindex) * weight;
							local_rhsVy(i) -= time_step * Ni_vel * Nj_p_grad[1] * locally_relevant_old_solutionP(jDoFindex) * weight;
#endif
						}//j
					}//i
				}//q_index
				
				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2)){
						feV_face_values.reinit (cell, face_number);
						
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							/*double duxdx = 0.0;
							double duydy = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVx; ++i){
								duxdx += feVx_face_values.shape_grad(i,q_point)[0] * old_solutionVx(cell->vertex_dof_index(i,0));
								duydy += feVx_face_values.shape_grad(i,q_point)[1] * old_solutionVy(cell->vertex_dof_index(i,0));
							}
						
							for (unsigned int i=0; i<dofs_per_cellVx; ++i)
								local_rhsVx(i) += mu * time_step * feVx_face_values.shape_value(i,q_point) * (4.0 / 3.0 * duxdx - 2.0 / 3.0 * duydy) *
									feVx_face_values.normal_vector(q_point)[0] * feVx_face_values.JxW(q_point);*/
							
							/*double duxdy = 0.0;
							double duydx = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVy; ++i){
								duxdy += feVy_face_values.shape_grad(i,q_point)[1] * old_solutionVx(cell->vertex_dof_index(i,0));
								duydx += feVy_face_values.shape_grad(i,q_point)[0] * old_solutionVy(cell->vertex_dof_index(i,0));
							}
						
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * (duxdy + duydx) *
									feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);*/

							for (unsigned int i=0; i<dofs_per_cellV; ++i)
								for (unsigned int j=0; j<dofs_per_cellV; ++j){
									local_matrixVx(i,j) -= mu * time_step * feV_face_values.shape_value(i,q_point) *
										4.0 / 3.0 * feV_face_values.shape_grad(j,q_point)[0] *	feV_face_values.normal_vector(q_point)[0] * feV_face_values.JxW(q_point);
									local_rhsVx(i) += mu * time_step * feV_face_values.shape_value(i,q_point) *
										(//4.0 / 3.0 * feVx_face_values.shape_grad(j,q_point)[0] * old_solutionVx(cell->vertex_dof_index(j,0))
										- 2.0 / 3.0 * feV_face_values.shape_grad(j,q_point)[1] * innerVy(cell->vertex_dof_index(j,0))) *
											feV_face_values.normal_vector(q_point)[0] * feV_face_values.JxW(q_point);
											
									local_matrixVy(i,j) -= mu * time_step * feV_face_values.shape_value(i,q_point) *
										feV_face_values.shape_grad(j,q_point)[0] *	feV_face_values.normal_vector(q_point)[0] * feV_face_values.JxW(q_point);
									local_rhsVy(i) += mu * time_step * feV_face_values.shape_value(i,q_point) *
										(//feVy_face_values.shape_grad(j,q_point)[0] * old_solutionVy(cell->vertex_dof_index(j,0)) +
										feV_face_values.shape_grad(j,q_point)[1] * innerVx(cell->vertex_dof_index(j,0))) *
											feV_face_values.normal_vector(q_point)[0] * feV_face_values.JxW(q_point);
								}
						}
					}		  
		  
				cell->get_dof_indices (local_dof_indicesV);
				constraintsPredVx.distribute_local_to_global (local_matrixVx, local_rhsVx, local_dof_indicesV, system_mVx, system_rVx);
				constraintsPredVy.distribute_local_to_global (local_matrixVy, local_rhsVy, local_dof_indicesV, system_mVy, system_rVy);
			}//cell

		system_mVx.compress (VectorOperation::add);
		system_rVx.compress (VectorOperation::add);
		system_mVy.compress (VectorOperation::add);
		system_rVy.compress (VectorOperation::add);
	}//V prediction

	solveVx ();
	solveVy ();

	//pressure equation
	system_mP = 0.0;
	system_rP = 0.0;
		
	for(const auto &cell : dof_handlerP.active_cell_iterators())
		if(cell->is_locally_owned()){
		feV_values.reinit (cell);
		feP_values.reinit (cell);
		local_matrixP = 0.0;
		local_rhsP = 0.0;
					
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index){
			weight = feV_values.JxW (q_index);
			
			for (unsigned int i=0; i<dofs_per_cellP; ++i) {
				const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);

				for (unsigned int j=0; j<dofs_per_cellP; ++j) {
					jDoFindex = cell->vertex_dof_index(j,0);
					
					const Tensor<0,2> Nj_vel = feV_values.shape_value (j,q_index);
					const Tensor<1,2> Njdx_pres = feP_values.shape_grad (j,q_index);

					aux = Nidx_pres * Njdx_pres * weight;
					local_matrixP(i,j) += aux;

					local_rhsP(i) += rho / time_step * (locally_relevant_predictionVx(jDoFindex) * Nidx_pres[0] + 
									locally_relevant_predictionVy(jDoFindex) * Nidx_pres[1]) * Nj_vel * weight;
#ifdef SCHEMEB
                    local_rhsP(i) += aux * locally_relevant_old_solutionP(jDoFindex);
#endif
				}//j
			}//i
		}

		for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2)){//inlet + outlet
				feV_face_values.reinit (cell, face_number);
				feP_face_values.reinit (cell, face_number);

				for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
					double Vx_q_point_value = 0.0;
					for (unsigned int i=0; i<dofs_per_cellP; ++i)
						Vx_q_point_value += feV_face_values.shape_value(i,q_point) * locally_relevant_predictionVx(cell->vertex_dof_index(i,0));

					for (unsigned int i=0; i<dofs_per_cellP; ++i)
						local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
										feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
					/*for (unsigned int i=0; i<dofs_per_cellP; ++i)
						for (unsigned int j=0; j<dofs_per_cellP; ++j)
							local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * feP_face_values.shape_value(j,q_point) * locally_relevant_predictionVx(cell->vertex_dof_index(j,0)) *
										feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);*/
				}
			}

		cell->get_dof_indices (local_dof_indicesP);
		constraintsP.distribute_local_to_global (local_matrixP, local_rhsP, local_dof_indicesP, system_mP, system_rP);
	}//cell

	system_mP.compress (VectorOperation::add);
	system_rP.compress (VectorOperation::add);
					
	solveP ();
	
	//Vx correction
	{
		system_mVx = 0.0;
		system_rVx = 0.0;
		system_mVy = 0.0;
		system_rVy = 0.0;
				
		for(const auto &cell : dof_handlerV.active_cell_iterators())
		    if(cell->is_locally_owned()) {
			feV_values.reinit (cell);
			feP_values.reinit (cell);
			local_matrixVx = 0.0;
			local_rhsVx = 0.0;
			local_matrixVy = 0.0;
			local_rhsVy = 0.0;
		
			for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
				weight = feV_values.JxW (q_index);
				
				for (unsigned int i=0; i<dofs_per_cellV; ++i) {
					const Tensor<0,2> Ni_vel = feV_values.shape_value (i,q_index);
					
					for (unsigned int j=0; j<dofs_per_cellV; ++j) {
						jDoFindex = cell->vertex_dof_index(j,0);
						
						const Tensor<0,2> Nj_vel = feV_values.shape_value (j,q_index);
						const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

						aux = rho * Ni_vel * Nj_vel * weight;
						local_matrixVx(i,j) += aux;
						local_matrixVy(i,j) += aux;
						local_rhsVx(i) += aux * locally_relevant_predictionVx(jDoFindex);
						local_rhsVy(i) += aux * locally_relevant_predictionVy(jDoFindex);
#ifndef SCHEMEB
                        local_rhsVx(i) -= time_step * Ni_vel * Nj_p_grad[0] * locally_relevant_solutionP(jDoFindex) * weight;
                        local_rhsVy(i) -= time_step * Ni_vel * Nj_p_grad[1] * locally_relevant_solutionP(jDoFindex) * weight;
#else
						local_rhsVx(i) -= time_step * Ni_vel * Nj_p_grad[0] * (locally_relevant_solutionP(jDoFindex) - locally_relevant_old_solutionP(jDoFindex)) * weight;
						local_rhsVy(i) -= time_step * Ni_vel * Nj_p_grad[1] * (locally_relevant_solutionP(jDoFindex) - locally_relevant_old_solutionP(jDoFindex)) * weight;
#endif
					}//j
				}//i
			}//q_index
      
			cell->get_dof_indices (local_dof_indicesV);
			constraintsVx.distribute_local_to_global (local_matrixVx, local_rhsVx, local_dof_indicesV, system_mVx, system_rVx);
			constraintsVy.distribute_local_to_global (local_matrixVy, local_rhsVy, local_dof_indicesV, system_mVy, system_rVy);
		}//cell
				
		system_mVx.compress (VectorOperation::add);
		system_rVx.compress (VectorOperation::add);
		system_mVy.compress (VectorOperation::add);
		system_rVy.compress (VectorOperation::add);
	}//Velocity correction
		
	solveVx (true);
	solveVy (true);
	}
}

/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void TurekBenchmark::solveVx(bool correction)
{
	TrilinosWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofsV, mpi_communicator);
	
	SolverControl solver_control (num_of_iter_, accuracy_);
        TrilinosWrappers::SolverGMRES solver (solver_control);
        TrilinosWrappers::PreconditionJacobi preconditioner;
	
	preconditioner.initialize(system_mVx);
	solver.solve (system_mVx, completely_distributed_solution, system_rVx, preconditioner);	
	
	if(solver_control.last_check() == SolverControl::success)
        pcout << "Solver for Vx converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
    else pcout << "Solver for Vx failed to converge" << std::endl;
        
    if (correction){
		constraintsVx.distribute (completely_distributed_solution);
		locally_relevant_solutionVx = completely_distributed_solution;		
	} else {
		constraintsPredVx.distribute (completely_distributed_solution);
		locally_relevant_predictionVx = completely_distributed_solution;		
	}
}

void TurekBenchmark::solveVy(bool correction)
{
	TrilinosWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofsV, mpi_communicator);
	
	SolverControl solver_control (num_of_iter_, accuracy_);
    TrilinosWrappers::SolverGMRES solver (solver_control);
    TrilinosWrappers::PreconditionJacobi preconditioner;
	
	preconditioner.initialize(system_mVy);
	solver.solve (system_mVy, completely_distributed_solution, system_rVy, preconditioner);	
	
	if(solver_control.last_check() == SolverControl::success)
        pcout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
    else pcout << "Solver for Vy failed to converge" << std::endl;
    
    if (correction){
		constraintsVy.distribute (completely_distributed_solution);
		locally_relevant_solutionVy = completely_distributed_solution;		
	} else {
		constraintsPredVy.distribute (completely_distributed_solution);
		locally_relevant_predictionVy = completely_distributed_solution;		
	}
}

void TurekBenchmark::solveP()
{
	TrilinosWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofsP, mpi_communicator);
    
    SolverControl solver_control (num_of_iter_, accuracy_);
    //LinearAlgebraTrilinos::SolverCG solver (solver_control);
    TrilinosWrappers::SolverGMRES solver (solver_control);
    //LinearAlgebraTrilinos::MPI::PreconditionSSOR preconditioner;
    TrilinosWrappers::PreconditionAMG preconditioner;

    preconditioner.initialize(system_mP);
    
	solver.solve (system_mP, completely_distributed_solution, system_rP, preconditioner);		
    
    if(solver_control.last_check() == SolverControl::success)
        pcout << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
    else pcout << "Solver for P failed to converge" << std::endl;
    
	constraintsP.distribute (completely_distributed_solution);
	locally_relevant_solutionP = completely_distributed_solution;
}

/*!
 * \brief Вывод результатов в формате VTK
 */
void TurekBenchmark::output_results(bool predictionCorrection) 
{
	TimerOutput::Scope timer_section(*timer, "Results output");
	
	DataOut<2> data_out;

	data_out.attach_dof_handler (dof_handlerV);
	data_out.add_data_vector (locally_relevant_solutionVx, "Vx");
	data_out.add_data_vector (locally_relevant_solutionVy, "Vy");
	data_out.add_data_vector (locally_relevant_solutionP, "P");
	
	if(predictionCorrection){
		data_out.add_data_vector (locally_relevant_predictionVx, "predVx");
		data_out.add_data_vector (locally_relevant_predictionVy, "predVy");
		//data_out.add_data_vector (correctionVx, "corVx");
		//data_out.add_data_vector (correctionVy, "corVy");
	}

    Vector<float> subdomain (tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector (dof_handlerV, subdomain, "subdomain");
	
	data_out.build_patches ();

	const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(this_mpi_process, 3) + ".vtu";
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);
    
    if (this_mpi_process==0) {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
			filenames.push_back ("solution-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(i, 3) + ".vtu");

		std::ofstream master_output (("solution-" + Utilities::int_to_string (timestep_number, 2) + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
	}//if
	
	//MPI_Barrier(mpi_communicator);
	//return;
	
	//вывод частиц
	const std::string filename2 =  "particles-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(this_mpi_process, 3) + ".vtu";
	std::ofstream output2 (filename2.c_str());
	
	//header
	output2 << "<?xml version=\"1.0\" ?> " << std::endl;
	output2 << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
	output2 << "  <UnstructuredGrid>" << std::endl;
	output2 << "    <Piece NumberOfPoints=\"" << particle_handler.n_global_particles() <<  "\" NumberOfCells=\"" << particle_handler.n_global_particles() << "\">" << std::endl;
	
	//точки
	output2 << "      <Points>" << std::endl;
	output2 << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
		output2 << "          " << (*particleIndex).second->get_location() << " 0.0" << std::endl;

	output2 << "        </DataArray>" << std::endl;
	output2 << "      </Points>" << std::endl;

	//"ячейки"
	output2 << "      <Cells>" << std::endl;
	output2 << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "          " << i << std::endl; 
	output2 << "        </DataArray>" << std::endl;

	output2 << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << std::endl;
	output2 << "        ";
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "  " << i + 1; 
	output2 << std::endl;
	output2 << "        </DataArray>" << std::endl;

	output2 << "        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">" << std::endl;
	output2 << "        ";
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "  " << 1;
	output2 << std::endl;
	output2 << "        </DataArray>" << std::endl;
	output2 << "      </Cells>" << std::endl;

	//данные в частицах
	output2 << "      <PointData Scalars=\"scalars\">" << std::endl;
	
	//скорость
	output2 << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
		output2 << "          " << (*particleIndex).second->get_velocity_component(0) << " " << (*particleIndex).second->get_velocity_component(1) << " 0.0" << std::endl;
	output2 << "        </DataArray>" << std::endl;
	output2 << "      </PointData>" << std::endl;

	//номер подобласти
	output2 << "        <DataArray type=\"Float32\" Name=\"subdomain\" Format=\"ascii\">" << std::endl;
	output2 << "        ";
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "  " << tria.locally_owned_subdomain();
	output2 << std::endl;
	output2 << "        </DataArray>" << std::endl;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
	

	//footer
	output2 << "    </Piece>" << std::endl;
	output2 << "  </UnstructuredGrid>" << std::endl;
	output2 << "</VTKFile>" << std::endl;
	
	if (this_mpi_process==0) {
        std::ofstream master_output (("particles-" + Utilities::int_to_string (timestep_number, 2) + ".pvtu").c_str());
        
        master_output << "<?xml version=\"1.0\" ?> " << std::endl;
		master_output << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
		master_output << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
		master_output << "    <PPointData Scalars=\"scalars\">" << std::endl;
		master_output << "      <PDataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\"/>" << std::endl;
		master_output << "    </PPointData>" << std::endl;
		master_output << "    <PPoints>" << std::endl;
        master_output << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
		master_output << "    </PPoints>" << std::endl;
        
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
			master_output << "    <Piece Source=\"particles-" << Utilities::int_to_string (timestep_number, 2) << "." << Utilities::int_to_string(i, 3) << ".vtu\"/>";
			
		master_output << "  </PUnstructuredGrid>" << std::endl;
		master_output << "</VTKFile>" << std::endl;
	}//if
	
	return;
	
	//const std::string filename2 =  "particles-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(this_mpi_process, 3) + ".vtk";
	//std::ofstream output2 (filename2.c_str());
	output2 << "# vtk DataFile Version 3.0" << std::endl;
	output2 << "Unstructured Grid Example" << std::endl;
	output2 << "ASCII" << std::endl;
	output2 << std::endl;
	output2 << "DATASET UNSTRUCTURED_GRID" << std::endl;
	output2 << "POINTS " << particle_handler.n_global_particles() << " float" << std::endl;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
		output2 << (*particleIndex).second->get_location() << " 0" << std::endl;
	
	output2 << std::endl;
	
	output2 << "CELLS " << particle_handler.n_global_particles() << " " << 2 * particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "1 " << i << std::endl; 
	
	output2 << std::endl;
	
	output2 << "CELL_TYPES " << particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i) output2 << "1 "; 
	output2 << std::endl;
	
	output2 << std::endl;
	
	output2 << "POINT_DATA " << particle_handler.n_global_particles() << std::endl;
	output2 << "VECTORS velocity float" << std::endl;
	for(auto particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
		output2 << (*particleIndex).second->get_velocity_component(0) << " " << (*particleIndex).second->get_velocity_component(1) << " 0" << std::endl;
}

void TurekBenchmark::check_balance(std::ofstream *out)
{
	TimerOutput::Scope timer_section(*timer, "Balance laws check");
	
	double velocityInt1(0.0), velocityInt2(0.0), velocityInt3(0.0);
	Tensor<1,2> point_valueV;

	for(std::vector<std::pair<DoFHandler<2>::cell_iterator, int>>::iterator it = sectionFaces1.begin(); it != sectionFaces1.end(); ++it){
		auto cell = (*it).first;
		feV_face_values.reinit (cell, (*it).second);

		for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
			point_valueV = 0.0;

			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				point_valueV[0] += locally_relevant_solutionVx(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
				point_valueV[1] += locally_relevant_solutionVy(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
			}

			velocityInt1 += point_valueV * feV_face_values.normal_vector(q_point) * feV_face_values.JxW (q_point);
		}
	}
	
	for(std::vector<std::pair<DoFHandler<2>::cell_iterator, int>>::iterator it = sectionFaces2.begin(); it != sectionFaces2.end(); ++it){
		auto cell = (*it).first;
		feV_face_values.reinit (cell, (*it).second);

		for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
			point_valueV = 0.0;

			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				point_valueV[0] += locally_relevant_solutionVx(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
				point_valueV[1] += locally_relevant_solutionVy(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
			}

			velocityInt2 += point_valueV * feV_face_values.normal_vector(q_point) * feV_face_values.JxW (q_point);
		}
	}
	
	for(std::vector<std::pair<DoFHandler<2>::cell_iterator, int>>::iterator it = sectionFaces3.begin(); it != sectionFaces3.end(); ++it){
		auto cell = (*it).first;
		feV_face_values.reinit (cell, (*it).second);

		for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
			point_valueV = 0.0;

			for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
				point_valueV[0] += locally_relevant_solutionVx(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
				point_valueV[1] += locally_relevant_solutionVy(cell->vertex_dof_index(vertex,0)) * feV_face_values.shape_value(vertex, q_point);
			}

			velocityInt3 += point_valueV * feV_face_values.normal_vector(q_point) * feV_face_values.JxW (q_point);
		}
	}
	
	double pxV(0.0), pyV(0.0), pxS(0.0), pyS(0.0), Fx(0.0), Fy(0.0), FxVis(0.0), FyVis(0.0);
	double point_valueP, dVtdn, weight;
	
	for(const auto &cell : dof_handlerP.active_cell_iterators())
		if(cell->is_locally_owned()){
			feV_values.reinit (cell);
			
			for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
				weight = feV_values.JxW (q_index);
			
				for (unsigned int i=0; i<dofs_per_cellV; ++i) {
					const Tensor<0,2> Ni_vel = feV_values.shape_value (i,q_index);
					
					pxV += locally_relevant_solutionVx(cell->vertex_dof_index(i,0)) * Ni_vel * weight;
					pyV += locally_relevant_solutionVy(cell->vertex_dof_index(i,0)) * Ni_vel * weight;
				}
			}
			
			for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary()) {
					feP_face_values.reinit (cell, face_number);

					for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point) {
						weight = feP_face_values.JxW (q_point);
						double coeff = 1;
						//if(cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2) coeff = -1;
						
						//if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 3 || cell->face(face_number)->boundary_id() == 4)){
							point_valueP = 0.0;
							dVtdn = 0.0;
	
							for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_cell; ++vertex){
								point_valueP += locally_relevant_solutionP(cell->vertex_dof_index(vertex,0)) * feP_face_values.shape_value(vertex, q_point);
								dVtdn += (locally_relevant_solutionVx(cell->vertex_dof_index(vertex,0)) * feP_face_values.normal_vector(q_point)[1] - locally_relevant_solutionVy(cell->vertex_dof_index(vertex,0)) * feP_face_values.normal_vector(q_point)[0]) *
										(feP_face_values.shape_grad(vertex, q_point)[0] * feP_face_values.normal_vector(q_point)[0] + feP_face_values.shape_grad(vertex, q_point)[1] * feP_face_values.normal_vector(q_point)[1]);
							}//vertex

							FxVis += mu * dVtdn * feP_face_values.normal_vector(q_point)[1] * weight;
							Fx -= point_valueP * feP_face_values.normal_vector(q_point)[0] * weight * coeff;
							FyVis -= mu * dVtdn * feP_face_values.normal_vector(q_point)[0] * weight;
							Fy -= point_valueP * feP_face_values.normal_vector(q_point)[1] * weight * coeff;
						//} else {
							for (unsigned int i=0; i<dofs_per_cellV; ++i) {
								const Tensor<0,2> Ni_P = feP_face_values.shape_value (i,q_point);
					
								pxS += locally_relevant_solutionVx(cell->vertex_dof_index(i,0)) * locally_relevant_solutionVx(cell->vertex_dof_index(i,0)) * feP_face_values.normal_vector(q_point)[0] * Ni_P * Ni_P * weight;
								pyS += locally_relevant_solutionVy(cell->vertex_dof_index(i,0)) * locally_relevant_solutionVx(cell->vertex_dof_index(i,0)) * feP_face_values.normal_vector(q_point)[0] * Ni_P * Ni_P * weight;
							}
						//}
					}//q_index
				}//if
		}
	
	const double local_integrals[11] = { velocityInt1, velocityInt2, velocityInt3, pxV, pyV, pxS, pyS, Fx, Fy, FxVis, FyVis };
	double global_integrals[11];

	//std::cout << "Process no. " << this_mpi_process << ", int1=" << velocityInt1 << std::endl;

	Utilities::MPI::sum(local_integrals, mpi_communicator, global_integrals);
		
	pcout << "Integral 1 = " << global_integrals[0] << ", Integral 2 = " << global_integrals[1] << ", Integral 3 = " << global_integrals[2] << std::endl;
	if (this_mpi_process == 0)
		*out << global_integrals[3] << "," << global_integrals[4] << "," << global_integrals[5] << "," << global_integrals[6] << "," << global_integrals[7] << "," << global_integrals[8]
			<< "," << global_integrals[9] << "," << global_integrals[10] << std::endl;
}

/*!
 * \brief Основная процедура программы
 * 
 * Подготовительные операции, цикл по времени, вызов вывода результатов
 */
void TurekBenchmark::run()
{	
	timer = new TimerOutput(pcout, TimerOutput::summary, TimerOutput::wall_times);

	std::ofstream os;
	//std::ofstream particleInfo;

	if (this_mpi_process == 0){
		system("rm solution-*");
		system("rm particles-*");
		os.open("forces.csv");
		os << "t,Cx,Cy,P,Cx_nu,Cx_p,Cy_nu,Cy_p" << std::endl;
		//particleInfo.open("telemetry.dat");
		//particleInfo << "t,particle_count,pxV,pyV,pxS,pyS,Fx,Fy,FxVis,FyVis" << std::endl;
	}
	
	build_grid();
	setup_system();
	
	locally_relevant_solutionVx=0.0;
	locally_relevant_solutionVy=0.0;
	locally_relevant_solutionP=0.0;

	seed_particles({(unsigned int)(num_of_particles_x_), (unsigned int)(num_of_particles_y_)});
	
	particle_handler.initialize_maps();

	for (; time <= final_time_; time += time_step, ++timestep_number) {
		pcout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;

		correct_particles_velocities();
		move_particles();
		distribute_particle_velocities_to_grid();

		fem_step();

		if((timestep_number - 1) % num_of_data_ == 0) 
			output_results(false);

		calculate_loads(3, &os);
		
		//int particleCount = Utilities::MPI::sum(particle_handler.n_locally_owned_particles(), mpi_communicator);
		//particleInfo << time << "," << particleCount << ",";
		//check_balance(&particleInfo);

		timer->print_summary();
	}//time
	
	delete timer;
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
	
	ParameterHandler prm;
	TurekBenchmark TurekBenchmarkProblem;

	TurekBenchmarkProblem.declare_parameters (prm);
	prm.parse_input ("input_data.prm");	
	
	//prm.print_parameters (std::cout, ParameterHandler::Text);
	// get parameters into the program
	//std::cout << "\n\n" << "Reading parameters:" << std::endl;
	
	TurekBenchmarkProblem.get_parameters (prm);
	
	TurekBenchmarkProblem.run ();
  
	return 0;
}
