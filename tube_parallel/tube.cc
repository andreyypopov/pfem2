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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include "pfem2particle.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>


using namespace dealii;

class parabolicBC : public Function<2>
{
public:
	parabolicBC() : Function<2>() {}
	
	virtual double value(const Point<2> &p, const unsigned int component = 0) const;
	double ddy(const Point<2> &p) const;
};

double parabolicBC::value(const Point<2> &p, const unsigned int) const
{
	return 50.0/3.0*(0.09 - p[1]*p[1]);
}

double parabolicBC::ddy(const Point<2> &p) const
{
	return -100.0/3.0*p[1];
}

class tube : public pfem2Solver
{
public:
	tube();

	void build_grid ();
	void setup_system();
	void assemble_system();
	void solveVx(bool correction = false);
	void solveVy(bool correction = false);
	void solveP();
	void output_results(bool predictionCorrection = false);
	void run();
	
	QGauss<2>   quadrature_formula;
	QGauss<1>   face_quadrature_formula;
	
	FEValues<2> feVx_values, feVy_values, feP_values; 
						   
	FEFaceValues<2> feVy_face_values, feP_face_values;
						                                       
	const unsigned int   n_q_points, n_face_q_points;
	
	FullMatrix<double>   local_matrixVx, local_matrixVy, local_matrixP;
						 
	Vector<double>       local_rhsVx, local_rhsVy, local_rhsP;
	
	const double mu, rho;
	
	PETScWrappers::MPI::SparseMatrix sparsity_patternVx, sparsity_patternVy, sparsity_patternP;
	PETScWrappers::MPI::SparseMatrix system_mVx, system_mVy, system_mP,
								     system_mCorrVx, system_mCorrVy, systemCorrVx, systemCorrVy;
	PETScWrappers::MPI::Vector system_rVx, system_rVy, system_rP;
	
	std::vector<types::global_dof_index> local_dofs_per_process;
	
    unsigned int n_local_cells;
    
    ConstraintMatrix   constraintsCorrVx, constraintsCorrVy, constraintsP;
};

tube::tube()
	: pfem2Solver(),
	quadrature_formula(2),
	face_quadrature_formula(2),
	feVx_values (feVx, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVy_values (feVy, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feP_values (feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVy_face_values (feVy, face_quadrature_formula, update_values | update_quadrature_points | update_normal_vectors | update_JxW_values),
    feP_face_values (feP, face_quadrature_formula, update_values | update_quadrature_points  | update_normal_vectors | update_JxW_values),
	n_q_points (quadrature_formula.size()),
	n_face_q_points (face_quadrature_formula.size()),
	local_matrixVx (dofs_per_cellVx, dofs_per_cellVx),
	local_matrixVy (dofs_per_cellVy, dofs_per_cellVy),
	local_matrixP (dofs_per_cellP, dofs_per_cellP),
	local_rhsVx (dofs_per_cellVx),
	local_rhsVy (dofs_per_cellVy),
	local_rhsP (dofs_per_cellP),
	mu (0.003333333),
	rho (1.0)
    
{
	time = 0.0;
	time_step=0.001;
	timestep_number = 1;
}

/*!
 * \brief Построение сетки
 * 
 * Используется объект tria
 */
void tube::build_grid ()
{
  TimerOutput::Scope timer_section(*timer, "Mesh construction");
  
  GridIn<2> gridin;
  gridin.attach_triangulation(tria);
  std::ifstream f("mesh7.unv");
  gridin.read_unv(f);
  f.close();
  
  pcout << "Grid has " << tria.n_cells(tria.n_levels()-1) << " cells" << std::endl;
  
  return;
  
  GridOut grid_out;

  std::ofstream out ("tube.eps");
  grid_out.write_eps (tria, out);
  std::cout << "Grid written to EPS" << std::endl;
  
  std::ofstream out2 ("tube.vtk");
  grid_out.write_vtk (tria, out2);  
  std::cout << "Grid written to VTK" << std::endl;
}

void tube::setup_system()
{
	TimerOutput::Scope timer_section(*timer, "System setup");

	dof_handlerVx.distribute_dofs (feVx);
	pcout << "Number of degrees of freedom Vx: "
			  << dof_handlerVx.n_dofs()
			  << std::endl;
			  
	locally_owned_dofsVx = dof_handlerVx.locally_owned_dofs ();
	
    DoFTools::extract_locally_relevant_dofs (dof_handlerVx, locally_relevant_dofsVx);
                                             
    locally_relevant_solutionVx.reinit (locally_owned_dofsVx, locally_relevant_dofsVx, mpi_communicator);
    locally_relevant_old_solutionVx.reinit (locally_owned_dofsVx, locally_relevant_dofsVx, mpi_communicator);   
    locally_relevant_correctionVx.reinit (locally_owned_dofsVx, locally_relevant_dofsVx, mpi_communicator);   
    locally_relevant_predictionVx.reinit (locally_owned_dofsVx, locally_relevant_dofsVx, mpi_communicator);   
			  
	dof_handlerVy.distribute_dofs (feVy);
	pcout << "Number of degrees of freedom Vy: "
			  << dof_handlerVy.n_dofs()
			  << std::endl;
			  
	locally_owned_dofsVy = dof_handlerVy.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handlerVy,
                                             locally_relevant_dofsVy);
                                             
    locally_relevant_solutionVy.reinit (locally_owned_dofsVy, locally_relevant_dofsVy, mpi_communicator);
    locally_relevant_old_solutionVy.reinit (locally_owned_dofsVy, locally_relevant_dofsVy, mpi_communicator);
    locally_relevant_correctionVy.reinit (locally_owned_dofsVy, locally_relevant_dofsVy, mpi_communicator);
    locally_relevant_predictionVy.reinit (locally_owned_dofsVy, locally_relevant_dofsVy, mpi_communicator);
			  
	dof_handlerP.distribute_dofs (feP);
	pcout << "Number of degrees of freedom P: "
			  << dof_handlerP.n_dofs()
			  << std::endl;
			  
	locally_owned_dofsP = dof_handlerP.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handlerP,
                                             locally_relevant_dofsP);
                                             
    locally_relevant_solutionP.reinit (locally_owned_dofsP, locally_relevant_dofsP, mpi_communicator);
    locally_relevant_old_solutionP.reinit (locally_owned_dofsP, locally_relevant_dofsP, mpi_communicator);
                                             
    n_local_cells = GridTools::count_cells_with_subdomain_association (tria, tria.locally_owned_subdomain ());
    
	//Vx	 
	constraintsVx.clear ();
	
	constraintsVx.reinit (locally_relevant_dofsVx);
    DoFTools::make_hanging_node_constraints (dof_handlerVx, constraintsVx);
    
	VectorTools::interpolate_boundary_values (dof_handlerVx, 1, parabolicBC(), constraintsVx);
	VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<2>(0.0), constraintsVx);
	VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), constraintsVx);
	
	constraintsVx.close ();
                                       
	DynamicSparsityPattern dspVx(locally_relevant_dofsVx);
	DoFTools::make_sparsity_pattern (dof_handlerVx, dspVx, constraintsVx, false);
	SparsityTools::distribute_sparsity_pattern (dspVx,
                                                dof_handlerVx.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofsVx);
	
	system_mVx.reinit (locally_owned_dofsVx, locally_owned_dofsVx, dspVx, mpi_communicator);
    system_rVx.reinit (locally_owned_dofsVx, mpi_communicator);
    
    //Vx correction
    constraintsCorrVx.clear ();
    
	constraintsCorrVx.reinit (locally_relevant_dofsVx);
    DoFTools::make_hanging_node_constraints (dof_handlerVx, constraintsCorrVx);
    
	VectorTools::interpolate_boundary_values (dof_handlerVx, 1, ConstantFunction<2>(0.0), constraintsCorrVx);
	VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<2>(0.0), constraintsCorrVx);
	VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), constraintsCorrVx);
                                       
    constraintsCorrVx.close ();
    
    DynamicSparsityPattern dspCorrVx(locally_relevant_dofsVx);
	DoFTools::make_sparsity_pattern (dof_handlerVx, dspCorrVx, constraintsCorrVx, false);
	SparsityTools::distribute_sparsity_pattern (dspCorrVx,
                                                dof_handlerVx.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofsVx);
                                                
	system_mCorrVx.reinit (locally_owned_dofsVx, locally_owned_dofsVx, dspCorrVx, mpi_communicator);
    
    //Vy
    constraintsVy.clear ();
    
    constraintsVy.reinit (locally_relevant_dofsVy);
    DoFTools::make_hanging_node_constraints (dof_handlerVy, constraintsVy);
    
	VectorTools::interpolate_boundary_values (dof_handlerVy, 1, ConstantFunction<2>(0.0), constraintsVy);
	VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<2>(0.0), constraintsVy);
	VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), constraintsVy);
	
    constraintsVy.close ();
	 
	DynamicSparsityPattern dspVy(locally_relevant_dofsVy);
	DoFTools::make_sparsity_pattern (dof_handlerVy, dspVy, constraintsVy, false);
	SparsityTools::distribute_sparsity_pattern (dspVy,
                                                dof_handlerVy.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofsVy);
	
	system_mVy.reinit (locally_owned_dofsVy, locally_owned_dofsVy, dspVy, mpi_communicator);
    system_rVy.reinit (locally_owned_dofsVy, mpi_communicator);
    
    //Vy correction
    constraintsCorrVy.clear ();
    
    constraintsCorrVy.reinit (locally_relevant_dofsVy);
    DoFTools::make_hanging_node_constraints (dof_handlerVy, constraintsCorrVy);
    
	VectorTools::interpolate_boundary_values (dof_handlerVy, 1, ConstantFunction<2>(0.0), constraintsCorrVy);
	VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<2>(0.0), constraintsCorrVy);
	VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), constraintsCorrVy);
	
    constraintsVy.close ();
    
    DynamicSparsityPattern dspCorrVy(locally_relevant_dofsVy);
	DoFTools::make_sparsity_pattern (dof_handlerVy, dspCorrVy, constraintsCorrVy, false);
	SparsityTools::distribute_sparsity_pattern (dspCorrVy,
                                                dof_handlerVy.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofsVy);
    
	system_mCorrVy.reinit (locally_owned_dofsVy, locally_owned_dofsVy, dspCorrVy, mpi_communicator);
	
	//P
	constraintsP.clear ();
	
	constraintsP.reinit (locally_relevant_dofsP);
    DoFTools::make_hanging_node_constraints (dof_handlerP, constraintsP);
                                              
    VectorTools::interpolate_boundary_values (dof_handlerP, 4, ConstantFunction<2>(0.0), constraintsP);	
    												
    constraintsP.close ();
    
    DynamicSparsityPattern dspP(locally_relevant_dofsP);
	DoFTools::make_sparsity_pattern (dof_handlerP, dspP, constraintsP, false);
	SparsityTools::distribute_sparsity_pattern (dspP,
                                                dof_handlerP.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofsP);
	
	system_mP.reinit (locally_owned_dofsP, locally_owned_dofsP, dspP, mpi_communicator);
    system_rP.reinit (locally_owned_dofsP, mpi_communicator);
       
    //Vx
    system_mVx = 0.0;
    
    {
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
		int number = 0;
		
		for (; cell!=endc; ++cell,++number) {
			if (cell->is_locally_owned()) {
				feVx_values.reinit (cell);
				local_matrixVx = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVx_values.shape_grad (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVx_values.shape_grad (j,q_index);

							local_matrixVx(i,j) += rho * Ni_vel * Nj_vel * feVx_values.JxW(q_index);
							//implicit account for tau_ij
							local_matrixVx(i,j) += mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]) * feVx_values.JxW (q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVx);
				
				constraintsVx.distribute_local_to_global (local_matrixVx, local_dof_indicesVx, system_mVx);
			}//if
		}//cell
		
		system_mVx.compress (VectorOperation::add);
	}//Vx
	
	//Vy
	system_mVy = 0.0;
	
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();		
			
		for (; cell!=endc; ++cell) {
			if (cell->is_locally_owned()) {
				feVy_values.reinit (cell);
				local_matrixVy = 0.0;

				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVy_values.shape_grad (i,q_index);
	
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVy_values.shape_grad (j,q_index);

							local_matrixVy(i,j) += rho * Ni_vel * Nj_vel * feVy_values.JxW(q_index);
							//implicit account for tau_ij
							local_matrixVy(i,j) += mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]) * feVy_values.JxW (q_index);						
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVy);
				
				constraintsVy.distribute_local_to_global (local_matrixVy, local_dof_indicesVy, system_mVy);
			}//if
		}//cell
		
		system_mVy.compress (VectorOperation::add);
	}//Vy
	
	//P
	system_mP = 0.0;
	
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
		for (; cell!=endc; ++cell) {
			if (cell->is_locally_owned()) {
				local_matrixP = 0.0;
				feP_values.reinit (cell);
					
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellP; ++i) {
						const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellP; ++j) {
							const Tensor<1,2> Njdx_pres = feP_values.shape_grad (j,q_index);
						
							local_matrixP(i,j) += Nidx_pres * Njdx_pres * feP_values.JxW(q_index);											
						}//j
					}//i
				}//q_index

				cell->get_dof_indices (local_dof_indicesP);
				
				constraintsP.distribute_local_to_global (local_matrixP, local_dof_indicesP, system_mP);
			}//if
		}//cell
		
		system_mP.compress (VectorOperation::add);
	}//P
		
	//Vx correction
	system_mCorrVx = 0.0;
	
	{		
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
		for (; cell!=endc; ++cell) {
			if (cell->is_locally_owned()) {
				feVx_values.reinit (cell);
				local_matrixVx = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);

							local_matrixVx(i,j) += Ni_vel * Nj_vel * feVx_values.JxW(q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVx);

				constraintsCorrVx.distribute_local_to_global (local_matrixVx, local_dof_indicesVx, system_mCorrVx);
			}//if
		}//cell
		
		system_mCorrVx.compress (VectorOperation::add);
	}//Vx correction
	
	//Vy correction
	system_mCorrVy = 0.0;
	
	{	
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
		for (; cell!=endc; ++cell) {
			if (cell->is_locally_owned()) {
				feVy_values.reinit (cell);
				local_matrixVy = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);

							local_matrixVy(i,j) += Ni_vel * Nj_vel * feVy_values.JxW(q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVy);
				
				constraintsCorrVy.distribute_local_to_global (local_matrixVy, local_dof_indicesVy, system_mCorrVy);
			}//if
		}//cell
		
		system_mCorrVy.compress (VectorOperation::add);
	}//Vy correction
}

void tube::assemble_system()
{
	TimerOutput::Scope timer_section(*timer, "FEM step");
	
	locally_relevant_old_solutionVx = locally_relevant_solutionVx; 
	locally_relevant_old_solutionVy = locally_relevant_solutionVy; 
	locally_relevant_old_solutionP = locally_relevant_solutionP; 
			
	for(int nOuterCorr = 0; nOuterCorr < 1; ++nOuterCorr){
		system_rVx=0.0;
			
		//Vx
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
			int number = 0;
		
			for (; cell!=endc; ++cell,++number) {
				if (cell->is_locally_owned()) {
					feVx_values.reinit (cell);
					feVy_values.reinit (cell);
					feP_values.reinit (cell);
					local_rhsVx = 0.0;
					
					for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
						for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
							const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
							const Tensor<1,2> Ni_vel_grad = feVx_values.shape_grad (i,q_index);
					
							for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
								const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
								const Tensor<1,2> Nj_vel_grad = feVx_values.shape_grad (j,q_index);

								//explicit account for tau_ij
								//local_rhsVx(i) -= mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
								local_rhsVx(i) -= mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * locally_relevant_old_solutionVy(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						
								local_rhsVx(i) += rho * Nj_vel * Ni_vel * locally_relevant_old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							}//j
						}//i
					}//q_index
      
					cell->get_dof_indices (local_dof_indicesVx);
					
					constraintsVx.distribute_local_to_global (local_rhsVx, local_dof_indicesVx, system_rVx);						
				}//if
			}//cell
			
			system_rVx.compress (VectorOperation::add);
		}//Vx
			
		solveVx ();
	
		//Vy
		system_rVy=0.0;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
			
			parabolicBC boundaryCond;			
			
			for (; cell!=endc; ++cell) {
				if (cell->is_locally_owned()) {
					feVx_values.reinit (cell);
					feVy_values.reinit (cell);
					feP_values.reinit (cell);
					local_rhsVy = 0.0;

					for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
						for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
							const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
							const Tensor<1,2> Ni_vel_grad = feVy_values.shape_grad (i,q_index);
	
							for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
								const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);
								const Tensor<1,2> Nj_vel_grad = feVy_values.shape_grad (j,q_index);
												
								//explicit account for tau_ij
								//local_rhsVy(i) -= mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
								local_rhsVy(i) -= mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]) * locally_relevant_old_solutionVy(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);

								local_rhsVy(i) += rho * Nj_vel * Ni_vel * locally_relevant_old_solutionVy(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index); 
							}//j
						}//i
					}//q_index
			
					for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
						if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1)){//inlet
							feVy_face_values.reinit (cell, face_number);
					
							for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
								for (unsigned int i=0; i<dofs_per_cellVy; ++i)
									local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * boundaryCond.ddy(feVy_face_values.quadrature_point(q_point)) *
										feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
						} else if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 4)){//outlet
							feVy_face_values.reinit (cell, face_number);
					
							for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
								double duxdy = 0.0;
								for (unsigned int i=0; i<dofs_per_cellVy; ++i)
									duxdy += feVx_values.shape_grad(i,q_point)[1] * locally_relevant_old_solutionVx(cell->vertex_dof_index(i,0));
					
								for (unsigned int i=0; i<dofs_per_cellVy; ++i)
									local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * duxdy *
										feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
							}//q_point
						}//else if
      
					cell->get_dof_indices (local_dof_indicesVy);
					
					constraintsVy.distribute_local_to_global (local_rhsVy, local_dof_indicesVy, system_rVy);
				}//if
			}//cell
			
			system_rVy.compress (VectorOperation::add);
		}//Vy
									
		solveVy ();

		//P
		
		for (int n_cor=0; n_cor<1; ++n_cor){
			system_rP=0.0;
		
			parabolicBC boundaryCond;
		
			{
				DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
				for (; cell!=endc; ++cell) {
					if (cell->is_locally_owned()) {
						local_rhsP = 0.0;
						feVx_values.reinit (cell);
						feVy_values.reinit (cell);
						feP_values.reinit (cell);
					
						for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
							for (unsigned int i=0; i<dofs_per_cellP; ++i) {
								const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);
						
								for (unsigned int j=0; j<dofs_per_cellP; ++j) {
									const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
											
									local_rhsP(i) += rho / time_step * (locally_relevant_predictionVx(cell->vertex_dof_index(j,0)) * Nidx_pres[0] + 
												locally_relevant_predictionVy(cell->vertex_dof_index(j,0)) * Nidx_pres[1]) * Nj_vel * feP_values.JxW (q_index);
								}//j
							}//i
						}//q_index

						for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
							if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1)){//inlet
								feP_face_values.reinit (cell, face_number);
					
								for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
									for (unsigned int i=0; i<dofs_per_cellP; ++i)
										local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * boundaryCond.value(feP_face_values.quadrature_point(q_point)) *
											feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
							} else if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 4)){//outlet
								feP_face_values.reinit (cell, face_number);
					
								for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
									double Vx_q_point_value = 0.0;
									for (unsigned int i=0; i<dofs_per_cellP; ++i)
										Vx_q_point_value += feVx_values.shape_value(i,q_point) * locally_relevant_predictionVx(cell->vertex_dof_index(i,0));
						
									for (unsigned int i=0; i<dofs_per_cellP; ++i)
										local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
											feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
								}//q_point
							}//else if

						cell->get_dof_indices (local_dof_indicesP);
						
						constraintsP.distribute_local_to_global (local_rhsP, local_dof_indicesP, system_rP);
					}//if
				}//cell
				
				system_rP.compress (VectorOperation::add);
			}//P
		
			solveP ();
			
			//Vx correction
			{
				system_rVx = 0.0;
			
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
				for (; cell!=endc; ++cell) {
					if (cell->is_locally_owned()) {
						feVx_values.reinit (cell);
						feVy_values.reinit (cell);
						feP_values.reinit (cell);
						local_rhsVx = 0.0;
		
						for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
							for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
								const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
					
								for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
									const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

									local_rhsVx(i) -= time_step/rho * Ni_vel * Nj_p_grad[0] * locally_relevant_solutionP(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
								}//j
							}//i
						}//q_index
      
						cell->get_dof_indices (local_dof_indicesVx);
						
						constraintsCorrVx.distribute_local_to_global (local_rhsVx, local_dof_indicesVx, system_rVx);
					}//if
				}//cell	
				
				system_rVx.compress (VectorOperation::add);			
			}//Vx correction
    						
			solveVx (true);
		
			//Vy correction
			{
				system_rVy = 0.0;
			
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
				for (; cell!=endc; ++cell) {
					if (cell->is_locally_owned()) {
						feVx_values.reinit (cell);
						feVy_values.reinit (cell);
						feP_values.reinit (cell);
						local_rhsVy = 0.0;
		
						for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
							for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
								const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
					
								for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
									const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

									local_rhsVy(i) -= time_step/rho * Ni_vel * Nj_p_grad[1] * locally_relevant_solutionP(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
								}//j
							}//i
						}//q_index
      
						cell->get_dof_indices (local_dof_indicesVy);
											
						constraintsCorrVy.distribute_local_to_global (local_rhsVy, local_dof_indicesVy, system_rVy);
					}//if
				}//cell	
				
				system_rVy.compress (VectorOperation::add);		
			}//Vy
	    						
			solveVy (true);
		
			locally_relevant_solutionVx = locally_relevant_predictionVx;
			locally_relevant_solutionVx += locally_relevant_correctionVx;
			locally_relevant_solutionVy = locally_relevant_predictionVy;
			locally_relevant_solutionVy += locally_relevant_correctionVy;
			
			locally_relevant_old_solutionP = locally_relevant_solutionP;
		}//n_cor
	}//nOuterCorr
}

/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void tube::solveVx(bool correction)
{	
	PETScWrappers::MPI::Vector 
	distribute_correctionVx (locally_owned_dofsVx, mpi_communicator),
	distribute_predictionVx (locally_owned_dofsVx, mpi_communicator);

	SolverControl solver_control (10000, 1e-7);
	PETScWrappers::SolverBicgstab solver (solver_control, mpi_communicator);
	PETScWrappers::PreconditionJacobi preconditioner (system_mVx);
		
	if(correction) solver.solve (system_mCorrVx, distribute_correctionVx, system_rVx, preconditioner);
	else solver.solve (system_mVx, distribute_predictionVx, system_rVx, preconditioner);
                 
    if(solver_control.last_check() == SolverControl::success)
		pcout << "Solver for Vx converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else pcout << "Solver for Vx failed to converge" << std::endl;
	
	if (correction){
		constraintsCorrVx.distribute (distribute_correctionVx);
		locally_relevant_correctionVx = distribute_correctionVx;
	} 
	else {
		constraintsVx.distribute (distribute_predictionVx);
		locally_relevant_predictionVx = distribute_predictionVx;
	}
}

void tube::solveVy(bool correction)
{
	PETScWrappers::MPI::Vector 
	distribute_correctionVy (locally_owned_dofsVy, mpi_communicator),
	distribute_predictionVy (locally_owned_dofsVy, mpi_communicator);
	
	SolverControl solver_control (10000, 1e-7);
	PETScWrappers::SolverBicgstab solver (solver_control, mpi_communicator);
	PETScWrappers::PreconditionJacobi preconditioner (system_mVy);
	
	if(correction) solver.solve (system_mCorrVy, distribute_correctionVy, system_rVy, preconditioner);
	else solver.solve (system_mVy, distribute_predictionVy, system_rVy, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		pcout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else pcout << "Solver for Vy failed to converge" << std::endl;
	
	if (correction){
		constraintsCorrVy.distribute (distribute_correctionVy);
		locally_relevant_correctionVy = distribute_correctionVy;
	} 
	else {
		constraintsVy.distribute (distribute_predictionVy);
		locally_relevant_predictionVy = distribute_predictionVy;
	}
}

void tube::solveP()
{
	PETScWrappers::MPI::Vector 
	distribute_solutionP (locally_owned_dofsP, mpi_communicator);
	
	SolverControl solver_control (10000, 1e-7);
	PETScWrappers::SolverBicgstab solver (solver_control, mpi_communicator);
	PETScWrappers::PreconditionSOR preconditioner (system_mP);
	//PETScWrappers::SSOR preconditioner (system_mP);
	//PETScWrappers::PreconditionJacobi preconditioner (system_mP);
	
	solver.solve (system_mP, distribute_solutionP, system_rP, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		pcout << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else pcout << "Solver for P failed to converge" << std::endl;
	
	constraintsP.distribute (distribute_solutionP);
	locally_relevant_solutionP = distribute_solutionP;
}

/*!
 * \brief Вывод результатов в формате VTK
 */
void tube::output_results(bool predictionCorrection) 
{
	TimerOutput::Scope timer_section(*timer, "Results output");
	
	DataOut<2> data_out;

	data_out.attach_dof_handler (dof_handlerVx);
	data_out.add_data_vector (locally_relevant_solutionVx, "Vx");
	data_out.add_data_vector (locally_relevant_solutionVy, "Vy");
	data_out.add_data_vector (locally_relevant_solutionP, "P");
	
	Vector<float> subdomain (tria.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");
	
	if(predictionCorrection){
		data_out.add_data_vector (locally_relevant_predictionVx, "predVx");
		data_out.add_data_vector (locally_relevant_predictionVy, "predVy");
		data_out.add_data_vector (locally_relevant_correctionVx, "corVx");
		data_out.add_data_vector (locally_relevant_correctionVy, "corVy");
	}
	
	data_out.build_patches ();

	const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(this_mpi_process,3) + ".vtu";
	std::ofstream output (filename.c_str());
	data_out.write_vtu (output);
	
	if (this_mpi_process==0) {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<n_mpi_processes; ++i)
			filenames.push_back ("solution-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(i,3) + ".vtu");

		std::ofstream master_output (("solution-" + Utilities::int_to_string (timestep_number, 2) + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
	}//if

		//вывод частиц
		const std::string filename2 =  "particles-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(this_mpi_process,3) + ".vtk";
		std::ofstream output2 (filename2.c_str());
		output2 << "# vtk DataFile Version 3.0" << std::endl;
		output2 << "Unstructured Grid Example" << std::endl;
		output2 << "ASCII" << std::endl;
		output2 << std::endl;
		output2 << "DATASET UNSTRUCTURED_GRID" << std::endl;
		output2 << "POINTS " << particle_handler.n_locally_owned_particles() << " float" << std::endl;
		for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
											particleIndex != particle_handler.end(); ++particleIndex){
			output2 << particleIndex->get_location() << " 0" << std::endl;
		}
	
		output2 << std::endl;
	
		output2 << "CELLS " << particle_handler.n_locally_owned_particles() << " " << 2 * particle_handler.n_locally_owned_particles() << std::endl;
		for (unsigned int i=0; i< particle_handler.n_locally_owned_particles(); ++i){
			output2 << "1 " << i << std::endl; 
		}
	
		output2 << std::endl;
	
		output2 << "CELL_TYPES " << particle_handler.n_locally_owned_particles() << std::endl;
		for (unsigned int i=0; i< particle_handler.n_locally_owned_particles(); ++i){
			output2 << "1 "; 
		}	
		output2 << std::endl;
	
		output2 << std::endl;
	
		output2 << "POINT_DATA " << particle_handler.n_locally_owned_particles() << std::endl;
		output2 << "VECTORS velocity float" << std::endl;
		for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
												particleIndex != particle_handler.end(); ++particleIndex){
			output2 << particleIndex->get_properties()[0] << " " << particleIndex->get_properties()[1] << " 0" << std::endl;
		}
		
		output2.close();
}

/*!
 * \brief Основная процедура программы
 * 
 * Подготовительные операции, цикл по времени, вызов вывода результатов
 */
void tube::run()
{	
	timer = new TimerOutput(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times);
	
	build_grid();
	setup_system();

	seed_particles({2, 2});
	
	locally_relevant_solutionVx = 0.0;
	locally_relevant_solutionVy = 0.0;
	locally_relevant_solutionP = 0.0;
	
	//удаление старых файлов VTK (специфическая команда Linux!!!)
	if (this_mpi_process == 0){
		system("rm solution-*.*.vtu");
		system("rm solution-*.pvtu");
		system("rm particles-*.vtk");
//		std::ofstream os("force.csv");
	}
	//system("rm particles-*.*.vtu");

	for (; timestep_number<=13; time+=time_step, ++timestep_number) {
		pcout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
		
		correct_particles_velocities();
		move_particles();
		distribute_particle_velocities_to_grid();
		
		assemble_system();
		if((timestep_number - 1) % 1 == 0) output_results();
		
		//calculate_loads(3, &os);
		
		timer->print_summary();
	}//time
	
	//os.close();
	
	delete timer;
}

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	
  tube tubeproblem;
  tubeproblem.run ();
  
  return 0;
}
