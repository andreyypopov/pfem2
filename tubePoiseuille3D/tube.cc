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

class tube : public pfem2Solver
{
public:
	tube();

	void build_grid ();
	void setup_system();
	void assemble_system();
	void solveVx(bool correction = false);
	void solveVy(bool correction = false);
	void solveVz(bool correction = false);
	void solveP();
	void output_results(bool predictionCorrection = false);
	void run();
	
	SparsityPattern sparsity_patternVx, sparsity_patternVy, sparsity_patternVz, sparsity_patternP;
	SparseMatrix<double> system_mVx, system_mVy, system_mVz, system_mP;
	Vector<double> system_rVx, system_rVy, system_rVz, system_rP;
};

tube::tube()
	: pfem2Solver()
{
	time = 0.0;
	time_step=0.005;
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
  
  const Point<3> bottom_left = Point<3> (-3,-2,-2);
  const Point<3> top_right = Point<3> (10,2,2);

  std::vector< unsigned int > repetitions {20,6,6}; 

  GridGenerator::subdivided_hyper_rectangle(tria,repetitions,bottom_left,top_right, true);
  
  return;
  
  std::cout << "Grid has " << tria.n_cells(tria.n_levels()-1) << " cells" << std::endl;
  
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
    std::cout << "Number of degrees of freedom Vx: " << dof_handlerVx.n_dofs() << std::endl;
    
    dof_handlerVy.distribute_dofs (feVy);
    std::cout << "Number of degrees of freedom Vy: " << dof_handlerVy.n_dofs() << std::endl;

    dof_handlerVz.distribute_dofs (feVz);
    std::cout << "Number of degrees of freedom Vz: " << dof_handlerVz.n_dofs() << std::endl;
    
    dof_handlerP.distribute_dofs (feP);
    std::cout << "Number of degrees of freedom P: " << dof_handlerP.n_dofs() << std::endl;

    //Vx
    DynamicSparsityPattern dspVx(dof_handlerVx.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handlerVx, dspVx);
    sparsity_patternVx.copy_from(dspVx);
    
    system_mVx.reinit (sparsity_patternVx);
   
    solutionVx.reinit (dof_handlerVx.n_dofs());
    predictionVx.reinit (dof_handlerVx.n_dofs());
    correctionVx.reinit (dof_handlerVx.n_dofs());
    old_solutionVx.reinit (dof_handlerVx.n_dofs());
    system_rVx.reinit (dof_handlerVx.n_dofs());
    
    //Vy
    DynamicSparsityPattern dspVy(dof_handlerVy.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handlerVy, dspVy);
    sparsity_patternVy.copy_from(dspVy);
    
    system_mVy.reinit (sparsity_patternVy);
    
    solutionVy.reinit (dof_handlerVy.n_dofs());
    predictionVy.reinit (dof_handlerVy.n_dofs());
    correctionVy.reinit (dof_handlerVy.n_dofs());
    old_solutionVy.reinit (dof_handlerVy.n_dofs());
    system_rVy.reinit (dof_handlerVy.n_dofs());
    
    //Vz
    DynamicSparsityPattern dspVz(dof_handlerVz.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handlerVz, dspVz);
    sparsity_patternVz.copy_from(dspVz);
    
    system_mVz.reinit (sparsity_patternVz);
    
    solutionVz.reinit (dof_handlerVz.n_dofs());
    predictionVz.reinit (dof_handlerVz.n_dofs());
    correctionVz.reinit (dof_handlerVz.n_dofs());
    old_solutionVz.reinit (dof_handlerVz.n_dofs());
    system_rVz.reinit (dof_handlerVz.n_dofs());
    
    //P
    DynamicSparsityPattern dspP(dof_handlerP.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handlerP, dspP);
    sparsity_patternP.copy_from(dspP);
    
    system_mP.reinit (sparsity_patternP);
    
    solutionP.reinit (dof_handlerP.n_dofs());
    old_solutionP.reinit (dof_handlerP.n_dofs());
    system_rP.reinit (dof_handlerP.n_dofs());
}

void tube::assemble_system()
{
	TimerOutput::Scope timer_section(*timer, "FEM step");
	
	old_solutionVx = solutionVx; 
	old_solutionVy = solutionVy;
	old_solutionVz = solutionVz;
	old_solutionP = solutionP;
	
	QGauss<3>   quadrature_formula(2);
    QGauss<2>   face_quadrature_formula(2);
	
    FEValues<3> feVx_values (feVx, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<3> feVy_values (feVy, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<3> feVz_values (feVz, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<3> feP_values (feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    FEFaceValues<3> feVx_face_values (feVx, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    FEFaceValues<3> feVy_face_values (feVy, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    FEFaceValues<3> feVz_face_values (feVz, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    FEFaceValues<3> feP_face_values (feP, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    
    const unsigned int dofs_per_cellVx = feVx.dofs_per_cell,
    dofs_per_cellVy = feVy.dofs_per_cell,
    dofs_per_cellVz = feVz.dofs_per_cell,
    dofs_per_cellP = feP.dofs_per_cell;
    
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    FullMatrix<double> local_matrixVx (dofs_per_cellVx, dofs_per_cellVx),
    local_matrixVy (dofs_per_cellVy, dofs_per_cellVy),
    local_matrixVz (dofs_per_cellVz, dofs_per_cellVz),
    local_matrixP (dofs_per_cellP, dofs_per_cellP);
    
    Vector<double> local_rhsVx (dofs_per_cellVx),
    local_rhsVy (dofs_per_cellVy),
    local_rhsVz (dofs_per_cellVz),
    local_rhsP (dofs_per_cellP);
    
    std::vector<types::global_dof_index> local_dof_indicesVx (dofs_per_cellVx),
    local_dof_indicesVy (dofs_per_cellVy),
    local_dof_indicesVz (dofs_per_cellVz),
    local_dof_indicesP (dofs_per_cellP);
										 
	const double mu = 1.0,
		         rho = 1.0;
		
	for(int nOuterCorr = 0; nOuterCorr < 1; ++nOuterCorr){
		system_mVx=0.0;
		system_rVx=0.0;
		system_mVy=0.0;
		system_rVy=0.0;
		system_mVz=0.0;
		system_rVz=0.0;
				
	//Vx
	{
		DoFHandler<3>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<3>::active_cell_iterator endc = dof_handlerVx.end();
		
		for (; cell!=endc; ++cell) {
			feVx_values.reinit (cell);
			local_matrixVx = 0.0;
			local_rhsVx = 0.0;
		
			for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
				for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
					const Tensor<0,3> Ni_vel = feVx_values.shape_value (i,q_index);
					const Tensor<1,3> Ni_vel_grad = feVx_values.shape_grad (i,q_index);
					
					for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
						const Tensor<0,3> Nj_vel = feVx_values.shape_value (j,q_index);
						const Tensor<1,3> Nj_vel_grad = feVx_values.shape_grad (j,q_index);

						local_matrixVx(i,j) += rho * Ni_vel * Nj_vel * feVx_values.JxW(q_index);
						//implicit account for tau_ij
						local_matrixVx(i,j) += mu * time_step * (4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0] + Ni_vel_grad[1] * Nj_vel_grad[1] + Ni_vel_grad[2] * Nj_vel_grad[2]) * feVx_values.JxW (q_index);

						//explicit account for tau_ij
						local_rhsVx(i) -= mu * time_step * ((Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0))
							+ (Ni_vel_grad[2] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[2]) * old_solutionVz(cell->vertex_dof_index(j,0))) * feVx_values.JxW (q_index);
						
						local_rhsVx(i) += rho * Nj_vel * Ni_vel * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
					}//j
				}//i
			}//q_index
			
			for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
					feVx_face_values.reinit (cell, face_number);
                    feVy_face_values.reinit (cell, face_number);
                    feVz_face_values.reinit (cell, face_number);
                        
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
						double duydy = 0.0;
						double duzdz = 0.0;
                            
                        for (unsigned int i=0; i<dofs_per_cellVx; ++i){
                            duydy += feVy_face_values.shape_grad(i,q_point)[1] * old_solutionVy(cell->vertex_dof_index(i,0));
                            duzdz += feVz_face_values.shape_grad(i,q_point)[2] * old_solutionVz(cell->vertex_dof_index(i,0));
						}
                            
                        for (unsigned int i=0; i<dofs_per_cellVx; ++i){
							local_rhsVx(i) += mu * time_step * feVx_face_values.shape_value(i,q_point) * (-2.0 * (duydy + duzdz) / 3.0) *
								feVx_face_values.normal_vector(q_point)[0] * feVx_face_values.JxW(q_point);
                        }
                    }
				}
      
			cell->get_dof_indices (local_dof_indicesVx);

			for (unsigned int i=0; i<dofs_per_cellVx; ++i)
				for (unsigned int j=0; j<dofs_per_cellVx; ++j)
					system_mVx.add (local_dof_indicesVx[i], local_dof_indicesVx[j], local_matrixVx(i,j));
											
			for (unsigned int i=0; i<dofs_per_cellVx; ++i)
				system_rVx(local_dof_indicesVx[i]) += local_rhsVx(i);
		}//cell
    }//Vx
	    
	std::map<types::global_dof_index,double> boundary_valuesVx2;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<3>(0.0), boundary_valuesVx2);
	MatrixTools::apply_boundary_values (boundary_valuesVx2,	system_mVx, predictionVx, system_rVx);

	std::map<types::global_dof_index,double> boundary_valuesVx3;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<3>(0.0), boundary_valuesVx3);
	MatrixTools::apply_boundary_values (boundary_valuesVx3, system_mVx,	predictionVx, system_rVx);

	std::map<types::global_dof_index,double> boundary_valuesVx4;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 4, ConstantFunction<3>(0.0), boundary_valuesVx4);
	MatrixTools::apply_boundary_values (boundary_valuesVx4,	system_mVx, predictionVx, system_rVx);

	std::map<types::global_dof_index,double> boundary_valuesVx5;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 5, ConstantFunction<3>(0.0), boundary_valuesVx5);
	MatrixTools::apply_boundary_values (boundary_valuesVx5, system_mVx,	predictionVx, system_rVx);
						
	solveVx ();
	
	//Vy
	{
		DoFHandler<3>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<3>::active_cell_iterator endc = dof_handlerVy.end();
			
		for (; cell!=endc; ++cell) {
			feVy_values.reinit (cell);
			local_matrixVy = 0.0;
			local_rhsVy = 0.0;

			for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
				for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
					const Tensor<0,3> Ni_vel = feVy_values.shape_value (i,q_index);
					const Tensor<1,3> Ni_vel_grad = feVy_values.shape_grad (i,q_index);
	
					for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
						const Tensor<0,3> Nj_vel = feVy_values.shape_value (j,q_index);
						const Tensor<1,3> Nj_vel_grad = feVy_values.shape_grad (j,q_index);

						local_matrixVy(i,j) += rho * Ni_vel * Nj_vel * feVy_values.JxW(q_index);
						//implicit account for tau_ij
						local_matrixVy(i,j) += mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1] + Ni_vel_grad[2] * Nj_vel_grad[2]) * feVy_values.JxW (q_index);						
												
						//explicit account for tau_ij
						local_rhsVy(i) -= mu * time_step * ((Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0))
							+ (Ni_vel_grad[2] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[2]) * old_solutionVz(cell->vertex_dof_index(j,0))) * feVy_values.JxW (q_index);

						local_rhsVy(i) += rho * Nj_vel * Ni_vel * old_solutionVy(cell->vertex_dof_index(j,0)) *  feVy_values.JxW (q_index); 
					}//j
				}//i
			}//q_index
			
			for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
                    feVy_face_values.reinit (cell, face_number);
                    feVx_face_values.reinit (cell, face_number);
                        
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
						double duxdy = 0.0;
                            
                        for (unsigned int i=0; i<dofs_per_cellVy; ++i)
                            duxdy += feVx_face_values.shape_grad(i,q_point)[1] * old_solutionVx(cell->vertex_dof_index(i,0));
                            
                        for (unsigned int i=0; i<dofs_per_cellVy; ++i){
                            local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * duxdy *
								feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
                        }
                    }
				}

			cell->get_dof_indices (local_dof_indicesVy);

			for (unsigned int i=0; i<dofs_per_cellVy; ++i)
				for (unsigned int j=0; j<dofs_per_cellVy; ++j)
					system_mVy.add (local_dof_indicesVy[i], local_dof_indicesVy[j],	local_matrixVy(i,j));

			for (unsigned int i=0; i<dofs_per_cellVy; ++i)
				system_rVy(local_dof_indicesVy[i]) += local_rhsVy(i);
		}//cell
    }//Vy
	    
	std::map<types::global_dof_index,double> boundary_valuesVy2;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<3>(0.0), boundary_valuesVy2);
	MatrixTools::apply_boundary_values (boundary_valuesVy2, system_mVy, predictionVy, system_rVy);
										
	std::map<types::global_dof_index,double> boundary_valuesVy3;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<3>(0.0), boundary_valuesVy3);
	MatrixTools::apply_boundary_values (boundary_valuesVy3,	system_mVy,	predictionVy, system_rVy);
											
	std::map<types::global_dof_index,double> boundary_valuesVy4;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 4, ConstantFunction<3>(0.0), boundary_valuesVy4);
	MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy, predictionVy, system_rVy);
										
	std::map<types::global_dof_index,double> boundary_valuesVy5;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 5, ConstantFunction<3>(0.0), boundary_valuesVy5);
	MatrixTools::apply_boundary_values (boundary_valuesVy5,	system_mVy,	predictionVy, system_rVy);

	solveVy ();

	//Vz
	{
		DoFHandler<3>::active_cell_iterator cell = dof_handlerVz.begin_active();
		DoFHandler<3>::active_cell_iterator endc = dof_handlerVz.end();
			
		for (; cell!=endc; ++cell) {
			feVz_values.reinit (cell);
			local_matrixVz = 0.0;
			local_rhsVz = 0.0;

			for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
				for (unsigned int i=0; i<dofs_per_cellVz; ++i) {
					const Tensor<0,3> Ni_vel = feVz_values.shape_value (i,q_index);
					const Tensor<1,3> Ni_vel_grad = feVz_values.shape_grad (i,q_index);
	
					for (unsigned int j=0; j<dofs_per_cellVz; ++j) {
						const Tensor<0,3> Nj_vel = feVz_values.shape_value (j,q_index);
						const Tensor<1,3> Nj_vel_grad = feVz_values.shape_grad (j,q_index);

						local_matrixVz(i,j) += rho * Ni_vel * Nj_vel * feVz_values.JxW(q_index);
						//implicit account for tau_ij
						local_matrixVz(i,j) += mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[2]) * feVz_values.JxW (q_index);						
												
						//explicit account for tau_ij
						local_rhsVz(i) -= mu * time_step * ((Ni_vel_grad[0] * Nj_vel_grad[2] - 2.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0))
							+ (Ni_vel_grad[1] * Nj_vel_grad[2] - 2.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0))) * feVz_values.JxW (q_index);

						local_rhsVz(i) += rho * Nj_vel * Ni_vel * old_solutionVz(cell->vertex_dof_index(j,0)) *  feVz_values.JxW (q_index); 
					}//j
				}//i
			}//q_index
			
			for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
                    feVz_face_values.reinit (cell, face_number);
                    feVx_face_values.reinit (cell, face_number);
                        
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
						double duxdz = 0.0;
                            
                        for (unsigned int i=0; i<dofs_per_cellVz; ++i)
                            duxdz += feVx_face_values.shape_grad(i,q_point)[2] * old_solutionVx(cell->vertex_dof_index(i,0));
                            
                        for (unsigned int i=0; i<dofs_per_cellVz; ++i){
                            local_rhsVz(i) += mu * time_step * feVz_face_values.shape_value(i,q_point) * duxdz *
								feVz_face_values.normal_vector(q_point)[0] * feVz_face_values.JxW(q_point);
                        }
                    }
				}
      
			cell->get_dof_indices (local_dof_indicesVz);

			for (unsigned int i=0; i<dofs_per_cellVz; ++i)
				for (unsigned int j=0; j<dofs_per_cellVz; ++j)
					system_mVz.add (local_dof_indicesVz[i], local_dof_indicesVz[j],	local_matrixVz(i,j));

			for (unsigned int i=0; i<dofs_per_cellVz; ++i)
				system_rVz(local_dof_indicesVz[i]) += local_rhsVz(i);
		}//cell
    }//Vz
	    
	std::map<types::global_dof_index,double> boundary_valuesVz2;
	VectorTools::interpolate_boundary_values (dof_handlerVz, 2, ConstantFunction<3>(0.0), boundary_valuesVz2);
	MatrixTools::apply_boundary_values (boundary_valuesVz2, system_mVz, predictionVz, system_rVz);
										
	std::map<types::global_dof_index,double> boundary_valuesVz3;
	VectorTools::interpolate_boundary_values (dof_handlerVz, 3, ConstantFunction<3>(0.0), boundary_valuesVz3);
	MatrixTools::apply_boundary_values (boundary_valuesVz3,	system_mVz,	predictionVz, system_rVz);
											
	std::map<types::global_dof_index,double> boundary_valuesVz4;
	VectorTools::interpolate_boundary_values (dof_handlerVz, 4, ConstantFunction<3>(0.0), boundary_valuesVz4);
	MatrixTools::apply_boundary_values (boundary_valuesVz4, system_mVz, predictionVz, system_rVz);
										
	std::map<types::global_dof_index,double> boundary_valuesVz5;
	VectorTools::interpolate_boundary_values (dof_handlerVz, 5, ConstantFunction<3>(0.0), boundary_valuesVz5);
	MatrixTools::apply_boundary_values (boundary_valuesVz5,	system_mVz,	predictionVz, system_rVz);

	solveVz ();

	//P
		
	for (int n_cor=0; n_cor<1; ++n_cor){
		system_mP=0.0;
		system_rP=0.0;
		
		{
			DoFHandler<3>::active_cell_iterator cell = dof_handlerP.begin_active();
			DoFHandler<3>::active_cell_iterator endc = dof_handlerP.end();
		
			for (; cell!=endc; ++cell) {
				local_matrixP = 0.0;
				local_rhsP = 0.0;
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feVz_values.reinit (cell);
				feP_values.reinit (cell);
					
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellP; ++i) {
						const Tensor<1,3> Nidx_pres = feP_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellP; ++j) {
							const Tensor<0,3> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,3> Njdx_pres = feP_values.shape_grad (j,q_index);
							
							local_matrixP(i,j) += Nidx_pres * Njdx_pres * feP_values.JxW(q_index);
											
							local_rhsP(i) += rho / time_step * (predictionVx(cell->vertex_dof_index(j,0)) * Nidx_pres[0] + 
							                   predictionVy(cell->vertex_dof_index(j,0)) * Nidx_pres[1] + predictionVz(cell->vertex_dof_index(j,0)) * Nidx_pres[2]) * Nj_vel * feP_values.JxW (q_index);
						}//j
					}//i
				}//q_index

				for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
						feP_face_values.reinit (cell, face_number);
                        feVx_face_values.reinit (cell, face_number);
					
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double Vx_q_point_value = 0.0;
							for (unsigned int i=0; i<dofs_per_cellP; ++i)
								Vx_q_point_value += feVx_face_values.shape_value(i,q_point) * predictionVx(cell->vertex_dof_index(i,0));
						
							for (unsigned int i=0; i<dofs_per_cellP; ++i){
								local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
                                    feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
								local_rhsP(i) += feP_face_values.shape_value(i,q_point) * (-10.0) * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
							}
						}
					}

				cell->get_dof_indices (local_dof_indicesP);

				for (unsigned int i=0; i<dofs_per_cellP; ++i)
					for (unsigned int j=0; j<dofs_per_cellP; ++j)
						system_mP.add (local_dof_indicesP[i], local_dof_indicesP[j], local_matrixP(i,j));

				for (unsigned int i=0; i<dofs_per_cellP; ++i)
					system_rP(local_dof_indicesP[i]) += local_rhsP(i);
			}//cell
		}//P
		
		std::map<types::global_dof_index,double> boundary_valuesP1;
		VectorTools::interpolate_boundary_values (dof_handlerP, 1, ConstantFunction<3>(0.0), boundary_valuesP1);
		MatrixTools::apply_boundary_values (boundary_valuesP1, system_mP, solutionP, system_rP);
		
		solveP ();
			
		//Vx correction
		{
			system_mVx = 0.0;
			system_rVx = 0.0;
			
			DoFHandler<3>::active_cell_iterator cell = dof_handlerVx.begin_active();
			DoFHandler<3>::active_cell_iterator endc = dof_handlerVx.end();
		
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVx = 0.0;
				local_rhsVx = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,3> Ni_vel = feVx_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,3> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,3> Nj_p_grad = feP_values.shape_grad (j,q_index);

							local_matrixVx(i,j) += Ni_vel * Nj_vel * feVx_values.JxW(q_index);

							local_rhsVx(i) -= time_step/rho * Ni_vel * Nj_p_grad[0] * solutionP(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVx);

				for (unsigned int i=0; i<dofs_per_cellVx; ++i)
					for (unsigned int j=0; j<dofs_per_cellVx; ++j)
						system_mVx.add (local_dof_indicesVx[i],	local_dof_indicesVx[j],	local_matrixVx(i,j));
											
				for (unsigned int i=0; i<dofs_per_cellVx; ++i)
					system_rVx(local_dof_indicesVx[i]) += local_rhsVx(i);
			}//cell
			
			std::map<types::global_dof_index,double> boundary_valuesVx2;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<3>(0.0), boundary_valuesVx2);
			MatrixTools::apply_boundary_values (boundary_valuesVx2,	system_mVx, predictionVx, system_rVx);

			std::map<types::global_dof_index,double> boundary_valuesVx3;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<3>(0.0), boundary_valuesVx3);
			MatrixTools::apply_boundary_values (boundary_valuesVx3, system_mVx,	predictionVx, system_rVx);

			std::map<types::global_dof_index,double> boundary_valuesVx4;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 4, ConstantFunction<3>(0.0), boundary_valuesVx4);
			MatrixTools::apply_boundary_values (boundary_valuesVx4,	system_mVx, predictionVx, system_rVx);

			std::map<types::global_dof_index,double> boundary_valuesVx5;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 5, ConstantFunction<3>(0.0), boundary_valuesVx5);
			MatrixTools::apply_boundary_values (boundary_valuesVx5, system_mVx,	predictionVx, system_rVx);
		}//Vx
	    						
		solveVx (true);
		
		//Vy correction
		{
			system_mVy = 0.0;
			system_rVy = 0.0;
			
			DoFHandler<3>::active_cell_iterator cell = dof_handlerVy.begin_active();
			DoFHandler<3>::active_cell_iterator endc = dof_handlerVy.end();
		
			for (; cell!=endc; ++cell) {
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVy = 0.0;
				local_rhsVy = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,3> Ni_vel = feVy_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,3> Nj_vel = feVy_values.shape_value (j,q_index);
							const Tensor<1,3> Nj_p_grad = feP_values.shape_grad (j,q_index);

							local_matrixVy(i,j) += Ni_vel * Nj_vel * feVy_values.JxW(q_index);

							local_rhsVy(i) -= time_step/rho * Ni_vel * Nj_p_grad[1] * solutionP(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVy);

				for (unsigned int i=0; i<dofs_per_cellVy; ++i)
					for (unsigned int j=0; j<dofs_per_cellVy; ++j)
						system_mVy.add (local_dof_indicesVy[i],	local_dof_indicesVy[j],	local_matrixVy(i,j));
											
				for (unsigned int i=0; i<dofs_per_cellVy; ++i)
					system_rVy(local_dof_indicesVy[i]) += local_rhsVy(i);
			}//cell
			
			std::map<types::global_dof_index,double> boundary_valuesVy2;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<3>(0.0), boundary_valuesVy2);
			MatrixTools::apply_boundary_values (boundary_valuesVy2, system_mVy, correctionVy, system_rVy);

			std::map<types::global_dof_index,double> boundary_valuesVy3;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<3>(0.0), boundary_valuesVy3);
			MatrixTools::apply_boundary_values (boundary_valuesVy3, system_mVy, correctionVy, system_rVy);
			
			std::map<types::global_dof_index,double> boundary_valuesVy4;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 4, ConstantFunction<3>(0.0), boundary_valuesVy4);
			MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy, correctionVy, system_rVy);

			std::map<types::global_dof_index,double> boundary_valuesVy5;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 5, ConstantFunction<3>(0.0), boundary_valuesVy5);
			MatrixTools::apply_boundary_values (boundary_valuesVy5, system_mVy, correctionVy, system_rVy);
		}//Vy

		solveVy (true);
		
		//Vz correction
		{
			system_mVz = 0.0;
			system_rVz = 0.0;
			
			DoFHandler<3>::active_cell_iterator cell = dof_handlerVz.begin_active();
			DoFHandler<3>::active_cell_iterator endc = dof_handlerVz.end();
		
			for (; cell!=endc; ++cell) {
				feVz_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVz = 0.0;
				local_rhsVz = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVz; ++i) {
						const Tensor<0,3> Ni_vel = feVz_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVz; ++j) {
							const Tensor<0,3> Nj_vel = feVz_values.shape_value (j,q_index);
							const Tensor<1,3> Nj_p_grad = feP_values.shape_grad (j,q_index);

							local_matrixVz(i,j) += Ni_vel * Nj_vel * feVz_values.JxW(q_index);

							local_rhsVz(i) -= time_step/rho * Ni_vel * Nj_p_grad[2] * solutionP(cell->vertex_dof_index(j,0)) * feVz_values.JxW (q_index);
						}//j
					}//i
				}//q_index
      
				cell->get_dof_indices (local_dof_indicesVz);

				for (unsigned int i=0; i<dofs_per_cellVz; ++i)
					for (unsigned int j=0; j<dofs_per_cellVz; ++j)
						system_mVz.add (local_dof_indicesVz[i],	local_dof_indicesVz[j],	local_matrixVz(i,j));
											
				for (unsigned int i=0; i<dofs_per_cellVz; ++i)
					system_rVz(local_dof_indicesVz[i]) += local_rhsVz(i);
			}//cell
			
			std::map<types::global_dof_index,double> boundary_valuesVz2;
			VectorTools::interpolate_boundary_values (dof_handlerVz, 2, ConstantFunction<3>(0.0), boundary_valuesVz2);
			MatrixTools::apply_boundary_values (boundary_valuesVz2, system_mVz, correctionVz, system_rVz);

			std::map<types::global_dof_index,double> boundary_valuesVz3;
			VectorTools::interpolate_boundary_values (dof_handlerVz, 3, ConstantFunction<3>(0.0), boundary_valuesVz3);
			MatrixTools::apply_boundary_values (boundary_valuesVz3, system_mVz, correctionVz, system_rVz);
			
			std::map<types::global_dof_index,double> boundary_valuesVz4;
			VectorTools::interpolate_boundary_values (dof_handlerVz, 4, ConstantFunction<3>(0.0), boundary_valuesVz4);
			MatrixTools::apply_boundary_values (boundary_valuesVz4, system_mVz, correctionVz, system_rVz);

			std::map<types::global_dof_index,double> boundary_valuesVz5;
			VectorTools::interpolate_boundary_values (dof_handlerVz, 5, ConstantFunction<3>(0.0), boundary_valuesVz5);
			MatrixTools::apply_boundary_values (boundary_valuesVz5, system_mVz, correctionVz, system_rVz);
		}//Vz

		solveVz (true);
		
		solutionVx = predictionVx;
		solutionVx += correctionVx;
		solutionVy = predictionVy;
		solutionVy += correctionVy;
		solutionVz = predictionVz;
		solutionVz += correctionVz;
		
		old_solutionP = solutionP;
	}//n_cor
	}
}

/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void tube::solveVx(bool correction)
{
	SolverControl solver_control (10000, 1e-7);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVx, 1.0);
	if(correction) solver.solve (system_mVx, correctionVx, system_rVx, preconditioner);
	else solver.solve (system_mVx, predictionVx, system_rVx, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vx converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vx failed to converge" << std::endl;
}

void tube::solveVy(bool correction)
{
	SolverControl solver_control (10000, 1e-7);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVy, 1.0);
	if(correction) solver.solve (system_mVy, correctionVy, system_rVy, preconditioner);
	else solver.solve (system_mVy, predictionVy, system_rVy, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vy failed to converge" << std::endl;
}

void tube::solveVz(bool correction)
{
	SolverControl solver_control (10000, 1e-7);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVz, 1.0);
	if(correction) solver.solve (system_mVz, correctionVz, system_rVz, preconditioner);
	else solver.solve (system_mVz, predictionVz, system_rVz, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vz converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vz failed to converge" << std::endl;
}

void tube::solveP()
{
	SolverControl solver_control (10000, 1e-7);
	SolverBicgstab<> solver (solver_control);

	PreconditionSSOR<> preconditioner;
	
	preconditioner.initialize(system_mP, 1.0);
	solver.solve (system_mP, solutionP, system_rP,
                  preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for P failed to converge" << std::endl;
}

/*!
 * \brief Вывод результатов в формате VTK
 */
void tube::output_results(bool predictionCorrection) 
{
    TimerOutput::Scope timer_section(*timer, "Results output");
    
    DataOut<3> data_out;
    
    data_out.attach_dof_handler (dof_handlerVx);
    data_out.add_data_vector (solutionVx, "Vx");
    data_out.add_data_vector (solutionVy, "Vy");
    data_out.add_data_vector (solutionVz, "Vz");
    data_out.add_data_vector (solutionP, "P");
    
    if(predictionCorrection){
        data_out.add_data_vector (predictionVx, "predVx");
        data_out.add_data_vector (predictionVy, "predVy");
        data_out.add_data_vector (predictionVy, "predVz");
        data_out.add_data_vector (correctionVx, "corVx");
        data_out.add_data_vector (correctionVy, "corVy");
        data_out.add_data_vector (correctionVy, "corVz");
    }
    
    data_out.build_patches ();
    
    const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) +    ".vtk";
    std::ofstream output (filename.c_str());
    data_out.write_vtk (output);
    
    //вывод частиц
    const std::string filename2 =  "particles-" + Utilities::int_to_string (timestep_number, 2) + ".vtk";
    std::ofstream output2 (filename2.c_str());
    output2 << "# vtk DataFile Version 3.0" << std::endl;
    output2 << "Unstructured Grid Example" << std::endl;
    output2 << "ASCII" << std::endl;
    output2 << std::endl;
    output2 << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output2 << "POINTS " << particle_handler.n_global_particles() << " float" << std::endl;
    for(std::unordered_multimap<int, pfem2Particle*>::iterator particleIndex = particle_handler.begin(); particleIndex != particle_handler.end(); ++particleIndex)
        output2 << (*particleIndex).second->get_location() << std::endl;
    
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
        output2 << (*particleIndex).second->get_velocity_component(0) << " " << (*particleIndex).second->get_velocity_component(1) << " " << (*particleIndex).second->get_velocity_component(2) << std::endl;
    
    output2 << std::endl;
}

/*!
 * \brief Основная процедура программы
 * 
 * Подготовительные операции, цикл по времени, вызов вывода результатов
 */
void tube::run()
{	
	timer = new TimerOutput(std::cout, TimerOutput::summary, TimerOutput::wall_times);
	
	build_grid();
	setup_system();
	seed_particles({2, 2, 2});
	
	solutionVx = 0.0;
	solutionVy = 0.0;
	solutionVz = 0.0;
	solutionP = 0.0;
	
	//удаление старых файлов VTK (специфическая команда Linux!!!)
	system("rm solution-*.vtk");
	system("rm particles-*.vtk");

	std::ofstream os("force.csv");

	for (; time<=15; time+=time_step, ++timestep_number) {
		std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
		
		correct_particles_velocities();
		move_particles();
		distribute_particle_velocities_to_grid();
		
		assemble_system();
		if((timestep_number - 1) % 10 == 0) output_results();
		
		//calculate_loads(3, &os);
		
		timer->print_summary();
	}//time
	
	os.close();
	
	delete timer;
}

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
	
  tube tubeproblem;
  tubeproblem.run ();
  
  return 0;
}
