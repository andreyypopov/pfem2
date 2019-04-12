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
	//return 150.0*(0.01 - p[1]*p[1]);
	return 3.0/8.0*(4 - p[1]*p[1]);
}

double parabolicBC::ddy(const Point<2> &p) const
{
	return -3.0/4.0*p[1];
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
	
	SparsityPattern sparsity_patternVx, sparsity_patternVy, sparsity_patternP;
	SparseMatrix<double> system_mVx, system_mVy, system_mP;
	Vector<double> system_rVx, system_rVy, system_rP;
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
  
  const Point<2> bottom_left = Point<2> (-3,-2);
  const Point<2> top_right = Point<2> (10,2);

  std::vector< unsigned int > repetitions {20,6}; 

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

	//имеющаяся процедура в почти неизменном виде
	dof_handlerVx.distribute_dofs (feVx);
	std::cout << "Number of degrees of freedom Vx: "
			  << dof_handlerVx.n_dofs()
			  << std::endl;
			  
	dof_handlerVy.distribute_dofs (feVy);
	std::cout << "Number of degrees of freedom Vy: "
			  << dof_handlerVy.n_dofs()
			  << std::endl;
			  
	dof_handlerP.distribute_dofs (feP);
	std::cout << "Number of degrees of freedom P: "
			  << dof_handlerP.n_dofs()
			  << std::endl;

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
	
	//P
    DynamicSparsityPattern dspP(dof_handlerP.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handlerP, dspP);
	sparsity_patternP.copy_from(dspP);
	
	system_mP.reinit (sparsity_patternP);
	
	solutionP.reinit (dof_handlerP.n_dofs());
	old_solutionP.reinit (dof_handlerP.n_dofs());
    system_rP.reinit (dof_handlerP.n_dofs());
    
    return;
    
    DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active(), endc = dof_handlerVx.end();
	std::ofstream vertices("vertices.txt");
	for (; cell!=endc; ++cell) {
		for (unsigned int i=0; i < 4; ++i){
			vertices << "DoF no. " << cell->vertex_dof_index(i,0) << " is located at " << cell->vertex(i) << std::endl;
		}
	}
	
	vertices.close();
}

void tube::assemble_system()
{
	TimerOutput::Scope timer_section(*timer, "FEM step");
	
	old_solutionVx = solutionVx; 
	old_solutionVy = solutionVy;
	old_solutionP = solutionP;
	
	QGauss<2>   quadrature_formula(2);
	QGauss<1>   face_quadrature_formula(2);
	
	FEValues<2> feVx_values (feVx, quadrature_formula,
                           update_values | update_gradients | update_quadrature_points |
						   update_JxW_values);
	FEValues<2> feVy_values (feVy, quadrature_formula,
                           update_values | update_gradients | update_quadrature_points |
						   update_JxW_values);
	FEValues<2> feP_values (feP, quadrature_formula,
                           update_values | update_gradients | update_quadrature_points |
						   update_JxW_values); 
						   
	FEFaceValues<2> feVy_face_values (feVy, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
	FEFaceValues<2> feP_face_values (feP, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
						                                       
    const unsigned int   dofs_per_cellVx = feVx.dofs_per_cell,
						 dofs_per_cellVy = feVy.dofs_per_cell,
						 dofs_per_cellP = feP.dofs_per_cell;
						 
	const unsigned int   n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();
	
	FullMatrix<double>   local_matrixVx (dofs_per_cellVx, dofs_per_cellVx),
						 local_matrixVy (dofs_per_cellVy, dofs_per_cellVy),
						 local_matrixP (dofs_per_cellP, dofs_per_cellP);
						 
	Vector<double>       local_rhsVx (dofs_per_cellVx),
						 local_rhsVy (dofs_per_cellVy),
						 local_rhsP (dofs_per_cellP);
	
	std::vector<types::global_dof_index> local_dof_indicesVx (dofs_per_cellVx),
	                                     local_dof_indicesVy (dofs_per_cellVy),
										 local_dof_indicesP (dofs_per_cellP);
										 
	const double mu = 1.0,
		         rho = 1.0;
		
	for(int nOuterCorr = 0; nOuterCorr < 1; ++nOuterCorr){
		system_mVx=0.0;
		system_rVx=0.0;
		system_mVy=0.0;
		system_rVy=0.0;
				
	//Vx
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
		int number = 0;
		
		for (; cell!=endc; ++cell,++number) {
			feVx_values.reinit (cell);
			feVy_values.reinit (cell);
			feP_values.reinit (cell);
			local_matrixVx = 0.0;
			local_rhsVx = 0.0;
		
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

						//explicit account for tau_ij
						//local_rhsVx(i) -= mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						local_rhsVx(i) -= mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						
						local_rhsVx(i) += rho * Nj_vel * Ni_vel * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
					}//j
				}//i
			}//q_index
      
			/*if(timestep_number == 1){
				const std::string filenameVx =  "matrixVx-" + Utilities::int_to_string (number) +	".txt";
				std::ofstream matrixFile (filenameVx.c_str());
			
				for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
					matrixFile << cell->vertex_dof_index(i,0) << " ";
				}
				matrixFile << std::endl;
			
				local_matrixVx.print(matrixFile, 16, 10);
			
				matrixFile.close();
			}*/
      
			cell->get_dof_indices (local_dof_indicesVx);

			for (unsigned int i=0; i<dofs_per_cellVx; ++i)
				for (unsigned int j=0; j<dofs_per_cellVx; ++j){
					system_mVx.add (local_dof_indicesVx[i], local_dof_indicesVx[j], local_matrixVx(i,j));
				}
											
			for (unsigned int i=0; i<dofs_per_cellVx; ++i)
				system_rVx(local_dof_indicesVx[i]) += local_rhsVx(i);
		}//cell
    }//Vx
	    
	/*if(timestep_number==1)
	{
		std::ofstream matrStream("PrematrixVx.txt");
		std::ofstream rhsStream("PrerhsVx.txt");
	
		system_mVx.print_formatted(matrStream, 8, true, 0, "0.0");
		system_rVx.print(rhsStream, 8, true);
	
		matrStream.close();
		rhsStream.close();
	}*/
	    
	std::map<types::global_dof_index,double> boundary_valuesVx1;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 0, parabolicBC(), boundary_valuesVx1);
	MatrixTools::apply_boundary_values (boundary_valuesVx1, system_mVx,	predictionVx,	system_rVx);
										
	/*std::map<types::global_dof_index,double> boundary_valuesVx2;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 0, ConstantFunction<2>(1.0), boundary_valuesVx2);
	MatrixTools::apply_boundary_values (boundary_valuesVx2, system_mVx,	predictionVx,	system_rVx);*/
										
	std::map<types::global_dof_index,double> boundary_valuesVx3;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<2>(0.0), boundary_valuesVx3);
	MatrixTools::apply_boundary_values (boundary_valuesVx3,	system_mVx, predictionVx, system_rVx);

	std::map<types::global_dof_index,double> boundary_valuesVx4;
	VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), boundary_valuesVx4);
	MatrixTools::apply_boundary_values (boundary_valuesVx4, system_mVx,	predictionVx, system_rVx);
						
	solveVx ();
	
	/*if(timestep_number==1)
	{
		std::ofstream matrStream("matrixVx.txt");
		std::ofstream rhsStream("rhsVx.txt");
		std::ofstream solStream("solVx.txt");
	
		system_mVx.print_formatted(matrStream, 8, true, 0, "0.0");
		system_rVx.print(rhsStream, 8, true);
		solutionVx.print(solStream, 8, true);
	
		matrStream.close();
		rhsStream.close();
		solStream.close();
	}*/
	
	//Vy
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
			
		parabolicBC boundaryCond;			
			
		for (; cell!=endc; ++cell) {
			feVx_values.reinit (cell);
			feVy_values.reinit (cell);
			feP_values.reinit (cell);
			local_matrixVy = 0.0;
			local_rhsVy = 0.0;

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
												
						//explicit account for tau_ij
						local_rhsVy(i) -= mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
						//local_rhsVy(i) -= mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);

						local_rhsVy(i) += rho * Nj_vel * Ni_vel * old_solutionVy(cell->vertex_dof_index(j,0)) *  feVy_values.JxW (q_index); 
					}//j
				}//i
			}//q_index
			
			for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0)){
					feVy_face_values.reinit (cell, face_number);
					
					for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
						for (unsigned int i=0; i<dofs_per_cellVy; ++i)
							local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * boundaryCond.ddy(feVy_face_values.quadrature_point(q_point)) *
								feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
				} else if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1)){
					feVy_face_values.reinit (cell, face_number);
					
					for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
						double duxdy = 0.0;
						for (unsigned int i=0; i<dofs_per_cellVy; ++i)
							duxdy += feVx_values.shape_grad(i,q_point)[1] * old_solutionVx(cell->vertex_dof_index(i,0));
					
						for (unsigned int i=0; i<dofs_per_cellVy; ++i)
							local_rhsVy(i) += mu * time_step * feVy_face_values.shape_value(i,q_point) * duxdy *
								feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
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
	    
	std::map<types::global_dof_index,double> boundary_valuesVy1;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 0, ConstantFunction<2>(0.0), boundary_valuesVy1);
	MatrixTools::apply_boundary_values (boundary_valuesVy1, system_mVy, predictionVy,	system_rVy);		
    
/*	std::map<types::global_dof_index,double> boundary_valuesVy3;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 1, ConstantFunction<2>(0.0), boundary_valuesVy3);
	MatrixTools::apply_boundary_values (boundary_valuesVy3, system_mVy, predictionVy,	system_rVy);
*/

	std::map<types::global_dof_index,double> boundary_valuesVy4;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<2>(0.0), boundary_valuesVy4);
	MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy, predictionVy, system_rVy);
										
	std::map<types::global_dof_index,double> boundary_valuesVy2;
	VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), boundary_valuesVy2);
	MatrixTools::apply_boundary_values (boundary_valuesVy2,	system_mVy,	predictionVy, system_rVy);
											
	solveVy ();

	//P
		
	for (int n_cor=0; n_cor<1; ++n_cor){
		system_mP=0.0;
		system_rP=0.0;
		
		parabolicBC boundaryCond;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
			for (; cell!=endc; ++cell) {
				local_matrixP = 0.0;
				local_rhsP = 0.0;
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
					
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellP; ++i) {
						const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellP; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Njdx_pres = feP_values.shape_grad (j,q_index);
							
							local_matrixP(i,j) += Nidx_pres * Njdx_pres * feP_values.JxW(q_index);
											
							local_rhsP(i) += rho / time_step * (predictionVx(cell->vertex_dof_index(j,0)) * Nidx_pres[0] + 
							                   predictionVy(cell->vertex_dof_index(j,0)) * Nidx_pres[1]) * Nj_vel * feP_values.JxW (q_index);
						}//j
					}//i
				}//q_index

				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0)){
						feP_face_values.reinit (cell, face_number);
					
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
							for (unsigned int i=0; i<dofs_per_cellP; ++i)
								local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * boundaryCond.value(feP_face_values.quadrature_point(q_point)) *
                                    feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
					} else if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1)){
						feP_face_values.reinit (cell, face_number);
					
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double Vx_q_point_value = 0.0;
							for (unsigned int i=0; i<dofs_per_cellP; ++i)
								Vx_q_point_value += feVx_values.shape_value(i,q_point) * predictionVx(cell->vertex_dof_index(i,0));
						
							for (unsigned int i=0; i<dofs_per_cellP; ++i)
								local_rhsP(i) -= rho / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
                                    feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);
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
		VectorTools::interpolate_boundary_values (dof_handlerP, 1, ConstantFunction<2>(0.0), boundary_valuesP1);
		MatrixTools::apply_boundary_values (boundary_valuesP1, system_mP, solutionP, system_rP);
		
		solveP ();
			
		//Vx correction
		{
			system_mVx = 0.0;
			system_rVx = 0.0;
			
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVx = 0.0;
				local_rhsVx = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

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
			
			std::map<types::global_dof_index,double> boundary_valuesVx1;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 0, ConstantFunction<2>(0.0), boundary_valuesVx1);
			MatrixTools::apply_boundary_values (boundary_valuesVx1, system_mVx,	correctionVx, system_rVx);
										
			std::map<types::global_dof_index,double> boundary_valuesVx3;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 2, ConstantFunction<2>(0.0), boundary_valuesVx3);
			MatrixTools::apply_boundary_values (boundary_valuesVx3, system_mVx, correctionVx, system_rVx);

			std::map<types::global_dof_index,double> boundary_valuesVx4;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), boundary_valuesVx4);
			MatrixTools::apply_boundary_values (boundary_valuesVx4, system_mVx, correctionVx, system_rVx);
		}//Vx
	    						
		solveVx (true);
		
		//Vy correction
		{
			system_mVy = 0.0;
			system_rVy = 0.0;
			
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_matrixVy = 0.0;
				local_rhsVy = 0.0;
		
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
					
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

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
			
			std::map<types::global_dof_index,double> boundary_valuesVy1;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 0, ConstantFunction<2>(0.0), boundary_valuesVy1);
			MatrixTools::apply_boundary_values (boundary_valuesVy1, system_mVy,	correctionVy, system_rVy);
										
			std::map<types::global_dof_index,double> boundary_valuesVy3;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 2, ConstantFunction<2>(0.0), boundary_valuesVy3);
			MatrixTools::apply_boundary_values (boundary_valuesVy3, system_mVy, correctionVy, system_rVy);

			std::map<types::global_dof_index,double> boundary_valuesVy4;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), boundary_valuesVy4);
			MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy, correctionVy, system_rVy);
		}//Vy
	    						
		solveVy (true);
		
		solutionVx = predictionVx;
		solutionVx += correctionVx;
		solutionVy = predictionVy;
		solutionVy += correctionVy;
		
		old_solutionP = solutionP;
	}//n_cor
	}
}

/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void tube::solveVx(bool correction)
{
	SolverControl solver_control (10000, 1e-13);
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
	SolverControl solver_control (10000, 1e-13);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVy, 1.0);
	if(correction) solver.solve (system_mVy, correctionVy, system_rVy, preconditioner);
	else solver.solve (system_mVy, predictionVy, system_rVy, preconditioner);
                  
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vy failed to converge" << std::endl;
}

void tube::solveP()
{
	SolverControl solver_control (10000, 1e-13);
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
	
	/*const std::string filenameVx =  "resultsVx-" + Utilities::int_to_string (timestep_number, 2) +	".txt";
	std::ofstream rawResults (filenameVx.c_str());
	for (unsigned int i=0; i<dof_handlerVx.n_dofs(); ++i){
		rawResults << "DoF no. " << i << ", Vx=" << solutionVx(i) << std::endl;
	}
	rawResults.close();*/
	
	DataOut<2> data_out;

	data_out.attach_dof_handler (dof_handlerVx);
	data_out.add_data_vector (solutionVx, "Vx");
	data_out.add_data_vector (solutionVy, "Vy");
	data_out.add_data_vector (solutionP, "P");
	
	if(predictionCorrection){
		data_out.add_data_vector (predictionVx, "predVx");
		data_out.add_data_vector (predictionVy, "predVy");
		data_out.add_data_vector (correctionVx, "corVx");
		data_out.add_data_vector (correctionVy, "corVy");
	}
	
	data_out.build_patches ();

	const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) +	".vtk";
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
	for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
		                                   particleIndex != particle_handler.end(); ++particleIndex){
		output2 << particleIndex->get_location() << " 0" << std::endl;
	}
	
	output2 << std::endl;
	
	output2 << "CELLS " << particle_handler.n_global_particles() << " " << 2 * particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i){
		output2 << "1 " << i << std::endl; 
	}
	
	output2 << std::endl;
	
	output2 << "CELL_TYPES " << particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i){
		output2 << "1 "; 
	}	
	output2 << std::endl;
	
	output2 << std::endl;
	
	output2 << "POINT_DATA " << particle_handler.n_global_particles() << std::endl;
	output2 << "VECTORS velocity float" << std::endl;
	for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
		                                   particleIndex != particle_handler.end(); ++particleIndex){
		output2 << velocity_x[particleIndex->get_id()] << " " << velocity_y[particleIndex->get_id()] << " 0" << std::endl;
	}
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
	seed_particles({2, 2});
	
	solutionVx=0.0;
	solutionVy=0.0;
	solutionP=0.0;
	
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
		if((timestep_number - 1) % 1 == 0) output_results(true);
		
		calculate_loads(3, &os);
		
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
