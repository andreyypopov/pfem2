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
#include <deal.II/lac/constraint_matrix.h>

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
	double ddy(const Point<2> &p) const;
};

double parabolicBC::value(const Point<2> &p, const unsigned int) const
{
	return 1.0;
	//return 0.06 * (25.0 - p[1]*p[1]);
}

double parabolicBC::ddy(const Point<2> &p) const
{
	return 0.0; //-3.0/4.0 * p[1];//
}

class cylinder2D : public pfem2Solver
{
public:
	cylinder2D();

	static void declare_parameters (ParameterHandler &prm);
	void get_parameters (ParameterHandler &prm);
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
						   
	FEFaceValues<2> feVx_face_values, feVy_face_values, feP_face_values;
	
	const unsigned int   dofs_per_cellVx, dofs_per_cellVy, dofs_per_cellP;
	
	const unsigned int n_q_points;
	const unsigned int n_face_q_points;
	
	FullMatrix<double> local_matrixVx, local_matrixVy, local_matrixP;
	
	Vector<double> local_rhsVx, local_rhsVy, local_rhsP;
	
	std::vector<types::global_dof_index> local_dof_indicesVx, local_dof_indicesVy, local_dof_indicesP;
	
	double mu() const {return mu_; };
	double rho() const {return rho_; };
	
	SparsityPattern sparsity_patternVx, sparsity_patternVy, sparsity_patternP;
	SparseMatrix<double> system_mVx, system_mVy, system_mP;
	Vector<double> system_rVx, system_rVy, system_rP;
	
private:
	double mu_, rho_, final_time_, accuracy_;
	int num_of_part_x_, num_of_part_y_;
	double velX_inlet_, velX_wall_, velX_cyl_,
		   velY_inlet_, velY_wall_, velY_cyl_,
		   press_outlet_;
	int num_of_iter_, num_of_particles_x_, num_of_particles_y_, num_of_data_;
	std::string mesh_file_;  
};

cylinder2D::cylinder2D()
	: pfem2Solver(),
	quadrature_formula(2),
	face_quadrature_formula(2),
	feVx_values (feVx, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVy_values (feVy, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feP_values (feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVx_face_values (feVx, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	feVy_face_values (feVy, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	feP_face_values (feP, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	dofs_per_cellVx (feVx.dofs_per_cell),
	dofs_per_cellVy (feVy.dofs_per_cell),
	dofs_per_cellP (feP.dofs_per_cell),
	n_q_points (quadrature_formula.size()),
	n_face_q_points (face_quadrature_formula.size()),
	local_matrixVx (dofs_per_cellVx, dofs_per_cellVx),
	local_matrixVy (dofs_per_cellVy, dofs_per_cellVy),
	local_matrixP (dofs_per_cellP, dofs_per_cellP),
	local_rhsVx (dofs_per_cellVx),
	local_rhsVy (dofs_per_cellVy),
	local_rhsP (dofs_per_cellP),
	local_dof_indicesVx (dofs_per_cellVx),
	local_dof_indicesVy (dofs_per_cellVy),
	local_dof_indicesP (dofs_per_cellP)
{

}

void cylinder2D::declare_parameters (ParameterHandler &prm)
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

void cylinder2D::get_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		mu_ = prm.get_double ("Dynamic viscosity");
		rho_ = prm.get_double ("Density");
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
void cylinder2D::build_grid ()
{
	TimerOutput::Scope timer_section(*timer, "Mesh construction"); 
	
	GridIn<2> gridin;
	gridin.attach_triangulation(tria);
	std::ifstream f(mesh_file_);
	gridin.read_unv(f);
	f.close();
	
	std::cout << "The mesh contains " << tria.n_active_cells() << " cells" << std::endl;
}

void cylinder2D::setup_system()
{
	TimerOutput::Scope timer_section(*timer, "System setup");

	dof_handlerVx.distribute_dofs (feVx);
	std::cout << "Number of degrees of freedom Vx: " << dof_handlerVx.n_dofs() << std::endl;
			  
	dof_handlerVy.distribute_dofs (feVy);
	std::cout << "Number of degrees of freedom Vy: " << dof_handlerVy.n_dofs() << std::endl;
			  
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
    
    //P
	DynamicSparsityPattern dspP(dof_handlerP.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handlerP, dspP);
	sparsity_patternP.copy_from(dspP);
	
	system_mP.reinit (sparsity_patternP);
	
	solutionP.reinit (dof_handlerP.n_dofs());
	old_solutionP.reinit (dof_handlerP.n_dofs());
    system_rP.reinit (dof_handlerP.n_dofs());
    
    //determine the numbers of DoFs near the point of flow deceleration
    {
		Point<2> probePoint(-0.5, 0.0);
		DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
		for (; cell != endc; ++cell)
			for(unsigned int i = 0; i < 4; i++)
				if(cell->vertex(i).distance(probePoint) < 1e-3){
					probeDoFnumbers.push_back(cell->vertex_dof_index(i, 0));
					break;
				}
	
		if(probeDoFnumbers.empty()){
			double firstDistance = 100.0;
			double secondDistance = 100.0;
			unsigned int firstDoF, secondDoF;
			
			for (cell = dof_handlerP.begin_active(); cell != endc; ++cell){
				for(unsigned int i = 0; i < 4; i++){
					double vertexDistance = cell->vertex(i).distance(probePoint);
					
					if(vertexDistance < firstDistance){
						firstDistance = vertexDistance;
						firstDoF = cell->vertex_dof_index(i, 0);
					} else if(vertexDistance < secondDistance && cell->vertex_dof_index(i, 0) != firstDoF){
						secondDistance = vertexDistance;
						secondDoF = cell->vertex_dof_index(i, 0);
					}
				}
			}
			
			probeDoFnumbers.push_back(firstDoF);
			probeDoFnumbers.push_back(secondDoF);
		}
		
		if(!probeDoFnumbers.empty()){
			std::cout << "Pressure will be probed at DoFs with numbers: ";
			for(std::vector<unsigned int>::iterator it = probeDoFnumbers.begin(); it != probeDoFnumbers.end(); ++it) std::cout << *it << " ";
			std::cout << std::endl;
		}
	}
    
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

void cylinder2D::assemble_system()
{
	TimerOutput::Scope timer_section(*timer, "FEM step");
	
	old_solutionVx = solutionVx; 
	old_solutionVy = solutionVy;
	old_solutionP = solutionP;
	
	Vector<double> innerVx, innerVy;
		
	for(int nOuterCorr = 0; nOuterCorr < 1; ++nOuterCorr){
		innerVx = solutionVx;
		innerVy = solutionVy;

		//Vx
		system_mVx = 0.0;
		system_rVx = 0.0;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
			
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				local_matrixVx = 0.0;
				local_rhsVx = 0.0;
			
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVx_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVx_values.shape_grad (j,q_index);

							local_matrixVx(i,j) += rho() * Ni_vel * Nj_vel * feVx_values.JxW(q_index);
							//implicit account for tau_ij
							local_matrixVx(i,j) += mu() * time_step * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]) * feVx_values.JxW (q_index);
							
							local_rhsVx(i) += rho() * Nj_vel * Ni_vel * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							//explicit account for tau_ij
							local_rhsVx(i) -= mu() * time_step * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * innerVy(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						}//j
					}//i
				}//q_index
				
				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2)){
						feVx_face_values.reinit (cell, face_number);
						
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double duxdx = 0.0;
							double duydy = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVy; ++i){
								duxdx += feVx_face_values.shape_grad(i,q_point)[0] * innerVx(cell->vertex_dof_index(i,0));
								duydy += feVx_face_values.shape_grad(i,q_point)[1] * innerVy(cell->vertex_dof_index(i,0));
							}
						
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								local_rhsVx(i) += mu() * time_step * feVx_face_values.shape_value(i,q_point) * (4.0 / 3.0 * duxdx - 2.0 / 3.0 * duydy) *
									feVx_face_values.normal_vector(q_point)[0] * feVx_face_values.JxW(q_point);
						}
					}		  
		  
				cell->get_dof_indices (local_dof_indicesVx);
				for (unsigned int i=0; i<dofs_per_cellVx; ++i){
                    for (unsigned int j=0; j<dofs_per_cellVx; ++j)
						system_mVx.add (local_dof_indicesVx[i], local_dof_indicesVx[j], local_matrixVx(i,j));
                
                    system_rVx(local_dof_indicesVx[i]) += local_rhsVx(i);
				}
			}//cell

			std::map<types::global_dof_index,double> boundary_valuesVx1;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 1, ConstantFunction<2>(1.0), boundary_valuesVx1);
			//VectorTools::interpolate_boundary_values (dof_handlerVx, 1, parabolicBC(), boundary_valuesVx1);
			MatrixTools::apply_boundary_values (boundary_valuesVx1, system_mVx,    predictionVx,    system_rVx);

			std::map<types::global_dof_index,double> boundary_valuesVx3;
			VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), boundary_valuesVx3);
			MatrixTools::apply_boundary_values (boundary_valuesVx3, system_mVx,    predictionVx,    system_rVx);
               
			//std::map<types::global_dof_index,double> boundary_valuesVx4;
			//VectorTools::interpolate_boundary_values (dof_handlerVx, 4, ConstantFunction<2>(0.0), boundary_valuesVx4);
			//MatrixTools::apply_boundary_values (boundary_valuesVx4, system_mVx,    predictionVx,    system_rVx);
		}//Vx

		solveVx ();
		
		//Vy
		system_mVy = 0.0;
		system_rVy = 0.0;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
			
			for (; cell!=endc; ++cell) {
				feVy_values.reinit (cell);
				local_matrixVy = 0.0;
				local_rhsVy = 0.0;

				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVy_values.shape_grad (i,q_index);
		
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVy_values.shape_grad (j,q_index);
													
							local_matrixVy(i,j) += rho() * Ni_vel * Nj_vel * feVy_values.JxW(q_index);
							//implicit account for tau_ij
							local_matrixVy(i,j) += mu() * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]) * feVy_values.JxW (q_index);

							local_rhsVy(i) += rho() * Nj_vel * Ni_vel * old_solutionVy(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index); 
							//explicit account for tau_ij
							local_rhsVy(i) -= mu() * time_step * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * innerVx(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
						}//j
					}//i
				}//q_index
				
				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2)){
						feVy_face_values.reinit (cell, face_number);
						
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double duxdy = 0.0;
							double duydx = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVy; ++i){
								duxdy += feVy_face_values.shape_grad(i,q_point)[1] * innerVx(cell->vertex_dof_index(i,0));
								duydx += feVy_face_values.shape_grad(i,q_point)[0] * innerVy(cell->vertex_dof_index(i,0));
							}
						
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								local_rhsVy(i) += mu() * time_step * feVy_face_values.shape_value(i,q_point) * (duxdy + duydx) *
									feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
						}
					}
		  
				cell->get_dof_indices (local_dof_indicesVy);
				for (unsigned int i=0; i<dofs_per_cellVy; ++i){
                    for (unsigned int j=0; j<dofs_per_cellVy; ++j)
						system_mVy.add (local_dof_indicesVy[i], local_dof_indicesVy[j], local_matrixVy(i,j));
                
                    system_rVy(local_dof_indicesVy[i]) += local_rhsVy(i);
				}
			}//cell
		
			std::map<types::global_dof_index,double> boundary_valuesVy1;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 1, ConstantFunction<2>(0.0), boundary_valuesVy1);
			MatrixTools::apply_boundary_values (boundary_valuesVy1, system_mVy,    predictionVy,    system_rVy);
        
			std::map<types::global_dof_index,double> boundary_valuesVy3;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), boundary_valuesVy3);
			MatrixTools::apply_boundary_values (boundary_valuesVy3, system_mVy,    predictionVy,    system_rVy);
               
			std::map<types::global_dof_index,double> boundary_valuesVy4;
			VectorTools::interpolate_boundary_values (dof_handlerVy, 4, ConstantFunction<2>(0.0), boundary_valuesVy4);
			MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy,    predictionVy,    system_rVy);
		}//Vy
		
		solveVy ();

		//P	
		{
			system_mP = 0.0;
			system_rP = 0.0;
		
			{
				DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
				for (; cell!=endc; ++cell) {
					feVx_values.reinit (cell);
					feVy_values.reinit (cell);
					feP_values.reinit (cell);
					local_matrixP = 0.0;
					local_rhsP = 0.0;
					
					for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
						for (unsigned int i=0; i<dofs_per_cellP; ++i) {
							const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);

							for (unsigned int j=0; j<dofs_per_cellP; ++j) {
								const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
								const Tensor<1,2> Njdx_pres = feP_values.shape_grad (j,q_index);

								local_matrixP(i,j) += Nidx_pres * Njdx_pres * feP_values.JxW(q_index);

								local_rhsP(i) += rho() / time_step * (predictionVx(cell->vertex_dof_index(j,0)) * Nidx_pres[0] + 
												predictionVy(cell->vertex_dof_index(j,0)) * Nidx_pres[1]) * Nj_vel * feP_values.JxW (q_index);
							}//j
						}//i
					}//q_index

					for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
						if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1 || cell->face(face_number)->boundary_id() == 2)){//inlet + outlet
							feVx_face_values.reinit (cell, face_number);
							feP_face_values.reinit (cell, face_number);
							
							for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
								double Vx_q_point_value = 0.0;
								for (unsigned int i=0; i<dofs_per_cellP; ++i)
									Vx_q_point_value += feVx_face_values.shape_value(i,q_point) * predictionVx(cell->vertex_dof_index(i,0));								
							
								for (unsigned int i=0; i<dofs_per_cellP; ++i){
									local_rhsP(i) -= rho() / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
											feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);			
								}
							}
						}

					cell->get_dof_indices (local_dof_indicesP);
					for (unsigned int i=0; i<dofs_per_cellP; ++i){
						for (unsigned int j=0; j<dofs_per_cellP; ++j)
							system_mP.add (local_dof_indicesP[i], local_dof_indicesP[j], local_matrixP(i,j));
                
						system_rP(local_dof_indicesP[i]) += local_rhsP(i);
					}
				}//cell
			
				std::map<types::global_dof_index,double> boundary_valuesP2;
				VectorTools::interpolate_boundary_values (dof_handlerP, 2, ConstantFunction<2>(0.0), boundary_valuesP2);
				MatrixTools::apply_boundary_values (boundary_valuesP2, system_mP,    solutionP,    system_rP);
			}//P
		
			solveP ();
	
			//Vx correction
			{
				system_mVx = 0.0;
				system_rVx = 0.0;
				
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
				for (; cell!=endc; ++cell) {
					feVx_values.reinit (cell);
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

								//local_rhsVx(i) -= time_step/rho() * Ni_vel * Nj_p_grad[0] * (solutionP(cell->vertex_dof_index(j,0)) - old_solutionP(cell->vertex_dof_index(j,0))) * feVx_values.JxW (q_index);
								local_rhsVx(i) -= time_step/rho() * Ni_vel * Nj_p_grad[0] * solutionP(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							}//j
						}//i
					}//q_index
      
					cell->get_dof_indices (local_dof_indicesVx);
					for (unsigned int i=0; i<dofs_per_cellVx; ++i){
						for (unsigned int j=0; j<dofs_per_cellVx; ++j)
							system_mVx.add (local_dof_indicesVx[i], local_dof_indicesVx[j], local_matrixVx(i,j));
                
						system_rVx(local_dof_indicesVx[i]) += local_rhsVx(i);
					}
				}//cell
				
				std::map<types::global_dof_index,double> boundary_valuesVx1;
				VectorTools::interpolate_boundary_values (dof_handlerVx, 1, ConstantFunction<2>(0.0), boundary_valuesVx1);
				MatrixTools::apply_boundary_values (boundary_valuesVx1, system_mVx,    correctionVx,    system_rVx);

				std::map<types::global_dof_index,double> boundary_valuesVx3;
				VectorTools::interpolate_boundary_values (dof_handlerVx, 3, ConstantFunction<2>(0.0), boundary_valuesVx3);
				MatrixTools::apply_boundary_values (boundary_valuesVx3, system_mVx,    correctionVx,    system_rVx);

				//std::map<types::global_dof_index,double> boundary_valuesVx4;
				//VectorTools::interpolate_boundary_values (dof_handlerVx, 4, ConstantFunction<2>(0.0), boundary_valuesVx4);
				//MatrixTools::apply_boundary_values (boundary_valuesVx4, system_mVx,    correctionVx,    system_rVx);
			}//correction Vx
		
			solveVx (true);
		
			//Vy correction
			{
				system_mVy = 0.0;
				system_rVy = 0.0;
			
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
				for (; cell!=endc; ++cell) {
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

								//local_rhsVy(i) -= time_step/rho() * Ni_vel * Nj_p_grad[1] * (solutionP(cell->vertex_dof_index(j,0)) - old_solutionP(cell->vertex_dof_index(j,0))) * feVy_values.JxW (q_index);
								local_rhsVy(i) -= time_step/rho() * Ni_vel * Nj_p_grad[1] * solutionP(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
							}//j
						}//i
					}//q_index
      
					cell->get_dof_indices (local_dof_indicesVy);
					for (unsigned int i=0; i<dofs_per_cellVy; ++i){
						for (unsigned int j=0; j<dofs_per_cellVy; ++j)
							system_mVy.add (local_dof_indicesVy[i], local_dof_indicesVy[j], local_matrixVy(i,j));

						system_rVy(local_dof_indicesVy[i]) += local_rhsVy(i);
					}
				}//cell
							
				std::map<types::global_dof_index,double> boundary_valuesVy1;							
				VectorTools::interpolate_boundary_values (dof_handlerVy, 1, ConstantFunction<2>(0.0), boundary_valuesVy1);
				MatrixTools::apply_boundary_values (boundary_valuesVy1, system_mVy,    correctionVy,    system_rVy);
        
				std::map<types::global_dof_index,double> boundary_valuesVy3;
				VectorTools::interpolate_boundary_values (dof_handlerVy, 3, ConstantFunction<2>(0.0), boundary_valuesVy3);
				MatrixTools::apply_boundary_values (boundary_valuesVy3, system_mVy,    correctionVy,    system_rVy);
               
				std::map<types::global_dof_index,double> boundary_valuesVy4;
				VectorTools::interpolate_boundary_values (dof_handlerVy, 4, ConstantFunction<2>(0.0), boundary_valuesVy4);
				MatrixTools::apply_boundary_values (boundary_valuesVy4, system_mVy,    correctionVy,    system_rVy);
			}//Vy
			
			solveVy (true);		
		
			solutionVx = predictionVx;
			solutionVx += correctionVx;
			solutionVy = predictionVy;
			solutionVy += correctionVy;
		
			old_solutionP = solutionP;
		}//n_cor
	}//nOuterCorr
}

/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void cylinder2D::solveVx(bool correction)
{
	SolverControl solver_control (num_of_iter_, accuracy_);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVx, 1.0);
	if(correction) solver.solve (system_mVx, correctionVx, system_rVx, preconditioner);
	else solver.solve (system_mVx, predictionVx, system_rVx, preconditioner);

    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vx converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vx failed to converge" << std::endl;
}

void cylinder2D::solveVy(bool correction)
{
	SolverControl solver_control (num_of_iter_, accuracy_);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVy, 1.0);
	if(correction) solver.solve (system_mVy, correctionVy, system_rVy, preconditioner);
	else solver.solve (system_mVy, predictionVy, system_rVy, preconditioner);

    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vy failed to converge" << std::endl;
}

void cylinder2D::solveP()
{
	SolverControl solver_control (num_of_iter_, accuracy_);
	SolverBicgstab<> solver (solver_control);

	PreconditionSSOR<> preconditioner;
	
	preconditioner.initialize(system_mP, 1.0);
	solver.solve (system_mP, solutionP, system_rP, preconditioner);
              
    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for P failed to converge" << std::endl;
}

/*!
 * \brief Вывод результатов в формате VTK
 */
void cylinder2D::output_results(bool predictionCorrection) 
{
	TimerOutput::Scope timer_section(*timer, "Results output");
	
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

/*!
 * \brief Основная процедура программы
 * 
 * Подготовительные операции, цикл по времени, вызов вывода результатов
 */
void cylinder2D::run()
{	
	timer = new TimerOutput(std::cout, TimerOutput::summary, TimerOutput::wall_times);
	
	build_grid();
	setup_system();
	seed_particles({(unsigned int)(num_of_particles_x_), (unsigned int)(num_of_particles_y_)});
	
	solutionVx=0.0;
	solutionVy=0.0;
	solutionP=0.0;
	
	//удаление старых файлов VTK (специфическая команда Linux!!!)
	system("rm solution-*.vtk");
	system("rm particles-*.vtk");

	std::ofstream os("force.csv");

	for (; time <= final_time_; time += time_step, ++timestep_number) {
		std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
		
		correct_particles_velocities();
		move_particles();
		distribute_particle_velocities_to_grid();
		
		assemble_system();
		if((timestep_number - 1) % num_of_data_ == 0) 
			output_results(false);
		
		calculate_loads(3, &os);
		
		timer->print_summary();
	}//time
	
	os.close();
	
	delete timer;
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
	
	ParameterHandler prm;
	cylinder2D cylinder2Dproblem;

	cylinder2Dproblem.declare_parameters (prm);
	prm.parse_input ("input_data.prm");	
	
	prm.print_parameters (std::cout, ParameterHandler::Text);
	// get parameters into the program
	std::cout << "\n\n" << "Getting parameters:" << std::endl;
	
	cylinder2Dproblem.get_parameters (prm);
	
	//std::cout << "mu = " << cylinder2Dproblem.mu() << "\n"; 
	//std::cout << "rho = " << cylinder2Dproblem.rho() << "\n";
	//std::cout << "__________________________________________________________________________________\n";
	//std::cout << "__________________________________________________________________________________\n";
	
	cylinder2Dproblem.run ();
  
	return 0;
}
