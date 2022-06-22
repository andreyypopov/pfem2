#ifndef PFEM2PARTICLE_H
#define PFEM2PARTICLE_H

#define PARTICLES_MOVEMENT_STEPS 3
#define MAX_PARTICLES_PER_CELL_PART 10

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <unordered_map>

#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/base/cuda_size.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

__device__ __constant__ size_t particleSize;
__device__ __constant__ size_t locationPos;
__device__ __constant__ size_t refLocationPos;
__device__ __constant__ size_t velocityPos;
__device__ __constant__ size_t velocityExtPos;
__device__ __constant__ size_t cellDoFsPos;

using namespace dealii;

class pfem2Particle
{
public:
    pfem2Particle();
	pfem2Particle(const Point<2> & location, const Point<2> & reference_location, const unsigned int id);
		
	void set_location (const Point<2> &new_location);
	const Point<2> get_location () const;
	
	void set_reference_location (const Point<2> &new_reference_location);
	const Point<2> get_reference_location () const;
	
	unsigned int get_id () const;
	
	void set_tria_position (const int &new_position);
	int get_tria_position () const;
	
	void set_velocity (const Tensor<1,2> &new_velocity);
	void set_velocity_component (const double value, int component);
	
	const Tensor<1,2> get_velocity_ext() const;
	void set_velocity_ext (const Tensor<1,2> &new_ext_velocity);
	
	const Tensor<1,2> get_velocity() const;
	double get_velocity_component(int component) const;
	
	Triangulation<2>::cell_iterator get_surrounding_cell(const Triangulation<2> &triangulation) const;	
	DoFHandler<2>::cell_iterator get_surrounding_cell(const Triangulation<2> &triangulation, const DoFHandler<2> &dof_handler) const;
	
	void set_cell_dofs(const typename DoFHandler<2>::active_cell_iterator &cell);
	
	unsigned int find_closest_vertex_of_cell(const typename Triangulation<2>::active_cell_iterator &cell, const Mapping<2> &mapping);
	
	std::size_t serialized_size_in_bytes() const;
	
	double location[2];
	double reference_location[2];
	unsigned int id;
	int cell_dofs[4];
	
	int tria_position;

	double velocity[2];						 //!< Скорость, которую переносит частица
	double velocity_ext[2];					 //!< Внешняя скорость (с которой частица переносится)
};

class pfem2ParticleHandler
{
public:
	pfem2ParticleHandler(const parallel::distributed::Triangulation<2> &tria, const Mapping<2> &coordMapping);
	~pfem2ParticleHandler();
	
	void clear_particles();
	
	std::vector<pfem2Particle>::iterator remove_particle(std::vector<pfem2Particle>::iterator particleIndex);
	void insert_particle(pfem2Particle &particle, const typename DoFHandler<2>::active_cell_iterator &cell);
	
	unsigned int n_global_particles() const;
 
    void sort_particles_into_subdomains_and_cells(const DoFHandler<2> &dof_handler);
    
    std::vector<pfem2Particle>::iterator begin();
    std::vector<pfem2Particle>::iterator end();
  
    std::vector<std::set<typename Triangulation<2>::active_cell_iterator>> vertex_to_cells;
    std::vector<std::vector<Tensor<1,2>>> vertex_to_cell_centers;
    
    void initialize_maps();
	
    std::vector<pfem2Particle> particles;
    
private:
    SmartPointer<const parallel::distributed::Triangulation<2>, pfem2ParticleHandler> triangulation;
    SmartPointer<const Mapping<2>,pfem2ParticleHandler> mapping;
    

};

class pfem2Solver
{
public:
	pfem2Solver();
	~pfem2Solver();

	virtual void build_grid() = 0;
	virtual void setup_system() = 0;
	virtual void solveVx(bool correction = false) = 0;
	virtual void solveVy(bool correction = false) = 0;
	virtual void solveP() = 0;
	virtual void output_results(bool predictionCorrection = false, bool exportParticles = false) = 0;
	
	/*!
	 * \brief Процедура первоначального "посева" частиц в ячейках сетки
	 * \param quantities Вектор количества частиц в каждом направлении (в терминах локальных координат)
	 */
	void seed_particles(const std::vector < unsigned int > & quantities);
	
	/*!
	 * \brief Коррекция скоростей частиц по скоростям в узлах сетки
	 * 
	 * Скорости частиц не сбрасываются (!). Для каждого узла сетки вычисляется изменение поля скоростей.
	 * Затем для каждой частицы по 4 узлам ячейки, в которой содержится частица, вычисляется изменение скорости (коэффициенты - значения функций формы)
	 * и посчитанное изменение добавляется к имеющейся скорости частицы.
	 */
	void correct_particles_velocities();
	
	/*!
	 * \brief "Раздача" скоростей с частиц на узлы сетки
	 * 
	 * Скорости в узлах обнуляются, после чего для каждого узла накапливается сумма скоростей от частиц (коэффициенты - значения функций формы)
	 * из ячеек, содержащих этот узел, и сумма коэффициентов. Итоговая скорость каждого узла - частное от деления первой суммы на вторую.
	 */
	void distribute_particle_velocities_to_grid();
	
	/*!
	 * \brief Перемещение частиц по известному полю скоростей в узлах
	 * 
	 * Перемещение происходит в форме 10 шагов (с шагом time_step/10). Предварительно в частицах корректируется и запоминается скорость. А затем на каждом шаге
	 * + обновляется информация о ячейке, которой принадлежит каждая частица (на первом шаге - за предварительного вычисления переносимой скорости);
	 * + вычисляется скорость частиц по скоростям в узлах сетки (на первом шаге - за предварительного вычисления переносимой скорости);
	 * + координаты частицы изменяются согласно формулам метода Эйлера.
	 */
	void move_particles();
	
	void calculate_loads(types::boundary_id patch_id, std::ofstream *out);
	
	double time,time_step;							//!< Шаг решения задачи методом конечных элементов
	int timestep_number;
	
	//!< Вектор решения (на текущем и предыдущем шаге по времени) и прогноза (на текущем шаге)
	TrilinosWrappers::MPI::Vector locally_relevant_solutionVx, locally_relevant_solutionVy, locally_relevant_solutionP,
				locally_relevant_predictionVx, locally_relevant_predictionVy, locally_relevant_old_solutionVx, locally_relevant_old_solutionVy, locally_relevant_old_solutionP;
	
	MPI_Comm mpi_communicator;
		
	parallel::distributed::Triangulation<2> tria;
	MappingQ1<2> mapping;
	
	pfem2ParticleHandler particle_handler;
	FE_Q<2>  			 feV, feP;
	FESystem<2> 		 fe;
	DoFHandler<2>        dof_handlerV, dof_handlerP;
	TimerOutput			 *timer;

	QGauss<2>   quadrature_formula;
	QGauss<1>   face_quadrature_formula;
	
	FEValues<2> feV_values, feP_values;
	FEFaceValues<2> feV_face_values, feP_face_values;
		
	const unsigned int dofs_per_cellV, dofs_per_cellP;
	std::vector<types::global_dof_index> local_dof_indicesV, local_dof_indicesP;
	
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	
	ConditionalOStream 	 pcout;
		
    IndexSet locally_owned_dofsV, locally_owned_dofsP;
    IndexSet locally_relevant_dofsV, locally_relevant_dofsP;
    
    AffineConstraints<double>  constraintsVx, constraintsVy, constraintsP, constraintsPredVx, constraintsPredVy;
	
	const unsigned int n_q_points;
	const unsigned int n_face_q_points;
	
	std::map<unsigned int, unsigned int> wallsAndBodyDoFs;
	std::set<unsigned int> boundaryDoFs;
	//std::map<unsigned int, Point<2>> boundaryDoFsCoords;
	TrilinosWrappers::MPI::Vector node_weights;
	
protected:
	void seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell);
	bool check_cells_for_empty_parts ();
	
	double mu, rho;
	double diam, uMean;
	int xaDoF, xeDoF;
	
private:
	void setCudaConstants();

	std::vector < unsigned int > quantities;
	int particleCount = 0;
	time_t solutionTime, startTime;
};

bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,2> &particle_direction, const std::vector<Tensor<1,2> > &center_directions);

#endif // PFEM2PARTICLE_H

