#ifndef PFEM2PARTICLE_H
#define PFEM2PARTICLE_H

#define PARTICLES_MOVEMENT_STEPS 3
#define MAX_PARTICLES_PER_CELL_PART 10

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <unordered_map>

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
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class pfem2Particle
{
public:
	pfem2Particle(const Point<2> & location,const Point<2> & reference_location,const unsigned int id);
	pfem2Particle(const void*& begin_data);

	void set_location (const Point<2> &new_location);
	const Point<2> & get_location () const;
	
	void set_reference_location (const Point<2> &new_reference_location);
	const Point<2> & get_reference_location () const;
	
	unsigned int get_id () const;
	
	void set_tria_position (const int &new_position);
	
	void set_map_position (const std::unordered_multimap<int, pfem2Particle*>::iterator &new_position);
	const std::unordered_multimap<int, pfem2Particle*>::iterator & get_map_position () const;
	
	void set_velocity (const Tensor<1,2> &new_velocity);
	void set_velocity_component (const double value, int component);
	
	const Tensor<1,2> & get_velocity_ext() const;
	void set_velocity_ext (const Tensor<1,2> &new_ext_velocity);
	
	const Tensor<1,2> & get_velocity() const;
	double get_velocity_component(int component) const;
	
	Triangulation<2>::cell_iterator get_surrounding_cell(const Triangulation<2> &triangulation) const;
	
	unsigned int find_closest_vertex_of_cell(const typename Triangulation<2>::active_cell_iterator &cell, const Mapping<2> &mapping);
	
	std::size_t serialized_size_in_bytes() const;

	void write_data(void*& data) const;

private:
	Point<2> location;
	Point<2> reference_location;
	unsigned int id;

	int tria_position;
	std::unordered_multimap<int, pfem2Particle*>::iterator map_position;

	Tensor<1,2> velocity;						 //!< Скорость, которую переносит частица
	Tensor<1,2> velocity_ext;					 //!< Внешняя скорость (с которой частица переносится)
};

class pfem2ParticleHandler
{
public:
	pfem2ParticleHandler(const parallel::distributed::Triangulation<2> &tria, const Mapping<2> &coordMapping);
	~pfem2ParticleHandler();
	
	void clear();
	void clear_particles();
	
	void remove_particle(const pfem2Particle *particle);
	void insert_particle(pfem2Particle *particle, const typename Triangulation<2>::active_cell_iterator &cell);
	
	unsigned int n_global_particles() const;
 
    unsigned int n_global_max_particles_per_cell() const;
 
    unsigned int n_locally_owned_particles() const;
    
    unsigned int n_particles_in_cell(const typename Triangulation<2>::active_cell_iterator &cell) const;
    
    void sort_particles_into_subdomains_and_cells();

#ifdef DEAL_II_WITH_MPI
	void send_recv_particles(const std::map<unsigned int, std::vector<std::unordered_multimap<int, pfem2Particle*>::iterator>>& particles_to_send,
		std::unordered_multimap<int, pfem2Particle*>& received_particles,
		const std::map<unsigned int, std::vector<typename Triangulation<2>::active_cell_iterator> >& new_cells_for_particles =
		std::map<unsigned int, std::vector<typename Triangulation<2>::active_cell_iterator> >());
#endif
    
    std::unordered_multimap<int, pfem2Particle*>::iterator begin();
    std::unordered_multimap<int, pfem2Particle*>::iterator end();
    
    std::unordered_multimap<int, pfem2Particle*>::iterator particles_in_cell_begin(const typename Triangulation<2>::active_cell_iterator &cell);
    std::unordered_multimap<int, pfem2Particle*>::iterator particles_in_cell_end(const typename Triangulation<2>::active_cell_iterator &cell);
    
	std::vector<std::set<typename Triangulation<2>::active_cell_iterator>> vertex_to_cells;
	std::vector<std::vector<Tensor<1, 2>>> vertex_to_cell_centers;

	void initialize_maps();

private:
    SmartPointer<const parallel::distributed::Triangulation<2>, pfem2ParticleHandler> triangulation;
    SmartPointer<const Mapping<2>,pfem2ParticleHandler> mapping;
    
    std::unordered_multimap<int, pfem2Particle*> particles;
    
    unsigned int global_number_of_particles;
 
    unsigned int global_max_particles_per_cell;
};

/*!
 *  \brief Структура "узел подвижной сетки"
 *  \rem Номер узла сетки (triangulation) - ключ в контейнере узлов сетки
 */
struct meshNode {
	Point<2> coords;	//!< Глобальные координаты узла
	int cellIndex;		//!< Индекс ячейки сетки, которой принадлежит узел
	int dofUindexX;		//!< Индекс степени свободы, соответствующей перемещению узла по x
	int dofUindexY;		//!< Индекс степени свободы, соответствующей перемещению узла по y
	int dofScalarIndex;	//!< Индекс степени свободы для скалярных полей (Vx, Vy, P) - используется при переинтерполяции этих полей
	bool isBoundaryNode;//!< Узел является граничным (для таких узлов только обновляются координаты, но не выполняется обновление ячейки-родителя и переинтерполяция полей)
};

/*!
 *  \brief Структура "граничный узел со степенью свободы"
 *  \rem Номер узла сетки (triangulation) - ключ в контейнере узлов сетки
 */
struct boundaryDoF {
	int dofScalarIndex;	//!< Индекс степени свободы для скалярных полей (Vx, Vy, P)
	int boundaryId;		//!< ID участка границы
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
	virtual void solveU() = 0;
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
		locally_relevant_predictionVx, locally_relevant_predictionVy, locally_relevant_old_solutionVx, locally_relevant_old_solutionVy, locally_relevant_old_solutionP, locally_relevant_solutionU,
		locally_relevant_vorticity;

	MPI_Comm mpi_communicator;
	
	double force_Fy;
	
	parallel::distributed::Triangulation<2> tria;
	MappingQ1<2> mapping;
	
	pfem2ParticleHandler particle_handler;
	FE_Q<2>  			 feV, feP, feU;
	FESystem<2> 		 fe, fe2d;
	DoFHandler<2>        dof_handlerV, dof_handlerP, dof_handlerU;
	TimerOutput			 *timer;
	
	std::vector<unsigned int> probeDoFnumbers;

	QGauss<2>   quadrature_formula;
	QGauss<1>   face_quadrature_formula;

	FEValues<2> feV_values, feP_values, feU_values;
	FEFaceValues<2> feV_face_values, feP_face_values;

	const unsigned int dofs_per_cellV, dofs_per_cellP, dofs_per_cellU;
	std::vector<types::global_dof_index> local_dof_indicesV, local_dof_indicesP, local_dof_indicesU;

	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;

	ConditionalOStream 	 pcout;

	IndexSet locally_owned_dofsV, locally_owned_dofsP, locally_owned_dofsU;
	IndexSet locally_relevant_dofsV, locally_relevant_dofsP, locally_relevant_dofsU;

	AffineConstraints<double>  constraintsVx, constraintsVy, constraintsP, constraintsPredVx, constraintsPredVy, constraintsU;

	const unsigned int n_q_points;
	const unsigned int n_face_q_points;

	std::map<unsigned int, boundaryDoF> wallsAndBodyDoFs;
	std::map<unsigned int, boundaryDoF> DirichletBoundaryDoFs;

	double body_y_, body_velVy_;
	
protected:
	void seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell);
	bool check_cell_for_empty_parts (const typename DoFHandler<2>::cell_iterator &cell);
	
	double mu, rho;
	double diam, uMean;
	int xaDoF, xeDoF;
	
	ParameterHandler prm;

	void check_empty_cells();	//!< Проверка наличия пустых ячеек (без частиц) и размещение в них частиц
	
	void calculate_vorticity();
	
	//работа с подвижной сеткой
	/*!
	 *  \brief Инициализация уникального набора узлов подвижной сетки
	 * 
	 *  Узел идентифицируется в контейнере по номеру вершины в сетке, которую он представляет. Заполняются
	 * - геометрические координаты;
	 * - индекс (номер) ячейки, которой узел принадлежит (первой по счету, в которой он встретился);
	 * - номера степеней свободы для поля перемещений по x и y, которые соответствуют узлу
	 */
	void initialize_moving_mesh_nodes();
	
	/*!
	 * \brief Переинтерполяция полей (V, p) в новые положения узлов сетки на основе значений в старых положениях
	 *
	 * Для каждого узла (кроме граничных) выполняется
	 * - поиск ячейки, в которой он окажется после перемещения (с большой долей вероятности - либо прежняя ячейка, либо одна из соседних);
	 * - вычисление локальных координат (как часть процедуры поиска ячейки для узла);
	 * - нахождение новых значений полей путем интерполяции значений в узлах найденной ячейки сетки с помощью функций формы.
	 *
	 * Только после обработки всех узлов найденные значения полей переписываются в векторы решений.
	 */
	void reinterpolate_fields();
	
	/*!
	 *  \brief Изменение координат узлов сетки (Triangulation) в соответствии с полями перемещений по x и y.
	 * 
	 * Дополнительно обновляются координаты в структурах узлов подвижной сетки.
	 * Операция выполняется в самом конце шага, после переинтерполяции полей в новые координаты узлов, но до конца переупорядочивания частиц.
	 */
	void transform_grid();
	
private:	
	std::vector < unsigned int > quantities;
	int particleCount = 0;
	time_t solutionTime, startTime;
	
	std::map<int, meshNode> movingNodes;
	std::map<int, DoFHandler<2>::active_cell_iterator> dofHandlerCells;
};

bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,2> &particle_direction, const std::vector<Tensor<1,2> > &center_directions);

#endif // PFEM2PARTICLE_H

