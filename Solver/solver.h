#ifndef _solver_h
#define _solver_h

#include "ChiMesh/MeshContinuum/chi_meshcontinuum.h"
#include "ChiMath/PETScUtils/petsc_utils.h"
#include "ChiMath/UnknownManager/unknown_manager.h"
#include "ChiMath/SpatialDiscretization/FiniteVolume/fv.h"

#include "ChiMath/chi_math_vectorNX.h"
#include "ChiMath/chi_math_tensorRNX.h"

//###################################################################
template<int NDD>
class INAVSSolver
{
public:
  struct Options
  {
    bool steady = true;

    //Gradient options
    enum class GradientMethod {
      GreenGauss=1, ///< Simple Green-Gauss
      WLSQ=2        ///< Weighted least-squares
      };
    enum class GradientLimiter {
      None                    = 1,
      Venkatakrishnan         = 2,
      MinMod                  = 3,
      ModifiedVenkatakrishnan = 4
    };
    GradientMethod  gradient_method = GradientMethod::WLSQ;
    GradientLimiter gradient_limiter = GradientLimiter::Venkatakrishnan;

  }options;

  const int num_dimensions;
  std::vector<int> dimensions;

  //=================================== Constants
  const double TWO_THIRDS = 2.0/3.0;
  const bool COMPUTE_GRADU = true;
  const bool COMPUTE_MF    = false;

  unsigned int VELOCITY = 0;
  unsigned int PRESSURE = 0;
  unsigned int GRAD_P = 0;
  unsigned int GRAD_U = 0;
  unsigned int PROPERTY = 0;

  const int U_X = 0, U_Y = 1, U_Z = 2;

  //=================================== Dyad to linear indexing
  int DUX_DX =  0, DUX_DY =  1, DUX_DZ = -1;
  int DUY_DX =  2, DUY_DY =  3, DUY_DZ = -1;
  int DUZ_DX = -1, DUZ_DY = -1, DUZ_DZ = -1;

  const int P_X = 0, P_Y = 1, P_Z = 2;

  const chi_mesh::Vector3 I_HAT = chi_mesh::Vector3(1.0, 0.0, 0.0);
  const chi_mesh::Vector3 J_HAT = chi_mesh::Vector3(0.0, 1.0, 0.0);
  const chi_mesh::Vector3 K_HAT = chi_mesh::Vector3(0.0, 0.0, 1.0);
  const chi_mesh::Vector3 VEC3_ONES = chi_mesh::Vector3(1.0, 1.0, 1.0);

  //=================================== Unknown managers
public:
  chi_math::UnknownManager uk_man_u;
  chi_math::UnknownManager uk_man_p;
  chi_math::UnknownManager uk_man_gradp;
  chi_math::UnknownManager uk_man_gradu;
  chi_math::UnknownManager uk_man_props;

  //=================================== Grid
public:
  chi_mesh::MeshContinuum* grid= nullptr;

  //=================================== PETSc matrices and vectors
public:
  // Matrices velocity and pressure
  std::vector<Mat> A_u;
  Mat              A_pc;

  // Velocity vectors
  std::vector<Vec> x_u;
  std::vector<Vec> x_uold;
  std::vector<Vec> x_umim;
  std::vector<Vec> x_a_P;
  std::vector<Vec> b_u;
  Vec              x_gradu;

  // Pressure
  Vec              x_p;
  Vec              x_gradp;

  // Pressure correction
  Vec              x_pc;
  Vec              x_gradpc;
  Vec              b_pc;

  // Cell properties
  Vec              x_V;   //Cell Volume
  Vec              x_rho; //Density
  Vec              x_mu;  //Viscosity

  //=================================== PETSc Solvers
public:
  std::vector<chi_math::PETScUtils::PETScSolverSetup> lin_solver_u;
  chi_math::PETScUtils::PETScSolverSetup lin_solver_p;

  //=================================== Discretization items
public:
  SpatialDiscretization_FV fv_sdm;

  int ndof_local_u=0;
  int ndof_globl_u=0;
  int ndof_local_p=0;
  int ndof_globl_p=0;
  int ndof_local_gradp=0;
  int ndof_globl_gradp=0;
  int ndof_local_gradu=0;
  int ndof_globl_gradu=0;

  int ndof_ghost_u=0;
  int ndof_ghost_p=0;
  int ndof_ghost_gradp=0;
  int ndof_ghost_gradu=0;

  std::vector<int> ghost_ids_u;
  std::vector<int> ghost_ids_p;
  std::vector<int> ghost_ids_gradp;
  std::vector<int> ghost_ids_gradu;


  //=================================== Cell momentum coefficients
public:
  struct CellInfo
  {
    chi_math::VectorN<NDD> a_t;
    chi_math::VectorN<NDD> a_P = chi_mesh::Vector3(1.0,1.0,1.0);
    chi_math::VectorN<NDD> b_P;
    std::vector<chi_math::VectorN<NDD>> a_N_f;
  };
  std::vector<CellInfo> cell_info;

  //=================================== Utilities
public:
  std::vector<std::vector<double>> mass_fluxes;
  std::vector<std::vector<double>> bndry_pressures;
  std::vector<bool> cell_bndry_flags;

  //=================================== Timing tags
  size_t tag_gradP_gg;
  size_t tag_gradU;
  size_t tag_mom_assy;
  size_t tag_mom_slv1;
  size_t tag_comp_mf;
  size_t tag_pc_assy;
  size_t tag_pc_slv1;
  size_t tag_gradP_pc;
  size_t tag_corr;

  //=================================== Methods
public:
       INAVSSolver(int nd=2) : num_dimensions(nd)
       {}
  void Initialize();
  void InitProperties();
  void Execute();

  void ComputeGradP_GreenGauss(Vec v_gradp, Vec v_p);
  void ComputeGradP_WLSQ(Vec v_gradp, Vec v_p,bool limited=false);
  void ComputeMassFluxMMIM();
  void ComputeGradU_WLSQ(bool limited=false);
  void AssembleMomentumSystem();

  void AssembleSolvePressureCorrectionSystem();
  void ComputeCorrections();
};

#endif 