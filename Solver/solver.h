#ifndef _solver_h
#define _solver_h

#include "ChiMesh/MeshContinuum/chi_meshcontinuum.h"
#include "ChiMath/PETScUtils/petsc_utils.h"
#include "ChiMath/UnknownManager/unknown_manager.h"
#include "ChiMath/SpatialDiscretization/FiniteVolume/fv.h"

//###################################################################
class INAVSSolver
{
public:
  struct
  {
    bool steady = true;
  }options;

  int num_dimensions = 2;
  std::vector<int> dimensions;
  const unsigned int U_X = 0;
  const unsigned int U_Y = 1;
  const unsigned int U_Z = 2;

  unsigned int DUX_DX = 0;
  unsigned int DUX_DY = 1;
  unsigned int DUX_DZ = -1;

  unsigned int DUY_DX = 2;
  unsigned int DUY_DY = 3;
  unsigned int DUY_DZ = -1;

  unsigned int DUZ_DX = -1;
  unsigned int DUZ_DY = -1;
  unsigned int DUZ_DZ = -1;

  const unsigned int P_X = 0;
  const unsigned int P_Y = 1;
  const unsigned int P_Z = 2;

public:
  chi_mesh::MeshContinuum* grid= nullptr;
public:
  std::vector<Mat> A_u;
  Mat A_pc;

  std::vector<Vec> x_u;
  std::vector<Vec> x_uold;
  std::vector<Vec> x_umim;
  std::vector<Vec> b_u;
  Vec x_gradu;

  Vec x_p;
  Vec x_gradp;

  Vec x_pc;
  Vec b_p;

public:
  chi_math::UnknownManager uk_man_u;
  chi_math::UnknownManager uk_man_p;
  chi_math::UnknownManager uk_man_gradp;
  chi_math::UnknownManager uk_man_gradu;

  unsigned int VELOCITY = 0;
  unsigned int PRESSURE = 0;
  unsigned int GRAD_P = 0;
  unsigned int GRAD_U = 0;

public:
  SpatialDiscretization_FV fv_sdm;

  unsigned int ndof_local_u=0;
  unsigned int ndof_globl_u=0;
  unsigned int ndof_local_p=0;
  unsigned int ndof_globl_p=0;
  unsigned int ndof_local_gradp=0;
  unsigned int ndof_globl_gradp=0;
  unsigned int ndof_local_gradu=0;
  unsigned int ndof_globl_gradu=0;

public:
  std::vector<chi_math::PETScUtils::PETScSolverSetup> lin_solver_u;

  chi_math::PETScUtils::PETScSolverSetup lin_solver_p;

public:
  std::vector<std::vector<double>> mass_fluxes;
  std::vector<std::vector<double>> bndry_pressures;
  std::vector<bool> cell_bndry_flags;
  struct CellMomemtumCoefficients
  {
//    double a_t_m = 0.0;
//    double a_x_m = 0.0;
//    double a_y_m = 0.0;
//    double a_z_m = 0.0;
//
//    double b_x = 0.0;
//    double b_y = 0.0;
//
//    std::vector<double> a_x_p;
//    std::vector<double> a_y_p;
//    std::vector<double> a_z_p;

    chi_mesh::Vector3 a_t;
    chi_mesh::Vector3 a_P;
    chi_mesh::Vector3 b_P;
    std::vector<chi_mesh::Vector3> a_N_f;
  };
  std::vector<CellMomemtumCoefficients> momentum_coeffs;

public:
  void Initialize();
  void Execute();

  void ComputeGradP_GG(Vec v_gradp, Vec v_p);
  void ComputeGradU();
  void AssembleMomentumSystem();

  void ComputeMassFlux();
  void AssemblePressureCorrectionSystem();
  void ComputeCorrections();

};

#endif 