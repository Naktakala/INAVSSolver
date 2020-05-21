#include "solver.h"

#include "ChiMesh/MeshHandler/chi_meshhandler.h"

#include "chi_log.h"

//###################################################################
/**Initializes the solver.*/
void INAVSSolver::Initialize()
{
  auto& log = ChiLog::GetInstance();
  log.Log(LOG_0) << "Initializing Incompressible Navier-Stokes Solver";

  auto handler = chi_mesh::GetCurrentHandler();
  grid = handler->GetGrid();

  dimensions.reserve(num_dimensions);
  for (int i=0; i<num_dimensions; ++i)
    dimensions.push_back(i);

  //======================================== Setup unknowns
  VELOCITY = uk_man_u.AddUnknown(chi_math::UnknownType::SCALAR);
  PRESSURE = uk_man_p.AddUnknown(chi_math::UnknownType::SCALAR);
  GRAD_P   = uk_man_gradp.AddUnknown((num_dimensions == 2)?
                                      chi_math::UnknownType::VECTOR_2 :
                                      chi_math::UnknownType::VECTOR_3);
  GRAD_U   = uk_man_gradu.AddUnknown(chi_math::UnknownType::VECTOR_N,
                                     (num_dimensions == 2)? 4 : 9);

  if (num_dimensions == 3)
  {
    DUX_DZ = 2;

    DUY_DX = 3;
    DUY_DY = 4;
    DUY_DZ = 5;

    DUZ_DX = 6;
    DUZ_DY = 7;
    DUZ_DZ = 8;
  }

  //======================================== Setup Spatial Discretization
  fv_sdm.AddViewOfLocalContinuum(grid);

  ndof_local_u = fv_sdm.GetNumLocalDOFs(grid ,&uk_man_u);
  ndof_globl_u = fv_sdm.GetNumGlobalDOFs(grid,&uk_man_u);

  ndof_local_p = fv_sdm.GetNumLocalDOFs(grid ,&uk_man_p);
  ndof_globl_p = fv_sdm.GetNumGlobalDOFs(grid,&uk_man_p);

  ndof_local_gradp = fv_sdm.GetNumLocalDOFs(grid ,&uk_man_gradp);
  ndof_globl_gradp = fv_sdm.GetNumGlobalDOFs(grid,&uk_man_gradp);

  ndof_local_gradu = fv_sdm.GetNumLocalDOFs(grid ,&uk_man_gradu);
  ndof_globl_gradu = fv_sdm.GetNumGlobalDOFs(grid,&uk_man_gradu);

  log.Log(LOG_0) << "Number of velocity unknowns = "
                 << ndof_globl_u << " " << ndof_globl_gradu;
  log.Log(LOG_0) << "Number of pressure unknowns = "
                 << ndof_globl_p << " " << ndof_globl_gradp;



  //================================= Create matrices and vectors
  A_u.resize(num_dimensions);
  x_u.resize(num_dimensions);
  x_uold.resize(num_dimensions);
  x_umim.resize(num_dimensions);
  b_u.resize(num_dimensions);

  for (int i=0; i<num_dimensions; ++i)
    A_u[i] = chi_math::PETScUtils::CreateSquareMatrix(ndof_local_u,ndof_globl_u);
  A_pc = chi_math::PETScUtils::CreateSquareMatrix(ndof_local_p,ndof_globl_p);

  std::vector<int> nz_in_diag_A_u;
  std::vector<int> nz_off_diag_A_u;

  std::vector<int> nz_in_diag_A_pc;
  std::vector<int> nz_off_diag_A_pc;

  fv_sdm.BuildSparsityPattern(grid,nz_in_diag_A_u,nz_off_diag_A_u, &uk_man_u);
  fv_sdm.BuildSparsityPattern(grid,nz_in_diag_A_pc,nz_off_diag_A_pc,&uk_man_p);

  for (int i=0; i<num_dimensions; ++i)
    chi_math::PETScUtils::InitMatrixSparsity(A_u[i],nz_in_diag_A_u,nz_off_diag_A_u);
  chi_math::PETScUtils::InitMatrixSparsity(A_pc,nz_in_diag_A_pc,nz_off_diag_A_pc);

  for (int i=0; i<num_dimensions; ++i)
  {
    x_u   [i]  = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);
    x_uold[i]  = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);
    x_umim[i]  = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);
    b_u   [i]  = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);
  }

  x_gradu = chi_math::PETScUtils::CreateVector(ndof_local_gradu,ndof_globl_gradu);

  x_p     = chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_globl_p);
  x_gradp = chi_math::PETScUtils::CreateVector(ndof_local_gradp,ndof_globl_gradp);

  x_pc    = chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_local_p);
  b_p     = chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_globl_p);

  for (int i=0; i<num_dimensions; ++i)
  {
    VecSet(x_u   [i],0.0);
    VecSet(x_uold[i],0.0);
    VecSet(x_umim[i],0.0);
    VecSet(b_u   [i],0.0);
  }

  VecSet(x_gradu,0.0);

  VecSet(x_p    ,0.0);
  VecSet(x_gradp,0.0);
  VecSet(x_pc   ,0.0);
  VecSet(b_p    ,0.0);

  //======================================== Setting up linear solvers
  lin_solver_u.resize(num_dimensions);

  for (int i=0; i<num_dimensions; ++i)
    lin_solver_u[i] = chi_math::PETScUtils::CreateCommonKrylovSolverSetup(
      A_u[i],            // Reference matrix
      "Momentum_solver",  // Solver name
      KSPRICHARDSON,        // Solver type
      PCHYPRE,       // Preconditioner type
      1.0e-1,       // Residual tolerance
      5);         // Maximum number of iterations

  lin_solver_p = chi_math::PETScUtils::CreateCommonKrylovSolverSetup(
    A_pc,            // Reference matrix
    "P_solver",  // Solver name
    KSPCG,        // Solver type
    PCHYPRE,       // Preconditioner type
    1.0e-1,       // Residual tolerance
    30);         // Maximum number of iterations


  if (log.GetVerbosity() == LOG_0VERBOSE_0)
  {
    for (int i=0; i<num_dimensions; ++i)
      KSPMonitorCancel(lin_solver_u[i].ksp);
    KSPMonitorCancel(lin_solver_p.ksp);
  }

  //======================================== Initialize mass fluxes
  mass_fluxes.clear();
  mass_fluxes.resize(grid->local_cells.size());
  bndry_pressures.clear();
  bndry_pressures.resize(grid->local_cells.size());
  cell_bndry_flags.clear();
  cell_bndry_flags.resize(grid->local_cells.size(),false);
  momentum_coeffs.resize(grid->local_cells.size());
  int c=-1;
  for (auto& cell : grid->local_cells)
  {
    ++c;
    mass_fluxes[c]    .resize(cell.faces.size(),0.0);
    bndry_pressures[c].resize(cell.faces.size(),0.0);

    momentum_coeffs[c].a_N_f.resize(cell.faces.size());

    for (auto& face : cell.faces)
    {
      if (grid->IsCellBndry(face.neighbor))
      {
        cell_bndry_flags[cell.local_id] = true;
        //TODO: Add boundary conditions
      }
    }
  }

  log.Log(LOG_0) << "Done initializing Incompressible Navier-Stokes Solver";
}