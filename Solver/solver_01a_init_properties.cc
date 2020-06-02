#include "solver.h"

#include "ChiMesh/MeshHandler/chi_meshhandler.h"

#include "chi_log.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/**Initializes the cell and material properties.*/
void INAVSSolver::InitProperties()
{
  for (auto& cell : grid->local_cells)
  {
    auto   cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    int i = fv_sdm.MapDOF(&cell,&uk_man_props,PROPERTY);

    VecSetValue(x_V  ,i,V_P,INSERT_VALUES);
    VecSetValue(x_rho,i,rho,INSERT_VALUES);
    VecSetValue(x_mu,i,mu,INSERT_VALUES);
  }

  VecAssemblyBegin(x_V);
  VecAssemblyBegin(x_rho);
  VecAssemblyBegin(x_mu);

  VecAssemblyEnd(x_V);
  VecAssemblyEnd(x_rho);
  VecAssemblyEnd(x_mu);

  chi_math::PETScUtils::CommunicateGhostEntries(x_V);
  chi_math::PETScUtils::CommunicateGhostEntries(x_rho);
  chi_math::PETScUtils::CommunicateGhostEntries(x_mu);
}