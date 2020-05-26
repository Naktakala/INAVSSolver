#include "solver.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/***/
void INAVSSolver::ComputeMassFlux()
{
  //============================================= Reset face mass flux
  for (auto& face_mass_fluxes : mass_fluxes)
    for (auto& val : face_mass_fluxes)
      val = 0.0;

  //============================================= Get local view
  std::vector<Vec> x_uL(3);
  std::vector<Vec> x_umimL(3);
  std::vector<Vec> x_a_PL(3);
  Vec x_gradpL;
  Vec x_pL;
  for (int dim : dimensions)
  {
    VecGhostGetLocalForm(x_u[dim],&x_uL[dim]);
    VecGhostGetLocalForm(x_umim[dim],&x_umimL[dim]);
    VecGhostGetLocalForm(x_a_P[dim],&x_a_PL[dim]);
  }
  VecGhostGetLocalForm(x_p,&x_pL);
  VecGhostGetLocalForm(x_gradp,&x_gradpL);

  //============================================= Compute MIM velocities
  for (auto& cell : grid->local_cells)
  {
    auto& cell_mom_coeffs = momentum_coeffs[cell.local_id];

    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);

    //====================================== Declare H_P coefficients
    chi_mesh::Vector3 H_P;

    //====================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        int lju = fv_sdm.MapDOFLocal(adj_cell, &uk_man_u, VELOCITY);

        auto& a_N_f = cell_mom_coeffs.a_N_f[f];

        chi_mesh::Vector3 u_N;
        for (auto dim : dimensions)
          VecGetValues(x_uL[dim], 1, &lju, &u_N(dim));

        H_P = H_P - a_N_f*u_N;
      }
    }

    //====================================== Add b coefficient
    chi_mesh::Vector3 u_mim = H_P + cell_mom_coeffs.b_P;

    //====================================== Set vector values
    for (auto dim : dimensions)
      VecSetValue(x_umim[dim], iu, u_mim[dim], INSERT_VALUES);
  }//for cells

  //============================================= Assemble x_umim
  for (int dim : dimensions)
  {
    VecAssemblyBegin(x_umim[dim]);
    VecAssemblyEnd(x_umim[dim]);
  }

  //============================================= Scatter x_umim
  for (int dim : dimensions)
    VecGhostUpdateBegin(x_umim[dim],INSERT_VALUES,SCATTER_FORWARD);
  for (int dim : dimensions)
    VecGhostUpdateEnd  (x_umim[dim],INSERT_VALUES,SCATTER_FORWARD);

  //============================================= Compute the fluxes
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    int ip            = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

    std::vector<int> igradp(3,-1);
    for (int dim : dimensions)
      igradp[dim] = fv_sdm.MapDOF(&cell,&uk_man_gradp,GRAD_P,dim);

    //====================================== Get previous iteration info
    double p_P = 0.0;
    chi_mesh::Vector3 gradp_P;
    chi_mesh::Vector3 u_mim_P;
    chi_mesh::Vector3 a_P;

    VecGetValues(x_p,1,&ip,&p_P);
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
    for (auto dim : dimensions)
    {
      VecGetValues(x_umim[dim], 1, &iu, &u_mim_P(dim));
      VecGetValues(x_a_P[dim] , 1, &iu, &a_P(dim));
    }

    //====================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      double             A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n   = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto adj_cell_fv_view = fv_sdm.MapNeighborFeView(adj_cell->global_id); //TODO: !!

        double V_N = adj_cell_fv_view->volume;

        //============================= Map row indices of unknowns
        int lj0    = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int ljp_p  = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> ljgradp(3,-1);
        for (int dim : dimensions)
          ljgradp[dim] =
            fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        //============================= Get previous iteration info
        double p_N;
        chi_mesh::Vector3 gradp_N;
        chi_mesh::Vector3 u_mim_N;
        chi_mesh::Vector3 a_N;

        VecGetValues(x_pL,1,&ljp_p,&p_N);
        VecGetValues(x_gradpL, num_dimensions, ljgradp.data(), &gradp_N(0));
        for (auto dim : dimensions)
        {
          VecGetValues(x_umimL[dim],1,&lj0,&u_mim_N(dim));
          VecGetValues(x_a_PL[dim] ,1,&lj0,&a_N(dim));
        }

        //============================= Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //============================= Compute interpolated values
        double dp                      = p_N - p_P;
        chi_mesh::Vector3 u_mim_f      = (1.0-rP)*u_mim_P + rP*u_mim_N;
        chi_mesh::Vector3 a_f          = (1.0-rP)*a_P     + rP*a_N;
        double            V_f          = (1.0-rP)*V_P     + rP*V_N;
        chi_mesh::Vector3 grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;

        chi_mesh::Vector3 a_f_inv = a_f.InverseZeroIfSmaller(1.0e-10);

        //============================= Compute grad_p_f
        chi_mesh::Vector3 grad_p_f = (dp/d_PN)*e_PN + grad_p_f_avg -
                                     grad_p_f_avg.Dot(e_PN)*e_PN;

        //============================= Compute face velocities
        chi_mesh::Vector3 u_f = (alpha_u*a_f_inv)*(u_mim_f - V_f * grad_p_f);

        mass_fluxes[cell.local_id][f] = rho*A_f*n.Dot(u_f);
      }//not bndry
    }//for faces
  }//for cells

  //============================================= Restore local views
  for (int dim : dimensions)
  {
    VecGhostRestoreLocalForm(x_u[dim],&x_uL[dim]);
    VecGhostRestoreLocalForm(x_umim[dim],&x_umimL[dim]);
    VecGhostRestoreLocalForm(x_a_P[dim],&x_a_PL[dim]);
  }
  VecGhostRestoreLocalForm(x_p,&x_pL);
  VecGhostRestoreLocalForm(x_gradp,&x_gradpL);
}