#include "solver.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/** Computes the gradient of the velocity where face velocities
 * are interpolated using the momentum equation.*/
void INAVSSolver::ComputeGradU()
{
  VecSet(x_gradu,0.0);

  //================================================== Compute MIM velocities
  for (auto& cell : grid->local_cells)
  {
    auto& cell_mom_coeffs = momentum_coeffs[cell.local_id];

    //======================================= Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);

    //======================================= Declare H_P coefficients
    chi_mesh::Vector3 H_P;

    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      if (face.neighbor>=0)
      {
        auto& a_N_f = cell_mom_coeffs.a_N_f[f];

        int ju = fv_sdm.MapDOF(face.neighbor, &uk_man_u, VELOCITY);

        chi_mesh::Vector3 u_N;
        for (int dim : dimensions)
          VecGetValues(x_u[dim], 1, &ju, &u_N(dim));

        H_P = H_P - a_N_f*u_N;
      }
    }

    chi_mesh::Vector3 u_mim = H_P + cell_mom_coeffs.b_P;

    for (int dim : dimensions)
      VecSetValue(x_umim[dim], iu, u_mim[dim], INSERT_VALUES);
  }//for cells

  for (int i=0; i<num_dimensions; ++i)
  {
    VecAssemblyBegin(x_umim[i]);
    VecAssemblyEnd(x_umim[i]);
  }


  //============================================= Compute the gradient
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    auto& a_P = momentum_coeffs[cell.local_id].a_P;

    //======================================= Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    int ip            = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

    std::vector<int> igradp(3,-1);
    igradp[P_X]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_X);
    igradp[P_Y]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Y);
    igradp[P_Z]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Z);

    std::vector<int> igrad_ux(num_dimensions,-1);
    std::vector<int> igrad_uy(num_dimensions,-1);
    std::vector<int> igrad_uz(num_dimensions,-1);

    for (int dim : dimensions)
    {
      igrad_ux[dim] = fv_sdm.MapDOF(cell.global_id,&uk_man_gradu,GRAD_U, DUX_DX+dim);
      igrad_uy[dim] = fv_sdm.MapDOF(cell.global_id,&uk_man_gradu,GRAD_U, DUY_DX+dim);
      igrad_uz[dim] = fv_sdm.MapDOF(cell.global_id,&uk_man_gradu,GRAD_U, DUZ_DX+dim);
    }

    //======================================= Get previous iteration info
    double p_m = 0.0;
    chi_mesh::Vector3 gradp_P;
    chi_mesh::Vector3 u_mim_P;

    VecGetValues(x_p,1,&ip,&p_m);
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
    for (int dim : dimensions)
      VecGetValues(x_umim[dim], 1, &iu, &u_mim_P(dim));

    //======================================= Declare grad coefficients
    chi_mesh::Vector3 a_grad_ux;
    chi_mesh::Vector3 a_grad_uy;
    chi_mesh::Vector3 a_grad_uz;

    //======================================= Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      double             A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n   = face.normal;

      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto adj_cell_fv_view = fv_sdm.MapFeView(adj_cell->local_id); //TODO: !!

        double V_N = adj_cell_fv_view->volume;

        auto& a_N = momentum_coeffs[adj_cell->local_id].a_P;

        //======================================= Map row indices of unknowns
        int j0            = fv_sdm.MapDOF(adj_cell,&uk_man_u,VELOCITY);
        int jp_p          = fv_sdm.MapDOF(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> jgradp(3,-1);
        jgradp[P_X] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_X);
        jgradp[P_Y] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Y);
        jgradp[P_Z] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Z);

        //======================================= Get previous iteration info
        double p_p;
        chi_mesh::Vector3 gradp_N;
        chi_mesh::Vector3 u_mim_N;

        VecGetValues(x_p,1,&jp_p,&p_p);
        VecGetValues(x_gradp, num_dimensions, jgradp.data(), &gradp_N(0));
        for (int dim : dimensions)
          VecGetValues(x_umim[dim],1,&j0,&u_mim_N(dim));

        //================================== Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //======================================= Compute interpolated values
        double dp                      = p_p - p_m;
        chi_mesh::Vector3 u_mim_f      = (1.0-rP)*u_mim_P + rP*u_mim_N;
        chi_mesh::Vector3 a_f          = (1.0-rP)*a_P     + rP*a_N;
        double            V_f          = (1.0-rP)*V_P     + rP*V_N;
        chi_mesh::Vector3 grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;

        chi_mesh::Vector3 a_f_inv = a_f.InverseZeroIfSmaller(1.0e-10);

        //======================================= Compute grad_p_f
        chi_mesh::Vector3 grad_p_f = (dp/d_PN)*e_PN + grad_p_f_avg -
                                     grad_p_f_avg.Dot(e_PN)*e_PN;

        //======================================= Compute face velocities
        chi_mesh::Vector3 u_f = (alpha_u*a_f_inv)*(u_mim_f - V_f * grad_p_f);

        for (int dim : dimensions)
        {
          a_grad_ux(dim) += A_f*n[dim]*u_f[U_X];
          a_grad_uy(dim) += A_f*n[dim]*u_f[U_Y];
          a_grad_uz(dim) += A_f*n[dim]*u_f[U_Z];
        }
      }//not bndry
      else
      {
        if (face.normal.Dot(chi_mesh::Vector3(0.0,1.0,0.0))>0.999)
        {
          for (int dim : dimensions)
          {
            a_grad_ux(dim) += A_f*n[dim]*U;
            a_grad_uy(dim) += A_f*n[dim]*0.0;
            a_grad_uz(dim) += A_f*n[dim]*0.0;
          }
        }//if north face
      }//bndry
    }//for faces

    for (int dim : dimensions)
    {
      VecSetValue(x_gradu,igrad_ux[dim],a_grad_ux[dim]/V_P,ADD_VALUES);
      VecSetValue(x_gradu,igrad_uy[dim],a_grad_uy[dim]/V_P,ADD_VALUES);
      VecSetValue(x_gradu,igrad_uz[dim],a_grad_uz[dim]/V_P,ADD_VALUES);
    }
  }//for cells

  VecAssemblyBegin(x_gradu);
  VecAssemblyEnd(x_gradu);
}