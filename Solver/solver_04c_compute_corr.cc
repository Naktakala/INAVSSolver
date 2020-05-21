#include "solver.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/***/
void INAVSSolver::ComputeCorrections()
{
  VecAXPY(x_p,alpha_p,x_pc);

  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V = cell_fv_view->volume;

    auto& a_P = momentum_coeffs[cell.local_id].a_P;

    //======================================= Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(cell.global_id, &uk_man_u, VELOCITY);
    int ip            = fv_sdm.MapDOF(cell.global_id, &uk_man_p, PRESSURE);

    std::vector<int> igradp(3,-1);
    igradp[P_X]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_X);
    igradp[P_Y]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Y);
    igradp[P_Z]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Z);

    //======================================= Get previous iteration info
    double p_P;
    chi_mesh::Vector3 gradp_P;

    VecGetValues(x_pc,1,&ip,&p_P);
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));

    //======================================= Declare velocity correction
    chi_mesh::Vector3 uc;

    //======================================= Face terms
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n = face.normal;

      if (not grid->IsCellBndry(face.neighbor))
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto& a_N = momentum_coeffs[adj_cell->local_id].a_P;

        //======================================= Map row indices of unknowns
        int jp   = fv_sdm.MapDOF(face.neighbor, &uk_man_p, PRESSURE);

        std::vector<int> jgradp(3,-1);
        jgradp[P_X] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_X);
        jgradp[P_Y] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Y);
        jgradp[P_Z] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Z);

        //======================================= Get previous iteration info
        double p_N;
        chi_mesh::Vector3 gradp_N;

        VecGetValues(x_pc, 1, &jp, &p_N);
        VecGetValues(x_gradp, num_dimensions, jgradp.data(), &gradp_N(0));

        //================================== Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //======================================= Compute interpolated values
        double delta_p                 = p_N - p_P;
        chi_mesh::Vector3 a_f          = (1.0-rP)*a_P     + rP*a_N;
        chi_mesh::Vector3 grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;

        chi_mesh::Vector3 a_f_inv = a_f.InverseZeroIfSmaller(1.0e-10);

        //======================================= Compute grad_p_f
        chi_mesh::Vector3 grad_pc_f = (delta_p/d_PN)*e_PN + grad_p_f_avg -
                                      grad_p_f_avg.Dot(e_PN)*e_PN;

        double m_f_old = mass_fluxes[cell.local_id][f];

        mass_fluxes[cell.local_id][f] =
          m_f_old - rho*alpha_u*V*A_f*n.Dot(a_f_inv*grad_pc_f);

        chi_mesh::Vector3 p_avg;
        for (auto dim : dimensions)
          p_avg(dim) = (p_P / a_P[dim] + p_N / a_N[dim]) /
                       (1.0/a_P[dim] + 1.0/a_N[dim]);

        chi_mesh::Vector3 Dp_dot_gradpc = (alpha_u*a_f_inv)*A_f*n*p_avg;

        uc = uc - Dp_dot_gradpc;
      }//interior face
      else
      {
        auto ds = face.centroid - cell.centroid;

        auto a_P_inv = a_P.InverseZeroIfSmaller(1.0e-10);

        double p_avg = p_P + gradp_P.Dot(ds);

        chi_mesh::Vector3 Dp_dot_gradpc = (alpha_u*a_P_inv)*A_f*n*p_avg;

        uc = uc - Dp_dot_gradpc;
      }
    }//for faces

    for (auto dim : dimensions)
      VecSetValue(x_u[dim], iu, uc[dim], ADD_VALUES);

  }//for cell

  for (int i=0; i<num_dimensions; ++i)
  {
    VecAssemblyBegin(x_u[i]);
    VecAssemblyEnd(x_u[i]);
  }

//  VecAXPY(x_p,alpha_p,x_pc);
}