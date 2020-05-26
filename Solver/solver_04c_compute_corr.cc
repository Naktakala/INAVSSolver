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
  //============================================= Get local views
  std::vector<Vec> x_a_PL(3);
  Vec x_gradpL;
  Vec x_pcL;
  for (int dim : dimensions)
    VecGhostGetLocalForm(x_a_P[dim],&x_a_PL[dim]);
  VecGhostGetLocalForm(x_gradp,&x_gradpL);
  VecGhostGetLocalForm(x_pc,&x_pcL);

  //============================================= Loop over cells
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V = cell_fv_view->volume;

    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    int ip            = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

    std::vector<int> igradp(3,-1);
    for (int dim : dimensions)
      igradp[dim] = fv_sdm.MapDOF(&cell,&uk_man_gradp,GRAD_P,dim);

    //====================================== Get previous iteration info
    double p_P;
    chi_mesh::Vector3 gradp_P;
    chi_mesh::Vector3 a_P;

    VecGetValues(x_pc,1,&ip,&p_P);
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
    for (int dim : dimensions)
      VecGetValues(x_a_P[dim] , 1, &iu, &a_P(dim));

    //====================================== Declare velocity correction
    chi_mesh::Vector3 uc;

    //====================================== Face terms
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor >= 0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        //============================= Map row indices of unknowns
        int lj0  = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int ljp  = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> ljgradp(3,-1);
        for (int dim : dimensions)
          ljgradp[dim] =
            fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        //============================= Get previous iteration info
        double p_N;
        chi_mesh::Vector3 gradp_N;
        chi_mesh::Vector3 a_N;

        VecGetValues(x_pcL, 1, &ljp, &p_N);
        VecGetValues(x_gradpL,num_dimensions,ljgradp.data(),&gradp_N(0));
        for (int dim : dimensions)
          VecGetValues(x_a_PL[dim] ,1,&lj0,&a_N(dim));


        //============================= Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //============================= Compute interpolated values
        double delta_p                 = p_N - p_P;
        chi_mesh::Vector3 a_f          = (1.0-rP)*a_P     + rP*a_N;
        chi_mesh::Vector3 grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;

        chi_mesh::Vector3 a_f_inv = a_f.InverseZeroIfSmaller(1.0e-10);

        //============================= Compute grad_p_f
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
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
      else
      {
        auto ds = face.centroid - cell.centroid;

        auto a_P_inv = a_P.InverseZeroIfSmaller(1.0e-10);

        double p_avg = p_P + gradp_P.Dot(ds);

        chi_mesh::Vector3 Dp_dot_gradpc = (alpha_u*a_P_inv)*A_f*n*p_avg;

        uc = uc - Dp_dot_gradpc;
      }
    }//for faces

    //====================================== Set vector values
    for (auto dim : dimensions)
      VecSetValue(x_u[dim], iu, uc[dim], ADD_VALUES);

  }//for cell

  //============================================= Restore local view
  for (int dim : dimensions)
    VecGhostRestoreLocalForm(x_a_P[dim],&x_a_PL[dim]);
  VecGhostRestoreLocalForm(x_gradp,&x_gradpL);
  VecGhostRestoreLocalForm(x_pc,&x_pcL);

  //============================================= Assemble x_u
  for (int i=0; i<num_dimensions; ++i)
  {
    VecAssemblyBegin(x_u[i]);
    VecAssemblyEnd(x_u[i]);
  }
  VecAXPY(x_p,alpha_p,x_pc);

  //============================================= Scatter forward
  for (int dim : dimensions)
    VecGhostUpdateBegin(x_u[dim],INSERT_VALUES,SCATTER_FORWARD);
  for (int dim : dimensions)
    VecGhostUpdateEnd  (x_u[dim],INSERT_VALUES,SCATTER_FORWARD);

  VecGhostUpdateBegin(x_p,INSERT_VALUES,SCATTER_FORWARD);
  VecGhostUpdateEnd  (x_p,INSERT_VALUES,SCATTER_FORWARD);

}