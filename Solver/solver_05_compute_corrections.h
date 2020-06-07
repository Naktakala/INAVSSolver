#include "solver.h"

#include "chi_log.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/***/
template<int NDD>
void INAVSSolver<NDD>::ComputeCorrections()
{
  auto& log = ChiLog::GetInstance();
  log.LogEvent(tag_corr,ChiLog::EventType::EVENT_BEGIN);

  //============================================= Get local views
  std::vector<Vec> x_a_PL(NDD);
  Vec x_gradpL;
  Vec x_pcL;
  for (int dim : dimensions)
    VecGhostGetLocalForm(x_a_P[dim],&x_a_PL[dim]);
  VecGhostGetLocalForm(x_gradpc,&x_gradpL);
  VecGhostGetLocalForm(x_pc,&x_pcL);

  std::vector<const double*> d_a_PL(NDD);
  const double* d_gradpL;
  const double* d_pcL;
  for (int dim : dimensions)
    VecGetArrayRead(x_a_PL[dim],&d_a_PL[dim]);
  VecGetArrayRead(x_pcL,&d_pcL);
  VecGetArrayRead(x_gradpL,&d_gradpL);

  //============================================= Loop over cells
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    int ip            = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

    std::vector<int> igradp(NDD,-1);
    for (int dim : dimensions)
      igradp[dim] = fv_sdm.MapDOF(&cell,&uk_man_gradp,GRAD_P,dim);

    //====================================== Get previous iteration info
    double                 p_P;
    chi_math::VectorN<NDD> gradp_P;
    chi_math::VectorN<NDD> a_P;
    double                 a_P_avg;

    VecGetValues(x_pc,1,&ip,&p_P);
    VecGetValues(x_gradpc, num_dimensions, igradp.data(), &gradp_P(0));
    for (int dim : dimensions)
      VecGetValues(x_a_P[dim],1,&iu,&a_P(dim));
    for (int i : dimensions)
      a_P_avg += a_P[i];
    a_P_avg /= num_dimensions;

    //====================================== Declare velocity correction
    chi_math::VectorN<NDD> uc;

    //====================================== Face terms
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_math::VectorN<NDD> n = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor >= 0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto adj_cell_fv_view = fv_sdm.MapNeighborFeView(face.neighbor);

        double V_N = adj_cell_fv_view->volume;

        //============================= Map row indices of unknowns
        int lj0  = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int ljp  = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> ljgradp(NDD,-1);
        for (int dim : dimensions)
          ljgradp[dim] =
            fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        //============================= Get previous iteration info
        double            p_N;
        chi_math::VectorN<NDD> gradp_N;
        chi_math::VectorN<NDD> a_N;
        double            a_N_avg;

        p_N = d_pcL[ljp];
        for (int dim : dimensions)
        {
          gradp_N(dim) = d_gradpL[ljgradp[dim]];
          a_N    (dim) = d_a_PL[dim][lj0];
          a_N_avg += d_a_PL[dim][lj0];
        }
        a_N_avg /= num_dimensions;

        //============================= Compute vectors
        chi_math::VectorN<NDD> PN = adj_cell->centroid - cell.centroid;
        chi_math::VectorN<NDD> PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_math::VectorN<NDD> e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //============================= Compute interpolated values
        double delta_p                      = p_N - p_P;
        chi_math::VectorN<NDD> a_f          = (1.0-rP)*a_P     + rP*a_N;
        chi_math::VectorN<NDD> grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;
        double V_f                          = (1.0-rP)*V_P     + rP*V_N;
        chi_math::VectorN<NDD> a_f_inv      = a_f.InverseZeroIfSmaller(1.0e-10);

        //============================= Compute grad_p_f
        chi_math::VectorN<NDD> grad_pc_f = (delta_p/d_PN)*e_PN + grad_p_f_avg -
                                           grad_p_f_avg.Dot(e_PN)*e_PN;

        double m_f_old = mass_fluxes[cell.local_id][f];

        mass_fluxes[cell.local_id][f] =
          m_f_old - rho*alpha_u*V_f*A_f*n.Dot(a_f_inv * grad_pc_f);
      }//interior face
    }//for faces

    auto a_inv = a_P.InverseZeroIfSmaller(1.0e-10);
    uc = -alpha_u * V_P * a_inv * gradp_P;

    //====================================== Set vector values
    for (auto dim : dimensions)
      VecSetValue(x_u[dim], iu, uc[dim], ADD_VALUES);



  }//for cell

  //============================================= Restore local view
  for (int dim : dimensions)
    VecRestoreArrayRead(x_a_PL[dim],&d_a_PL[dim]);
  VecRestoreArrayRead(x_pcL,&d_pcL);
  VecRestoreArrayRead(x_gradpL,&d_gradpL);

  for (int dim : dimensions)
    VecGhostRestoreLocalForm(x_a_P[dim],&x_a_PL[dim]);
  VecGhostRestoreLocalForm(x_gradpc,&x_gradpL);
  VecGhostRestoreLocalForm(x_pc,&x_pcL);

  //============================================= Assemble x_u
  for (int dim : dimensions)
  {
    VecAssemblyBegin(x_u[dim]);
    VecAssemblyEnd(x_u[dim]);
  }
  VecAXPY(x_p,alpha_p,x_pc);

  //============================================= Scatter forward
  for (int dim : dimensions)
    VecGhostUpdateBegin(x_u[dim],INSERT_VALUES,SCATTER_FORWARD);
  for (int dim : dimensions)
    VecGhostUpdateEnd  (x_u[dim],INSERT_VALUES,SCATTER_FORWARD);

  VecGhostUpdateBegin(x_p,INSERT_VALUES,SCATTER_FORWARD);
  VecGhostUpdateEnd  (x_p,INSERT_VALUES,SCATTER_FORWARD);

  log.LogEvent(tag_corr,ChiLog::EventType::EVENT_END);
}