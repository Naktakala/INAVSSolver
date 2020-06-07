#include "solver.h"

#include "chi_log.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

#include "ChiMath/chi_math_vectorNX.h"
#include "ChiMath/chi_math_tensorRNX.h"

//###################################################################
/** Computes the gradient of the velocity where face velocities
 * are interpolated using the momentum equation.*/
template<int NDD>
void INAVSSolver<NDD>::ComputeMassFluxMMIM()
{
  auto& log = ChiLog::GetInstance();
  log.LogEvent(tag_comp_mf,ChiLog::EventType::EVENT_BEGIN);

  typedef std::vector<int> VecInt;
  typedef std::vector<VecInt> VecVecInt;

  //============================================= Get local views
  std::vector<chi_math::PETScUtils::GhostVecLocalRaw> d_uL(NDD);
  std::vector<chi_math::PETScUtils::GhostVecLocalRaw> d_umimL(NDD);
  std::vector<chi_math::PETScUtils::GhostVecLocalRaw> d_a_PL(NDD);

  for (int dim : dimensions)
  {
    d_uL[dim]   = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_u[dim]);
    d_a_PL[dim] = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_a_P[dim]);
  }
  auto d_pL     = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_p);
  auto d_gradpL = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_gradp);

  //============================================= Compute MIM velocities
  for (auto& cell : grid->local_cells)
  {
    auto& cell_mom_coeffs = cell_info[cell.local_id];

    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);

    //====================================== Declare H_P coefficients
    chi_math::VectorN<NDD> H_P;

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

        chi_math::VectorN<NDD> u_N;
        for (int dim : dimensions)
          u_N(dim) = d_uL[dim][lju];

        H_P = H_P - a_N_f*u_N;
      }//if internal
    }//for faces

    //====================================== Add b coefficient
    chi_math::VectorN<NDD> u_mim = H_P + cell_mom_coeffs.b_P;

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

  //============================================= Get local view of umim
  for (int dim : dimensions)
    d_umimL[dim] = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_umim[dim]);

  //============================================= Compute the gradient
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

    VecVecInt igrad_u(num_dimensions,VecInt(num_dimensions,-1));
    for (int dimv : dimensions)
      for (int dim : dimensions)
        igrad_u[dimv][dim] =
          fv_sdm.MapDOF(&cell,&uk_man_gradu,GRAD_U,dimv*NDD+dim);

    //====================================== Get previous iteration info
    double p_P = 0.0;
    chi_math::VectorN<NDD> gradp_P;
    chi_math::VectorN<NDD> u_mim_P;
    chi_math::VectorN<NDD> a_P;

    VecGetValues(x_p,1,&ip,&p_P);
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
    for (int dim : dimensions)
    {
      VecGetValues(x_umim[dim], 1, &iu, &u_mim_P(dim));
      VecGetValues(x_a_P[dim] , 1, &iu, &a_P(dim));
    }

    //====================================== Declare grad coefficients
    std::vector<chi_math::VectorN<NDD>> a_gradu(num_dimensions);

    //====================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      double             A_f = cell_fv_view->face_area[f];
      chi_math::VectorN<NDD> n = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto adj_cell_fv_view = fv_sdm.MapNeighborFeView(face.neighbor);

        double V_N = adj_cell_fv_view->volume;

        //============================= Map row indices of unknowns
        int lju          = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int ljp          = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> ljgradp(NDD,-1);
        for (int dim : dimensions)
          ljgradp[dim] = fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        //============================= Get previous iteration info
        double p_N;
        chi_math::VectorN<NDD> gradp_N;
        chi_math::VectorN<NDD> u_mim_N;
        chi_math::VectorN<NDD> a_N;

        p_N = d_pL[ljp];
        for (int dim :dimensions)
        {
          gradp_N(dim) = d_gradpL[ljgradp[dim]];
          u_mim_N(dim) = d_umimL[dim][lju];
          a_N    (dim) = d_a_PL[dim][lju];
        }


        //============================= Compute vectors
        chi_math::VectorN<NDD> PN = adj_cell->centroid - cell.centroid;
        chi_math::VectorN<NDD> PF = face.centroid - cell.centroid;

        double d_PN = PN.Norm();

        chi_math::VectorN<NDD> e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //============================= Compute interpolated values
        double dp                           = p_N - p_P;
        chi_math::VectorN<NDD> u_mim_f      = (1.0-rP)*u_mim_P + rP*u_mim_N;
        chi_math::VectorN<NDD> a_f          = (1.0-rP)*a_P     + rP*a_N;
        double            V_f               = (1.0-rP)*V_P     + rP*V_N;
        chi_math::VectorN<NDD> grad_p_f_avg = (1.0-rP)*gradp_P + rP*gradp_N;

        chi_math::VectorN<NDD> a_f_inv = a_f.InverseZeroIfSmaller(1.0e-10);

        //============================= Compute grad_p_f
//        chi_mesh::Vector3 grad_p_f = (dp/d_PN)*e_PN + grad_p_f_avg -
//                                     grad_p_f_avg.Dot(e_PN)*e_PN;
        chi_math::VectorN<NDD> grad_p_f = (dp/d_PN)*e_PN;

        //============================= Compute face velocities
        chi_math::VectorN<NDD> u_f = (alpha_u*a_f_inv)*(u_mim_f - V_f * grad_p_f);

        mass_fluxes[cell.local_id][f] = rho*A_f*n.Dot(u_f);
      }//not bndry
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
    }//for faces

  }//for cells

  //============================================= Restore local views
  for (int dim : dimensions)
  {
    chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_u[dim],d_uL[dim]);
    chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_umim[dim],d_umimL[dim]);
    chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_a_P[dim],d_a_PL[dim]);
  }
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_p,d_pL);
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_gradp,d_gradpL);

  log.LogEvent(tag_comp_mf,ChiLog::EventType::EVENT_END);
}