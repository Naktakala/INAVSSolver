#include "solver.h"

#include "chi_log.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/**Assemble conservation of momentum system.*/
void INAVSSolver::AssembleMomentumSystem()
{
  typedef std::vector<int> VecInt;
  typedef std::vector<VecInt> VecVecInt;
  const int ND = num_dimensions;

  typedef std::vector<chi_mesh::Vector3> VecVec3;

  for (int i=0; i<num_dimensions; ++i)
  {
    MatZeroEntries(A_u[i]);
    VecSet(b_u[i],0.0);
  }

  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    //======================================= Map row indices of unknowns
    int iu       = fv_sdm.MapDOF(cell.global_id, &uk_man_u, VELOCITY);

    VecVecInt igrad_u(num_dimensions,VecInt(num_dimensions,-1));
    for (int dimv : dimensions)
      for (int dim : dimensions)
        igrad_u[dimv][dim] = fv_sdm.MapDOF(cell.global_id,
                                           &uk_man_gradu,GRAD_U,dimv*ND+dim);

    std::vector<int> igradp(3,-1);
    for (int dim : dimensions)
      igradp[dim]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, dim);

    //======================================= Get previous iteration info
    chi_mesh::Vector3              u_P;
    chi_mesh::Vector3              u_P_old;
    std::vector<chi_mesh::Vector3> gradu_P(num_dimensions);
    chi_mesh::Vector3              gradp_P;

    for (auto dim : dimensions)
    {
      VecGetValues(x_u[dim]   , 1, &iu, &u_P(dim));
      VecGetValues(x_uold[dim], 1, &iu, &u_P_old(dim));
      VecGetValues(x_gradu, num_dimensions,
                   igrad_u[dim].data(),&(gradu_P[dim])(0));
    }

    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));

    //======================================= Init matrix coefficients
    chi_mesh::Vector3 a_t;
    chi_mesh::Vector3 a_P;
    chi_mesh::Vector3 b_P;
    std::vector<chi_mesh::Vector3> a_N_f(cell.faces.size());

    std::vector<int>    neighbor_indices(cell.faces.size(), -1);

    //======================================= Time derivative
    if (not options.steady)
    {
      for (auto dim : dimensions)
        a_t(dim) = rho * V_P / dt;

      b_P = b_P + a_t*u_P_old;
    }

    //======================================= Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n = face.normal;

      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto& a_N = a_N_f[f];

        //===================== Map column/row indices of unknowns
        int ju       = fv_sdm.MapDOF(face.neighbor,&uk_man_u,VELOCITY);

        VecVecInt jgrad_u(num_dimensions,VecInt(num_dimensions,-1));
        for (int dimv : dimensions)
          for (int dim : dimensions)
            jgrad_u[dimv][dim] = fv_sdm.MapDOF(face.neighbor,
                                               &uk_man_gradu,GRAD_U,dimv*ND+dim);

        neighbor_indices[f] = ju;

        //===================== Get neighbor previous iteration values
        std::vector<chi_mesh::Vector3> gradu_N(num_dimensions);

        for (int dim : dimensions)
          VecGetValues(x_gradu, num_dimensions, jgrad_u[dim].data(),&(gradu_N[dim])(0));

        //===================== Compute PN, PF, Nf
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;
        chi_mesh::Vector3 NF = face.centroid - adj_cell->centroid;

        double d_PN    = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);
        double rP    = d_PFi/d_PN;

        chi_mesh::Vector3 Fi  = cell.centroid + d_PFi*e_PN;
        chi_mesh::Vector3 FiF = face.centroid - Fi;

        double A_d = A_f/(e_PN.Dot(n));

        chi_mesh::Vector3 A_t = A_f*n - A_d*e_PN;

        //===================== Develop diffusion entry
        double diffusion_entry = -(mu*A_d/d_PN);

        // Orthogonal terms
        a_N = a_N + diffusion_entry * VEC3_ONES;
        a_P = a_P - diffusion_entry * VEC3_ONES;

        // Non-orthogonal terms
        for (auto dim : dimensions)
        {
          b_P(dim) += diffusion_entry*(gradu_N[dim] - gradu_P[dim]).Dot(FiF);
          b_P(dim) += mu*A_t.Dot((1.0-rP)*gradu_P[dim] + rP*gradu_N[dim]);
        }

        //===================== Develop convection entry
        double m_f = mass_fluxes[cell.local_id][f];

        // First-order terms
        a_N = a_N + std::fmin(m_f,0.0) * VEC3_ONES;
        a_P = a_P + std::fmax(m_f,0.0) * VEC3_ONES;

        // Second-order terms
        if (m_f > 0.0)
          for (int dim : dimensions)
            b_P(dim) += - m_f*gradu_P[dim].Dot(PF);
        else
          for (int dim : dimensions)
            b_P(dim) += - m_f*gradu_N[dim].Dot(NF);

      }//interior face
      else
      {
        //===================== Compute Area vector
        auto PN = face.centroid - cell.centroid;
        double d_PN = PN.Norm();
        double d_perp = PN.Dot(n);

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double A_d = A_f/(e_PN.Dot(n));

        chi_mesh::Vector3 A_t = A_f*n - A_d*e_PN;

        //===================== Compute/set average face values
        auto u_b = chi_mesh::Vector3(0.0,0.0,0.0);

        if (n.Dot(J_HAT) > 0.999)
          u_b = chi_mesh::Vector3(U,0.0,0.0);

        //===================== Develop diffusion entry
        double diffusion_entry = -(mu*A_d/d_perp);

        a_P = a_P - diffusion_entry * VEC3_ONES;
        b_P = b_P - diffusion_entry * u_b;

        //===================== Develop convection entry
        //Zero except if there is a bc

        //===================== Develop pressure entry
        //Handled in another loop
      }//bndry face
    }//for faces

    //================================= Pressure source term
    chi_mesh::Vector3 s_pressure = -1.0*V_P*gradp_P;
//    chi_mesh::Vector3 s_pressure;

    //================================= Body force source term
    chi_mesh::Vector3 s_body = V_P*chi_mesh::Vector3(0.0,0.0,0.0);

    //================================= Add under-relaxation to system
    b_P = b_P + ((1.0-alpha_u)/alpha_u)*a_P*u_P;

    auto a_P_UR = (a_P + a_t)/alpha_u;

    //================================= Assemble diagonal entries
    for (auto dim : dimensions)
      MatSetValue(A_u[dim], iu, iu, a_P_UR[dim], ADD_VALUES);

    //================================= Assemble off-diagonal entries
    for (f=0; f<cell.faces.size(); ++f)
    {
      if (neighbor_indices[f] < 0) continue;

      for (auto dim : dimensions)
        MatSetValue(A_u[dim], iu, neighbor_indices[f], a_N_f[f][dim], ADD_VALUES);
    }

    //================================= Assemble RHS
    auto rhs_entry = b_P + s_pressure + s_body;
    for (auto dim : dimensions)
      VecSetValue(b_u[dim], iu, rhs_entry[dim], ADD_VALUES);

    //================================= Store momentum coefficients
    auto& cell_mom_coeffs = momentum_coeffs[cell.local_id];

    cell_mom_coeffs.a_t   = a_t;
    cell_mom_coeffs.a_P   = a_P;
    cell_mom_coeffs.b_P   = b_P;
    cell_mom_coeffs.a_N_f = a_N_f;

  }//for cell


  //================================================== Pressure source term
//  for (auto& cell : grid->local_cells)
//  {
//    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);
//
//    auto& a_P = momentum_coeffs[cell.local_id].a_P;
//
//    //======================================= Map row indices of unknowns
//    int iu       = fv_sdm.MapDOF(cell.global_id, &uk_man_u, VELOCITY);
//    int ip       = fv_sdm.MapDOF(cell.global_id,&uk_man_p,PRESSURE);
//
//    std::vector<int> igradp(3,-1);
//    igradp[P_X]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_X);
//    igradp[P_Y]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Y);
//    igradp[P_Z]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Z);
//
//    //=========================================== Get cur-cell values
//    double p_P;
//    chi_mesh::Vector3 gradp_P;
//
//    VecGetValues(x_p,1,&ip,&p_P);
//    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
//
//    //=========================================== Declare coeficients
//    chi_mesh::Vector3 bp_P;
//
//    //=========================================== Loop over faces
//    int f=-1;
//    for (auto& face : cell.faces)
//    {
//      ++f;
//      auto A_f = cell_fv_view->face_area[f];
//      chi_mesh::Vector3& n = face.normal;
//
//      if (face.neighbor>=0)
//      {
//        chi_mesh::Cell* adj_cell = nullptr;
//        if (face.IsNeighborLocal(grid))
//          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
//        else
//          adj_cell = grid->cells[face.neighbor];
//
//        auto& a_N = momentum_coeffs[adj_cell->local_id].a_P;
//
//        //================================== Map indices
//        int jp   = fv_sdm.MapDOF(face.neighbor, &uk_man_p, PRESSURE);
//
//        //================================== Get adj-cell values
//        double p_N;
//
//        VecGetValues(x_p, 1, &jp, &p_N);
//
//        //================================== Compute face average values
//        chi_mesh::Vector3 p_avg;
//        for (auto dim : dimensions)
//          p_avg(dim) = (p_P / a_P[dim] + p_N / a_N[dim]) /
//                       (1.0/a_P[dim] + 1.0/a_N[dim]);
//
//        //================================== Develop pressure entry
//        bp_P = bp_P - A_f*n*p_avg;
//      }//not bndry
//      else
//      {
//        //================================== Compute Area vector
//        auto ds = face.centroid - cell.centroid;
//
//        double p_avg  = p_P + gradp_P.Dot(ds);
//
//        //================================== Develop pressure entry
//        bp_P = bp_P - A_f*n*p_avg;
//      }//bndry
//    }//for faces
//
//    for (auto dim : dimensions)
//      VecSetValue(b_u[dim], iu, bp_P[dim], ADD_VALUES);
//
//  }//for cells


  //============================================= Assemble matrices globally
  for (int i=0; i<num_dimensions; ++i)
  {
    MatAssemblyBegin(A_u[i],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_u[i],MAT_FINAL_ASSEMBLY);

    VecAssemblyBegin(x_u[i]);
    VecAssemblyEnd(x_u[i]);

    VecAssemblyBegin(b_u[i]);
    VecAssemblyEnd(b_u[i]);
  }
}