#include "solver.h"

#include "chi_log.h"
extern ChiLog& chi_log;

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
  //============================================= Declare aliases
  typedef std::vector<int> VecInt;
  typedef std::vector<VecInt> VecVecInt;
  const int ND = num_dimensions;

  //============================================= Reset matrices and RHSs
  for (int dim : dimensions)
  {
    MatZeroEntries(A_u[dim]);
    VecSet(b_u[dim],0.0);
  }

  //============================================= Get local views
  auto d_graduL = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_gradu);
  std::vector<chi_math::PETScUtils::GhostVecLocalRaw> d_a_PL(3);
  for (int dim : dimensions)
    d_a_PL[dim] = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_a_P[dim]);
  auto d_gradpL = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_gradp);
  auto d_pL     = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_p);


  //============================================= Loop over cells
  for (auto& cell : grid->local_cells)
  {
    auto   cell_fv_view = fv_sdm.MapFeView(cell.local_id);
    double V_P          = cell_fv_view->volume;

    auto& cur_cell_info = cell_info[cell.local_id];

    //====================================== Map row indices of unknowns
    int iu       = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    int ip       = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

    VecVecInt igrad_u(num_dimensions,VecInt(num_dimensions,-1));
    for (int i : dimensions)
      for (int j : dimensions)
        igrad_u[i][j] = fv_sdm.MapDOF(&cell,&uk_man_gradu,GRAD_U,i*ND+j);

    std::vector<int> igradp(3,-1);
    for (int dim : dimensions)
      igradp[dim]   = fv_sdm.MapDOF(&cell,&uk_man_gradp,GRAD_P, dim);

    //====================================== Get previous iteration info
    chi_mesh::Vector3              u_P;           //for under-relaxation
    chi_mesh::Vector3              u_P_old;       //for previous time
    chi_mesh::TensorRank2Dim3      gradu_P;       //for upwindinding
    double                         divu_P = 0.0;  //for stress term 2/3
    chi_mesh::TensorRank2Dim3      gradu_P_T;     //for cross-diffusion
    chi_mesh::Vector3              a_P_old;       //for pressure interpolation
    double                         a_P_avg;       //for pressure interpolation
    chi_mesh::Vector3              gradp_P;       //for pressure interpolation
    double                         p_P;           //for pressure interpolation

    for (auto dim : dimensions)
    {
      VecGetValues(x_u[dim]   , 1, &iu, &u_P(dim));
      VecGetValues(x_uold[dim], 1, &iu, &u_P_old(dim));
      VecGetValues(x_gradu, ND, igrad_u[dim].data(),&(gradu_P[dim](0)));
    }
    divu_P    = gradu_P.DiagSum();
    gradu_P_T = gradu_P.Transpose();
    a_P_old = cur_cell_info.a_P;
    for (int i : dimensions)
      a_P_avg += a_P_old[i];
    a_P_avg /= ND;
    VecGetValues(x_gradp, num_dimensions, igradp.data(), &gradp_P(0));
    VecGetValues(x_p,1,&ip,&p_P);

    //====================================== Init matrix coefficients
    chi_mesh::Vector3              a_t;
    chi_mesh::Vector3              a_P;
    chi_mesh::Vector3              b_P;
    chi_mesh::Vector3              b_P_pressure;
    std::vector<chi_mesh::Vector3> a_N_f(cell.faces.size());

    std::vector<int>    ju_at_f(cell.faces.size(), -1);

    //====================================== Time derivative
    if (not options.steady)
    {
      a_t = (rho*V_P/dt)*VEC3_ONES;

      b_P = b_P + a_t*u_P_old;
    }

    //====================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell = nullptr;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto& a_N = a_N_f[f];

        //============================= Map column/row indices of unknowns
        int ju       = fv_sdm.MapDOF(adj_cell,&uk_man_u,VELOCITY);
        int lju      = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int ljp      = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        VecVecInt ljgrad_u(num_dimensions,VecInt(num_dimensions,-1));
        for (int i : dimensions)
          for (int j : dimensions)
            ljgrad_u[i][j] =
              fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradu,GRAD_U,i*ND+j);

        std::vector<int> ljgradp(3,-1);
        for (int dim : dimensions)
          ljgradp[dim] = fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        ju_at_f[f] = ju;

        //============================= Get neighbor previous iteration values
        chi_mesh::TensorRank2Dim3 gradu_N;      //for upwindinding
        double                    divu_N = 0.0; //for stress term 2/3
        chi_mesh::TensorRank2Dim3 gradu_N_T;    //for cross-diffusion
        chi_mesh::Vector3         a_N_old;      //for pressure interpolation
        double                    a_N_avg;      //for pressure interpolation
        chi_mesh::Vector3         gradp_N;      //for pressure interpolation
        double                    p_N;          //for pressure interpolation

        for (int i : dimensions)
          for (int j : dimensions)
            gradu_N[i](j)  = d_graduL[ljgrad_u[i][j]];
        gradu_N_T = gradu_N.Transpose();
        divu_N    = gradu_N.DiagSum();
        for (int i : dimensions)
        {
          a_N_old(i) = d_a_PL[i][lju];
          a_N_avg += d_a_PL[i][lju];
          gradp_N(i) = d_gradpL[ljgradp[i]];
        }
        a_N_avg /= ND;
        p_N = d_pL[ljp];


        //============================= Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;
        chi_mesh::Vector3 NF = face.centroid - adj_cell->centroid;

        double d_PN    = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);
        double rP    = d_PFi/d_PN;

        chi_mesh::Vector3 Fi  = cell.centroid + d_PFi*e_PN;
        chi_mesh::Vector3 FiF = face.centroid - Fi;

        double A_p = A_f / (e_PN.Dot(n));

        chi_mesh::Vector3 A_t = A_f*n - A_p * e_PN;

        double PF_dot_n = PF.Dot(n);
        double FN_dot_n = (-1.0*NF).Dot(n);

        double r_f = PF_dot_n/(PF_dot_n + FN_dot_n);

        //============================= Develop diffusion entry
        double diffusion_entry = -(mu * A_p/d_PN);

        // Orthogonal terms
        a_N = a_N + diffusion_entry*VEC3_ONES;
        a_P = a_P - diffusion_entry*VEC3_ONES;

        // Non-orthogonal terms
        b_P -= diffusion_entry*(gradu_N - gradu_P).Dot(FiF);
        b_P += mu*A_t.Dot((1.0-rP)*gradu_P + rP*gradu_N);

        //============================= Develop cross-diffusion entry
        b_P += mu*A_f*n.Dot((1.0-r_f)*gradu_P_T + r_f*gradu_N_T);

        //============================= Develop divergence term
        b_P -= TWO_THIRDS*mu*A_f*n*((1-r_f)*divu_P + r_f*divu_N);

        //============================= Develop convection entry
        double m_f = mass_fluxes[cell.local_id][f];

        // First-order terms
        a_N = a_N + std::fmin(m_f,0.0) * VEC3_ONES;
        a_P = a_P + std::fmax(m_f,0.0) * VEC3_ONES;

        // Second-order terms
        if (m_f > 0.0)
          b_P += -m_f*PF.Dot(gradu_P);
        else
          b_P += -m_f*NF.Dot(gradu_N);

        //============================= Pressure entry
        double p_f_P = p_P + gradp_P.Dot(PF);
        double p_f_N = p_N + gradp_N.Dot(NF);

        double p_f = (a_P_avg*p_f_P + a_N_avg*p_f_N)/
                     (a_P_avg + a_N_avg);

        b_P_pressure += A_f*n*p_f;
      }//interior face
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary faces
      else
      {
        //============================= Compute Area vector
        auto PN = face.centroid - cell.centroid;
        double d_PN = PN.Norm();
        double d_perp = PN.Dot(n);

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double A_d = A_f/(e_PN.Dot(n));

        chi_mesh::Vector3 A_t = A_f*n - A_d*e_PN;

        //============================= Compute/set average face values
        auto u_b = chi_mesh::Vector3(0.0,0.0,0.0);

        if (n.Dot(J_HAT) > 0.999)
          u_b = chi_mesh::Vector3(U,0.0,0.0);

        //============================= Develop diffusion entry
        double diffusion_entry = -(mu*A_d/d_perp);

        a_P = a_P - diffusion_entry * VEC3_ONES;
        b_P = b_P - diffusion_entry * u_b;

        //============================= Develop convection entry
        //Zero except if there is a bc

        //============================= Pressure entry
        double p_f = p_P + gradp_P.Dot(PN);

        b_P_pressure += A_f*n*p_f;
      }//bndry face
    }//for faces

    //====================================== Pressure source term
//    chi_mesh::Vector3 s_pressure = -1.0*V_P*gradp_P;
    chi_mesh::Vector3 s_pressure = -1.0*b_P_pressure;

    //====================================== Body force source term
    chi_mesh::Vector3 s_body = V_P*chi_mesh::Vector3(0.0,0.0,0.0);

    //====================================== Add under-relaxation to system
    b_P = b_P + ((1.0-alpha_u)/alpha_u)*a_P*u_P;

    auto a_P_UR = (a_P + a_t)/alpha_u;

    //====================================== Assemble diagonal entries
    for (auto dim : dimensions)
      MatSetValue(A_u[dim], iu, iu, a_P_UR[dim], ADD_VALUES);

    //====================================== Assemble off-diagonal entries
    for (f=0; f<cell.faces.size(); ++f)
      if (ju_at_f[f] >= 0)
        for (auto dim : dimensions)
          MatSetValue(A_u[dim], iu, ju_at_f[f], a_N_f[f][dim], ADD_VALUES);

    //====================================== Assemble RHS
    auto rhs_entry = b_P + s_pressure + s_body;
    for (auto dim : dimensions)
      VecSetValue(b_u[dim], iu, rhs_entry[dim], ADD_VALUES);

    //====================================== Store momentum coefficients
    cur_cell_info.a_t   = a_t;
    cur_cell_info.a_P   = a_P;
    cur_cell_info.b_P   = b_P;
    cur_cell_info.a_N_f = a_N_f;

  }//for cell

  //============================================= Restore local views
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_gradu,d_graduL);
  for (int dim : dimensions)
    chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_a_P[dim],d_a_PL[dim]);
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_gradp,d_gradpL);

  //============================================= Update x_a_P
  for (auto& cell : grid->local_cells)
  {
    int iu = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);
    for (int dim : dimensions)
      VecSetValue(x_a_P[dim],iu,cell_info[cell.local_id].a_P[dim],INSERT_VALUES);
  }

  //============================================= Assemble matrices globally
  for (int i : dimensions)
  {
    MatAssemblyBegin(A_u[i],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_u[i],MAT_FINAL_ASSEMBLY);

    VecAssemblyBegin(x_u[i]);
    VecAssemblyEnd(x_u[i]);

    VecAssemblyBegin(x_a_P[i]);
    VecAssemblyEnd(x_a_P[i]);

    VecAssemblyBegin(b_u[i]);
    VecAssemblyEnd(b_u[i]);
  }

  //============================================= Scatter diagonal coefficients
  for (int dim : dimensions)
    VecGhostUpdateBegin(x_a_P[dim],INSERT_VALUES,SCATTER_FORWARD);
  for (int dim : dimensions)
    VecGhostUpdateEnd  (x_a_P[dim],INSERT_VALUES,SCATTER_FORWARD);
}