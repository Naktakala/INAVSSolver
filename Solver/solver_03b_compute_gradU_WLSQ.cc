#include "solver.h"

#include "ChiMath/chi_math.h"

#include "chi_log.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/** Computes the gradient of the velocity where face velocities
 * are interpolated using the momentum equation.*/
void INAVSSolver::ComputeGradU_WLSQ(bool limited)
{
  auto& log = ChiLog::GetInstance();

  typedef std::vector<int> VecInt;
  typedef std::vector<VecInt> VecVecInt;
  const int ND = num_dimensions;

  //============================================= Create work vectors
  Vec v_gradu_old;
  VecDuplicate(x_gradu,&v_gradu_old);

  VecSet(x_gradu,0.0);

  //============================================= Get local views
  std::vector<chi_math::PETScUtils::GhostVecLocalRaw> d_uL(3);
  for (int dim : dimensions)
    d_uL[dim] = chi_math::PETScUtils::GetGhostVectorLocalViewRead(x_u[dim]);
  auto d_graduL =
    chi_math::PETScUtils::GetGhostVectorLocalViewRead(v_gradu_old);

  //============================================= Max/Mins
  std::vector<std::vector<double>> u_max;
  std::vector<std::vector<double>> u_min;
  if (limited)
  {
    u_max.resize(grid->local_cells.size());
    u_min.resize(grid->local_cells.size());

    for (auto& cell : grid->local_cells)
    {
      //==================================== Map row indices of unknowns
      int iu = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);

      //==================================== Get previous iteration info
      chi_mesh::Vector3 u_P;

      for (int dim : dimensions)
        VecGetValues(x_u[dim], 1, &iu, &u_P(dim));

      //==================================== Setup max/mins
      u_max[cell.local_id].resize(num_dimensions,0.0);
      u_min[cell.local_id].resize(num_dimensions,0.0);

      for (int dim : dimensions)
      {
        u_max[cell.local_id][dim] = u_P[dim];
        u_min[cell.local_id][dim] = u_P[dim];
      }

      //==================================== Loop over faces
      for (auto& face : cell.faces)
      {
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
        if (face.neighbor>=0)
        {
          chi_mesh::Cell* adj_cell;
          if (face.IsNeighborLocal(grid))
            adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
          else
            adj_cell = grid->cells[face.neighbor];

          //=========================== Map row indices of unknowns
          int lju = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);

          //=========================== Get previous iteration info
          chi_mesh::Vector3         u_N;

          for (int dim : dimensions)
            u_N(dim) = d_uL[dim][lju];

          //=========================== Determine max/min
          for (int dim :dimensions)
          {
            double prev_max = u_max[cell.local_id][dim];
            double prev_min = u_min[cell.local_id][dim];
            u_max[cell.local_id][dim] = std::fmax(u_N[dim],prev_max);
            u_min[cell.local_id][dim] = std::fmin(u_N[dim],prev_min);
          }
        }
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
        else
        {

        }
      }
    }//for cells
  }//if limited

  //============================================= Compute the gradient
  for (auto& cell : grid->local_cells)
  {
    //====================================== Map row indices of unknowns
    int iu            = fv_sdm.MapDOF(&cell, &uk_man_u, VELOCITY);

    VecVecInt igrad_u(num_dimensions,VecInt(num_dimensions,-1));
    for (int i : dimensions)
      for (int j : dimensions)
        igrad_u[i][j] =
          fv_sdm.MapDOF(&cell,&uk_man_gradu,GRAD_U,i*ND+j);

    //====================================== Get previous iteration info
    chi_mesh::Vector3         u_P;
    chi_mesh::TensorRank2Dim3 gradu_P;

    for (int dim : dimensions)
    {
      VecGetValues(x_u[dim], 1, &iu, &u_P(dim));
      VecGetValues(v_gradu_old, ND, igrad_u[dim].data(),&(gradu_P[dim](0)));
    }

    //====================================== Declare systems
    std::vector<MatDbl> A(num_dimensions);
    std::vector<VecDbl> b(num_dimensions);
    for (int dim : dimensions)
    {
      A[dim].resize(num_dimensions,VecDbl(num_dimensions,0.0));
      b[dim].resize(num_dimensions,0.0);
    }

    //====================================== Compute deltas
    chi_mesh::Vector3 alpha = chi_mesh::Vector3(1.0,1.0,1.0);
    std::vector<double> delta_max(num_dimensions,0.0);
    std::vector<double> delta_min(num_dimensions,0.0);

    if (limited)
    {
      for (int dim : dimensions)
      {
        delta_max[dim] = u_max[cell.local_id][dim] - u_P[dim];
        delta_min[dim] = u_min[cell.local_id][dim] - u_P[dim];
      }
    }

    //====================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      chi_mesh::Vector3& n   = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal face
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        //============================= Map row indices of unknowns
        int lju          = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);

        VecVecInt ljgrad_u(num_dimensions,VecInt(num_dimensions,-1));
        for (int i : dimensions)
          for (int j : dimensions)
            ljgrad_u[i][j] =
              fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradu,GRAD_U,i*ND+j);

        //============================= Get previous iteration info
        chi_mesh::Vector3         u_N;
        chi_mesh::TensorRank2Dim3 gradu_N;

        for (int dim :dimensions)
          u_N(dim) = d_uL[dim][lju];

        for (int i : dimensions)
          for (int j : dimensions)
            gradu_N[i](j)  = d_graduL[ljgrad_u[i][j]];

        //=========================== Compute vectors
        chi_mesh::Vector3           PN = adj_cell->centroid - cell.centroid;
        chi_mesh::TensorRank2Dim3 PNPN = PN.OTimes(PN);
        double                 norm_PN = PN.NormSquare();

        double w = 1.0/norm_PN;

        //=========================== Compute alpha_f
        if (limited)
        {
          chi_mesh::Vector3      PF = face.centroid - cell.centroid;
          chi_mesh::Vector3 delta_f = gradu_P.Dot(PF);
          chi_mesh::Vector3 r;
          chi_mesh::Vector3 alpha_f;

          for (int dim : dimensions)
          {
            if (delta_f[dim]>0.0)
              r(dim) = delta_max[dim]/delta_f[dim];
            else
              r(dim) = delta_min[dim]/delta_f[dim];
          }

          for (int dim : dimensions)
          {
            alpha_f(dim) = (r[dim]*r[dim] + 2.0*r[dim])/
                           (r[dim]*r[dim] + r[dim] + 2.0);
            alpha(dim) = std::fmin(alpha_f[dim],alpha[dim]);
          }
        }

        //=========================== Set matrix coefficients
        for (int dim : dimensions)
          for (int i : dimensions)
            for (int j : dimensions)
              A[dim][i][j] += w*PNPN[i][j];

        for (int dim : dimensions)
          for (int i : dimensions)
            b[dim][i] += w * PN[i] * (u_N[dim] - u_P[dim]);
      }//not bndry
      // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
      else
      {
        //=========================== Compute vectors
        chi_mesh::Vector3           PF = face.centroid - cell.centroid;
        chi_mesh::TensorRank2Dim3 PFPF = PF.OTimes(PF);
        double                 norm_PF = PF.Norm();

        double w = 1.0/norm_PF;

        //=========================== Set boundary value
        chi_mesh::Vector3 u_N;
        if (n.Dot(J_HAT) > 0.999)
          u_N = chi_mesh::Vector3(U,0.0,0.0);

        //=========================== Set matrix coefficients
        for (int dim : dimensions)
          for (int i : dimensions)
            for (int j : dimensions)
              A[dim][i][j] += w*PFPF[i][j];

        for (int dim : dimensions)
          for (int i : dimensions)
            b[dim][i] += w*PF[i]*(u_N[dim] - u_P[dim]);
      }//bndry
    }//for faces

    //==================================== Solve systems
    for (int dim : dimensions)
      chi_math::GaussElimination(A[dim],b[dim],num_dimensions);

    //====================================== Set vector values
    for (int i : dimensions)
      for (int j : dimensions)
        VecSetValue(x_gradu,igrad_u[i][j],alpha[i]*b[i][j],ADD_VALUES);
  }//for cells

  //============================================= Restore local views
  for (int dim : dimensions)
    chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(x_u[dim],d_uL[dim]);
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(v_gradu_old,d_graduL);

  //============================================= Assemble applicable units
  VecAssemblyBegin(x_gradu);
  VecAssemblyEnd(x_gradu);

  //============================================= Scatter applicable units
  VecGhostUpdateBegin(x_gradu,INSERT_VALUES,SCATTER_FORWARD);
  VecGhostUpdateEnd  (x_gradu,INSERT_VALUES,SCATTER_FORWARD);

  //============================================= Destroy work vectors
  VecDestroy(&v_gradu_old);
}