#include "solver.h"

#include "chi_log.h"

#include "ChiMath/chi_math.h"

//###################################################################
/** Computes the gradient of the pressure.*/
void INAVSSolver::ComputeGradP_WLSQ(Vec v_gradp, Vec v_p, bool limited)
{
  auto& log = ChiLog::GetInstance();

  //============================================= Create work vectors
  Vec v_gradp_old;
  VecDuplicate(v_gradp,&v_gradp_old);

  VecSet(x_gradp,0.0);

  //============================================= Get local views
  auto d_pL =
    chi_math::PETScUtils::GetGhostVectorLocalViewRead(v_p);
  auto d_gradpL =
    chi_math::PETScUtils::GetGhostVectorLocalViewRead(v_gradp_old);

  //============================================= Max/Mins
  std::vector<double> p_max;
  std::vector<double> p_min;
  if (limited)
  {
    p_max.resize(grid->local_cells.size(),0.0);
    p_min.resize(grid->local_cells.size(),0.0);

    for (auto& cell : grid->local_cells)
    {
      //==================================== Map row indices of unknowns
      int ip = fv_sdm.MapDOF(&cell, &uk_man_p, PRESSURE);

      //==================================== Get previous iteration info
      double p_P;

      VecGetValues(x_p, 1, &ip, &p_P);

      //==================================== Setup max/mins
      p_max[cell.local_id] = p_P;
      p_min[cell.local_id] = p_P;

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
          int ljp = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

          //=========================== Get previous iteration info
          double p_N;

          p_N = d_pL[ljp];

          //=========================== Determine max/min
          double prev_max = p_max[cell.local_id];
          double prev_min = p_min[cell.local_id];
          p_max[cell.local_id] = std::fmax(p_N,prev_max);
          p_min[cell.local_id] = std::fmin(p_N,prev_min);
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
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    //==================================== Map indices
    int ip        = fv_sdm.MapDOF(&cell,&uk_man_p,PRESSURE);

    std::vector<int> igradp(3,-1);
    for (int dim : dimensions)
      igradp[dim] =
        fv_sdm.MapDOF(&cell,&uk_man_gradp,GRAD_P,dim);

    //==================================== Get cur-cell values
    double            p_P;
    chi_mesh::Vector3 gradp_P;

    VecGetValues(v_p,1,&ip,&p_P);
    VecGetValues(v_gradp_old,num_dimensions,igradp.data(),&gradp_P(0));

    //==================================== Declare system
    MatDbl A(num_dimensions,VecDbl(num_dimensions,0.0));
    VecDbl b(num_dimensions,0.0);

    //====================================== Compute deltas
    double alpha = 1.0;
    double delta_max = 0.0;
    double delta_min = 0.0;

    if (limited)
    {
      delta_max = p_max[cell.local_id] - p_P;
      delta_min = p_min[cell.local_id] - p_P;
    }

    //==================================== Loop over faces
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      double             A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n   = face.normal;

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Internal faces
      if (face.neighbor>=0)
      {
        chi_mesh::Cell* adj_cell;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        //=========================== Map indices
        int ljp = fv_sdm.MapDOFLocal(adj_cell,&uk_man_p,PRESSURE);

        std::vector<int> ljgradp(3,-1);
        for (int dim : dimensions)
          ljgradp[dim] =
            fv_sdm.MapDOFLocal(adj_cell,&uk_man_gradp,GRAD_P,dim);

        //=========================== Get adj-cell values
        double p_N;
        chi_mesh::Vector3 gradp_N;

        p_N = d_pL[ljp];
        for (int i : dimensions)
          gradp_N(i) = d_gradpL[ljgradp[i]];

        //=========================== Compute vectors
        chi_mesh::Vector3           PN = adj_cell->centroid - cell.centroid;
        chi_mesh::TensorRank2Dim3 PNPN = PN.OTimes(PN);
        double                 norm_PN = PN.NormSquare();

        double w = 1.0/norm_PN;

        //=========================== Compute alpha_f
        if (limited)
        {
          chi_mesh::Vector3 PF = face.centroid - cell.centroid;
          double delta_f = gradp_P.Dot(PF);
          double r = 0.0;
          double alpha_f = 0.0;

          if (delta_f>0.0)
            r = delta_max/delta_f;
          else
            r = delta_min/delta_f;

          alpha_f = (r*r + 2.0*r)/
                    (r*r + r + 2.0);
          alpha = std::fmin(alpha_f,alpha);
        }

        //=========================== Set matrix coefficients
        for (int i : dimensions)
          for (int j : dimensions)
            A[i][j] += w * PNPN[i][j];

        for (int i : dimensions)
          b[i] += w*PN[i]*(p_N - p_P);
      }//not bndry
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
      else
      {
        //=========================== Compute vectors
        chi_mesh::Vector3           PF = face.centroid - cell.centroid;
        chi_mesh::TensorRank2Dim3 PFPF = PF.OTimes(PF);
        double                 norm_PF = PF.NormSquare();

        double w = 1.0/norm_PF;

        //=========================== Compute boundary value
        double p_N = p_P + gradp_P.Dot(PF);

        //=========================== Set matrix coefficients
        for (int i : dimensions)
          for (int j : dimensions)
            A[i][j] += w * PFPF[i][j];

        for (int i : dimensions)
          b[i] += w*PF[i]*(p_N - p_P);
      }
    }//for faces

    //==================================== Solve system
    chi_math::GaussElimination(A,b,num_dimensions);

    //==================================== Set vector values
    for (auto dim : dimensions)
      VecSetValue(v_gradp,igradp[dim],alpha*b[dim],ADD_VALUES);
  }//for cells

  //============================================= Restore local views
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(v_p,d_pL);
  chi_math::PETScUtils::RestoreGhostVectorLocalViewRead(v_gradp_old,d_gradpL);

  //======================================== Assemble units
  VecAssemblyBegin(v_gradp);
  VecAssemblyEnd(v_gradp);

  //============================================= Scatter gradp
  chi_math::PETScUtils::CommunicateGhostEntries(v_gradp);

  //============================================= Destroy work vectors
  VecDestroy(&v_gradp_old);
}

