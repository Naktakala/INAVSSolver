#include "solver.h"

//###################################################################
/** Computes the gradient of the pressure.*/
//void INAVSSolver::ComputeGradP(Vec v_gradp, Vec v_p)
//{
//
//  Vec v_gradp_old;
//  VecDuplicate(v_gradp,&v_gradp_old);
//
//  VecSet(v_gradp,0.0);
//
//  for (auto& cell : grid->local_cells)
//  {
//    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);
//
//    double V = cell_fv_view->volume;
//
//    //=========================================== Map indices
//    int ip            = fv_sdm.MapDOF(cell.global_id,&uk_man_p,PRESSURE);
//    int igradp_x      = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_X);
//    int igradp_y      = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Y);
//
//    //=========================================== Get cur-cell values
//    double p_m;       VecGetValues(v_p,1,&ip,&p_m);
//    chi_mesh::Vector3 gradp_old_P;
//    VecGetValues(v_gradp_old, 1, &igradp_x, &gradp_old_P.x);
//    VecGetValues(v_gradp_old, 1, &igradp_y, &gradp_old_P.y);
//
//    double a_x_m = 0.0;
//    double a_y_m = 0.0;
//
//    int f=-1;
//    for (auto& face : cell.faces)
//    {
//      ++f;
//      double             A_f = cell_fv_view->face_area[f];
//      chi_mesh::Vector3& n   = face.normal;
//
//      if (face.neighbor>=0)
//      {
//        //================================== Map indices
//        int jp_p          = fv_sdm.MapDOF(face.neighbor,&uk_man_p,PRESSURE);
//
//        //================================== Get adj-cell values
//        double p_p;       VecGetValues(v_p,1,&jp_p,&p_p);
//
//        double p_f = 0.5*(p_p + p_m);
//
//        a_x_m += A_f*n.x*p_f;
//        a_y_m += A_f*n.y*p_f;
//      }//not bndry
//      else
//      {
//        double p_b = p_m + gradp_old_P.Dot(face.centroid - cell.centroid);
//
//        a_x_m += A_f*n.x*p_b;
//        a_y_m += A_f*n.y*p_b;
//      }
//    }//for faces
//
//    VecSetValue(v_gradp,igradp_x,a_x_m/V,ADD_VALUES);
//    VecSetValue(v_gradp,igradp_y,a_y_m/V,ADD_VALUES);
//  }//for cells
//
//  VecAssemblyBegin(v_gradp);
//  VecAssemblyEnd(v_gradp);
//
//  VecDestroy(&v_gradp_old);
//}

//###################################################################
/** Computes the gradient of the pressure.*/
void INAVSSolver::ComputeGradP_GG(Vec v_gradp, Vec v_p)
{
  Vec v_gradp_old;
  VecDuplicate(v_gradp,&v_gradp_old);

  double fnorm_gradp_old = 0.0; VecNorm(v_gradp_old,NORM_2,&fnorm_gradp_old);

  for (int k=0; k<2; ++k)
  {
    for (auto& cell : grid->local_cells)
    {
      auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

      double V = cell_fv_view->volume;

      //=========================================== Map indices
      int ip        = fv_sdm.MapDOF(cell.global_id,&uk_man_p,PRESSURE);

      std::vector<int> igradp(3,-1);
      igradp[P_X]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_X);
      igradp[P_Y]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Y);
      igradp[P_Z]   = fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P, P_Z);

      //=========================================== Get cur-cell values
      double            p_P;
      chi_mesh::Vector3 gradp_P;

      VecGetValues(v_p,1,&ip,&p_P);
      VecGetValues(v_gradp_old, num_dimensions, igradp.data(), &gradp_P(0));

      //=========================================== Declare coeficients
      chi_mesh::Vector3 a_P;

      //=========================================== Loop over faces
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

          //================================== Map indices
          int jp_N     = fv_sdm.MapDOF(face.neighbor,&uk_man_p,PRESSURE);

          std::vector<int> jgradp(3,-1);
          jgradp[P_X] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_X);
          jgradp[P_Y] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Y);
          jgradp[P_Z] = fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P, P_Z);

          //================================== Get adj-cell values
          double            p_N;
          chi_mesh::Vector3 gradp_N;

          VecGetValues(v_p,1,&jp_N,&p_N);
          VecGetValues(v_gradp_old, num_dimensions, jgradp.data(), &gradp_N(0));

          //================================== Compute vectors
          chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
          chi_mesh::Vector3 PF = face.centroid - cell.centroid;
          chi_mesh::Vector3 NF = face.centroid - adj_cell->centroid;

          double d_PN = PN.Norm();

          chi_mesh::Vector3 e_PN = PN/d_PN;

          double d_PFi = PF.Dot(e_PN);

          double rP = d_PFi/d_PN;

          //================================== Compute face value
          double p_f = (1.0-rP)*p_P + rP*p_N +
                       (1.0-rP)*gradp_P.Dot(PF) + rP*gradp_N.Dot(NF);

          a_P = a_P + A_f*n*p_f;
        }//not bndry
        else
        {
          //================================== Compute vectors
          chi_mesh::Vector3 PF = face.centroid - cell.centroid;

          //================================== Compute face value
          double p_b = p_P + gradp_P.Dot(PF);

          a_P = a_P + A_f*n*p_b;
        }
      }//for faces

      for (auto dim : dimensions)
        VecSetValue(v_gradp,igradp[dim],a_P[dim]/V,INSERT_VALUES);
    }//for cells

    VecAssemblyBegin(v_gradp);
    VecAssemblyEnd(v_gradp);

    double fnorm_gradp = 0.0; VecNorm(v_gradp,NORM_2,&fnorm_gradp);
    double norm_ratio  = 0.0;

    if (fnorm_gradp_old > 1.0e-12)
      norm_ratio = fnorm_gradp/fnorm_gradp_old;

    if (std::fabs(norm_ratio - 1.0) < 1.0e-4)
      break;

    VecCopy(v_gradp,v_gradp_old);
    fnorm_gradp_old = fnorm_gradp;
  }//for k iteration

  VecDestroy(&v_gradp_old);
}