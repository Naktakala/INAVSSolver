#include "solver.h"

//###################################################################
/** Computes the gradient of the pressure.*/
void INAVSSolver::ComputeGradP_GG(Vec v_gradp, Vec v_p)
{
  Vec v_gradp_old;
  Vec v_cell_vols;
  VecDuplicate(v_gradp,&v_gradp_old);
  VecDuplicate(v_gradp,&v_cell_vols);

  double fnorm_gradp_old = 0.0;
  VecNorm(v_gradp_old,NORM_2,&fnorm_gradp_old);

  for (int k=0; k<2; ++k)
  {
    VecSet(v_gradp,0.0);
    for (auto& cell : grid->local_cells)
    {
      auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

      double V = cell_fv_view->volume;

      //=========================================== Map indices
      int ip        = fv_sdm.MapDOF(cell.global_id,&uk_man_p,PRESSURE);

      std::vector<int> igradp(3,-1);
      for (int dim : dimensions)
        igradp[dim] =
          fv_sdm.MapDOF(cell.global_id,&uk_man_gradp,GRAD_P,dim);

      //=========================================== Get cur-cell values
      double            p_P;
      chi_mesh::Vector3 gradp_P;

      VecGetValues(v_p,1,&ip,&p_P);
      VecGetValues(v_gradp_old,num_dimensions,igradp.data(),&gradp_P(0));

      //=========================================== Declare coeficients
      chi_mesh::Vector3 a_P;

      //=========================================== Set cell volumes
      for (int dim : dimensions)
        VecSetValue(v_cell_vols,igradp[dim],V,INSERT_VALUES);

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
          std::vector<int> jgradp(3,-1);
          for (int dim : dimensions)
            jgradp[dim] =
              fv_sdm.MapDOF(face.neighbor,&uk_man_gradp,GRAD_P,dim);

          //================================== Get adj-cell values
          // We are using one-sided assembly so we don't need
          // the values.

          //================================== Compute vectors
          chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
          chi_mesh::Vector3 PF = face.centroid - cell.centroid;

          double d_PN = PN.Norm();

          chi_mesh::Vector3 e_PN = PN/d_PN;

          double d_PFi = PF.Dot(e_PN);

          double rP = d_PFi/d_PN;

          //================================== Compute face value
          double p_f_P = (1.0-rP)*p_P + (1.0-rP)*gradp_P.Dot(PF);

               a_P = a_P + A_f*n*p_f_P;
          auto a_N = A_f*(-1.0*n)*p_f_P;

          for (auto dim : dimensions)
            VecSetValue(v_gradp,jgradp[dim],a_N[dim],ADD_VALUES);
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
        VecSetValue(v_gradp,igradp[dim],a_P[dim],ADD_VALUES);
    }//for cells

    VecAssemblyBegin(v_gradp);
    VecAssemblyEnd(v_gradp);

    //**************************************************
    VecAssemblyBegin(v_cell_vols);
    VecAssemblyEnd(v_cell_vols);
    VecPointwiseDivide(v_gradp,v_gradp,v_cell_vols);
    //**************************************************

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
  VecDestroy(&v_cell_vols);
}