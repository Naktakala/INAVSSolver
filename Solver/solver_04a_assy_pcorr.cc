#include "solver.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

//###################################################################
/**Assembles the pressure system.*/
void INAVSSolver::AssemblePressureCorrectionSystem()
{
  //============================================= Reset matrix and RHS
  MatZeroEntries(A_pc);
  VecSet(b_pc, 0.0);

  //============================================= Get local views
  std::vector<Vec> x_a_PL(3);
  for (int dim : dimensions)
    VecGhostGetLocalForm(x_a_P[dim],&x_a_PL[dim]);

  std::vector<const double*> d_a_PL(3);
  for (int dim : dimensions)
    VecGetArrayRead(x_a_PL[dim],&d_a_PL[dim]);

  //============================================= Loop over cells
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    //====================================== Map row indices of unknowns
    int iu = fv_sdm.MapDOF(&cell,&uk_man_u,VELOCITY);
    int ip = fv_sdm.MapDOF(&cell,&uk_man_p,PRESSURE);

    //====================================== Get values
    chi_mesh::Vector3 a_P;
    for (int dim : dimensions)
      VecGetValues(x_a_P[dim] , 1, &iu, &a_P(dim));

    //====================================== Zero cell 0
    if (cell.global_id == 0)
    {
      MatSetValue(A_pc,ip,ip,1.0,ADD_VALUES);
      VecSetValue(b_pc,ip,   0.0,ADD_VALUES);
      continue;
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
        chi_mesh::Cell* adj_cell;
        if (face.IsNeighborLocal(grid))
          adj_cell = &grid->local_cells[face.GetNeighborLocalID(grid)];
        else
          adj_cell = grid->cells[face.neighbor];

        auto adj_cell_fv_view = fv_sdm.MapNeighborFeView(face.neighbor);

        double V_N = adj_cell_fv_view->volume;

        //============================= Map row indices of unknowns
        int lju    = fv_sdm.MapDOFLocal(adj_cell,&uk_man_u,VELOCITY);
        int jp     = fv_sdm.MapDOF(adj_cell,&uk_man_p,PRESSURE);

        //============================= Get Values
        chi_mesh::Vector3 a_N;

        for (int dim : dimensions)
          a_N(dim) = d_a_PL[dim][lju];

        //============================= Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;
        chi_mesh::Vector3 NF = face.centroid - adj_cell->centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //============================= Compute face values
        auto   a_f = (1.0-rP)*a_P + rP*a_N;
        double V_f = (1.0-rP)*V_P + rP*V_N;

        double a_f_avg = 0.0;
        for (int dim : dimensions)
          a_f_avg += a_f[dim];
        a_f_avg /= num_dimensions;

        //============================= Compute ds, a_inv_ds_inv
        chi_mesh::Vector3 ds = adj_cell->centroid - cell.centroid;
        auto a_ds = a_f_avg*ds;
        auto a_inv_ds_inv = a_ds.InverseZeroIfSmaller(1.0e-10);

        //============================= Develop diffusion entry
        double diffusion_entry = -(rho*alpha_u*V_f*(A_f*n).Dot(a_inv_ds_inv));

        if (face.neighbor == 0)
          VecSetValue(b_pc, ip, -diffusion_entry * 0.0, ADD_VALUES);
        else
          MatSetValue(A_pc, ip, jp, diffusion_entry, ADD_VALUES);
        MatSetValue(A_pc,ip,ip  ,-diffusion_entry,ADD_VALUES);

        //============================= Develop RHS
        double m_f = mass_fluxes[cell.local_id][f];
        VecSetValue(b_pc, ip, -m_f, ADD_VALUES);
      }//interior face
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Boundary face
      else
      {
        double a_f_avg = 0.0;
        for (int dim : dimensions)
          a_f_avg += a_P[dim];
        a_f_avg /= num_dimensions;

        //============================= Compute Area vector
        auto ds = face.centroid - cell.centroid;
        auto a_ds = a_f_avg*ds;
        auto a_inv_ds_inv = a_ds.InverseZeroIfSmaller(1.0e-10);

        //============================= Develop diffusion entry
        double diffusion_entry = -((rho*alpha_u*A_f*n).Dot(a_inv_ds_inv));

        //-A_f n dot s_inv (p_p - p_m)
//        if (cell.local_id == 0 and
//            face.normal.Dot(chi_mesh::Vector3(-1.0,0.0,0.0))>0.999)
//        {
//          VecSetValue(b_pc,ip,-diffusion_entry*0.0,ADD_VALUES);
//          MatSetValue(A_pc,ip,ip  ,-diffusion_entry,ADD_VALUES);
//        }

        //============================= Develop RHS
        double m_f = mass_fluxes[cell.local_id][f];
        VecSetValue(b_pc, ip, -m_f, ADD_VALUES);
      }//bndry faces
    }//for faces


  }//for cell

  //============================================= Restore local views
  for (int dim : dimensions)
    VecRestoreArrayRead(x_a_PL[dim],&d_a_PL[dim]);
  for (int dim : dimensions)
    VecGhostRestoreLocalForm(x_a_P[dim],&x_a_PL[dim]);

  //============================================= Assemble matrices globally
  MatAssemblyBegin(A_pc,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A_pc,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin(x_pc);
  VecAssemblyEnd(x_pc);

  VecAssemblyBegin(b_pc);
  VecAssemblyEnd(b_pc);
}