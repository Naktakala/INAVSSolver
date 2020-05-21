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
  MatZeroEntries(A_pc);
  VecSet(b_p,0.0);

  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = fv_sdm.MapFeView(cell.local_id);

    double V_P = cell_fv_view->volume;

    auto& a_P = momentum_coeffs[cell.local_id].a_P;

    //======================================= Map row indices of unknowns
    int ip = fv_sdm.MapDOF(cell.global_id,&uk_man_p,PRESSURE);

    //======================================= Face terms
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

        auto adj_cell_fv_view = fv_sdm.MapFeView(adj_cell->local_id); //TODO: !!

        double V_N = adj_cell_fv_view->volume;

        auto& a_N = momentum_coeffs[adj_cell->local_id].a_P;

        //======================================= Map row indices of unknowns
        int jp_p   = fv_sdm.MapDOF(face.neighbor,&uk_man_p,PRESSURE);

        //================================== Compute vectors
        chi_mesh::Vector3 PN = adj_cell->centroid - cell.centroid;
        chi_mesh::Vector3 PF = face.centroid - cell.centroid;
        chi_mesh::Vector3 NF = face.centroid - adj_cell->centroid;

        double d_PN = PN.Norm();

        chi_mesh::Vector3 e_PN = PN/d_PN;

        double d_PFi = PF.Dot(e_PN);

        double rP = d_PFi/d_PN;

        //===================== Compute face values
        auto   a_f = (1.0-rP)*a_P + rP*a_N;
        double V_f = (1.0-rP)*V_P + rP*V_N;

        //===================== Compute ds, a_inv_ds_inv
        chi_mesh::Vector3 ds = adj_cell->centroid - cell.centroid;
        auto a_ds = a_f*ds;
        auto a_inv_ds_inv = a_ds.InverseZeroIfSmaller(1.0e-10);

        //===================== Develop diffusion entry
        double diffusion_entry = -(rho*alpha_u*V_f*(A_f*n).Dot(a_inv_ds_inv));

        MatSetValue(A_pc,ip,jp_p, diffusion_entry,ADD_VALUES);
        MatSetValue(A_pc,ip,ip  ,-diffusion_entry,ADD_VALUES);

        //===================== Develop RHS
        double m_f = mass_fluxes[cell.local_id][f];
        VecSetValue(b_p,ip,-m_f,ADD_VALUES);
      }//interior face
      else
      {
        //===================== Compute Area vector
        auto ds = face.centroid - cell.centroid;
        auto a_ds = a_P*ds;
        auto a_inv_ds_inv = a_ds.InverseZeroIfSmaller(1.0e-10);

        //===================== Develop diffusion entry
        double diffusion_entry = -((rho*alpha_u*A_f*n).Dot(a_inv_ds_inv));

        //-A_f n dot s_inv (p_p - p_m)
//        if (cell.local_id == 0 and
//            face.normal.Dot(chi_mesh::Vector3(-1.0,0.0,0.0))>0.999)
//        {
//          VecSetValue(b_p,ip,-diffusion_entry*0.0,ADD_VALUES);
//          MatSetValue(A_pc,ip,ip  ,-diffusion_entry,ADD_VALUES);
//        }

        //===================== Develop RHS
        double m_f = mass_fluxes[cell.local_id][f];
        VecSetValue(b_p,ip,-m_f,ADD_VALUES);
      }//bndry faces
    }//for faces


  }//for cell


  //======================================== Assemble matrices globally
  MatAssemblyBegin(A_pc,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A_pc,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin(x_pc);
  VecAssemblyEnd(x_pc);

  VecAssemblyBegin(b_p);
  VecAssemblyEnd(b_p);
}