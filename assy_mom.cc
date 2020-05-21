#include "chi_log.h"

#include "ChiMesh/chi_mesh.h"
#include "ChiMesh/MeshHandler/chi_meshhandler.h"
#include "ChiMesh/MeshContinuum/chi_meshcontinuum.h"

#include "ChiMath/SpatialDiscretization/FiniteVolume/fv.h"
#include "ChiMath/chi_math.h"
#include "ChiMath/UnknownManager/unknown_manager.h"

#include "ChiPhysics/chi_physics_namespace.h"

#include "ChiMath/PETScUtils/petsc_utils.h"

extern double U;
extern double mu;
extern double rho;
extern double dt;
extern double alpha_p;
extern double alpha_u;

void AssembleU2D(chi_mesh::MeshContinuum* grid,
                 SpatialDiscretization_FV& sdm,
                 chi_math::UnknownManager* uk_man_u,
                 chi_math::UnknownManager* uk_man_p,
                 const unsigned int VELOCITY,
                 const unsigned int PRESSURE,
                 std::vector<double>& boundary_p,
                 Mat A_u, Vec x_u, Vec b_u, Vec x_p,
                 std::vector<double>& a_p_x,
                 std::vector<double>& a_p_y)
{
  MatZeroEntries(A_u);
  VecSet(b_u,0.0);
  for (double& v : a_p_x) v=0.0;
  for (double& v : a_p_y) v=0.0;

  auto& log = ChiLog::GetInstance();

  const unsigned int U_X = 0;
  const unsigned int U_Y = 1;

  const double inv_alpha = 1.0/alpha_u;

  //====================================== Assembling
  MPI_Barrier(MPI_COMM_WORLD);
  int boundary_counter=-1;
  for (auto& cell : grid->local_cells)
  {
    auto cell_fv_view = sdm.MapFeView(cell.local_id);

    double V = cell_fv_view->volume;

    //======================================= Map row indices of unknowns
    int i0            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_X);
    int i1            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_Y);
    int ip            = sdm.MapDOF(&cell,uk_man_p,PRESSURE);

    double u_x_m; VecGetValues(x_u,1,&i0,&u_x_m);
    double u_y_m; VecGetValues(x_u,1,&i1,&u_y_m);
    double p_m;   VecGetValues(x_p,1,&ip,&p_m);

    //========================================= Initialize neighbor terms
    //                                          and indices
    std::vector<double> a_n_ux(cell.faces.size(),0.0);
    std::vector<double> a_n_uy(cell.faces.size(),0.0);
    std::vector<int>    i_n_ux(cell.faces.size(),-1);
    std::vector<int>    i_n_uy(cell.faces.size(),-1);

    //========================================= Time derivative
    VecSetValue(b_u,i0,(rho*V/dt)*u_x_m,ADD_VALUES);
    VecSetValue(b_u,i1,(rho*V/dt)*u_y_m,ADD_VALUES);

    a_p_x[ip] += rho*V/dt;
    a_p_y[ip] += rho*V/dt;

    //========================================= Face currents
    int f=-1;
    for (auto& face : cell.faces)
    {
      ++f;
      auto A_f = cell_fv_view->face_area[f];
      chi_mesh::Vector3& n = face.normal;

      if (not grid->IsCellBndry(face.neighbor))
      {
        auto adj_cell = grid->cells[face.neighbor];

        //===================== Map column/row indices of unknowns
        int ju_x_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_X);
        int ju_y_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_Y);
        int jp_p   = sdm.MapDOF(adj_cell,uk_man_p,PRESSURE);

        i_n_ux[f] = ju_x_p;
        i_n_uy[f] = ju_y_p;

        //===================== Get neighbor pressures and velocity
        double u_x_p; VecGetValues(x_u,1,&ju_x_p,&u_x_p);
        double u_y_p; VecGetValues(x_u,1,&ju_y_p,&u_y_p);
        double p_p;   VecGetValues(x_u,1,&jp_p,&p_p);

        double u_x_avg = 0.5*(u_x_p + u_x_m);
        double u_y_avg = 0.5*(u_y_p + u_y_m);
        double p_avg   = 0.5*(p_p + p_m);

        //===================== Compute ds
        chi_mesh::Vector3 ds = adj_cell->centroid - cell.centroid;
        auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

        //===================== Develop diffusion entry
        double diffusion_entry = -mu*((A_f*n).Dot(ds_inv));

        //-mu A_f n dot s_inv (u_x_p - u_x_m)
        a_n_ux[f] +=  diffusion_entry;
        a_p_x[ip] += -diffusion_entry;

        //-nu A_f n dot s_inv (u_y_p - u_y_m)
        a_n_uy[f] +=  diffusion_entry;
        a_p_y[ip] += -diffusion_entry;

        //===================== Develop convection entry
        double L = ds.Norm();
        double Pe_x = rho*n.x*u_x_avg*L/mu;
        double Pe_y = rho*n.y*u_y_avg*L/mu;

        double PL_x = (std::fabs(Pe_x) < 1.0e-12)? 0.5 :
                      (exp(0.5*Pe_x) - 1) / (exp(Pe_x)-1);
        double PL_y = (std::fabs(Pe_y) < 1.0e-12)? 0.5 :
                      (exp(0.5*Pe_y) - 1) / (exp(Pe_y)-1);

        double convection_entry_x = rho*u_x_avg*A_f*n.x;
        double convection_entry_y = rho*u_y_avg*A_f*n.y;

        a_n_ux[f] += convection_entry_x*PL_x;
        a_p_x[ip] += convection_entry_x * (1.0 - PL_x);

        a_n_uy[f] += convection_entry_y*PL_y;
        a_p_y[ip] += convection_entry_y * (1.0 - PL_y);

        //===================== Develop pressure entry
        // -1/rho A_f n dot P_avg
        VecSetValue(b_u,i0,-A_f*n.x*p_avg,ADD_VALUES);
        VecSetValue(b_u,i1,-A_f*n.y*p_avg,ADD_VALUES);
      }
      else
      {
        ++boundary_counter;
        //===================== Map column indices indices
        int ju_x_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_X);
        int ju_y_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_Y);
        int jp_m   = sdm.MapDOF(&cell   ,uk_man_p,PRESSURE);

        double p_p = boundary_p[boundary_counter];

        double u_x_avg = 0.0;
        double u_y_avg = 0.0;
        double p_avg   = p_m;
        // double p_avg   = 0.5*(p_p + p_m);

        if (face.normal.Dot(chi_mesh::Vector3(0.0,1.0,0.0))>0.999)
        {
            u_x_avg = U;
            u_y_avg = 0.0;
        }

        //===================== Compute Area vector
        auto ds = 2.0*(face.centroid - cell.centroid);
        auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

        //===================== Develop diffusion entry
        double diffusion_entry = -mu*((A_f*n).Dot(ds_inv));

        //-nu A_f n dot s_inv (u_x_p - u_x_m)
        VecSetValue(b_u,i0,-diffusion_entry*u_x_avg,ADD_VALUES);
        a_p_x[ip] += -diffusion_entry;

        //-nu A_f n dot s_inv (u_y_p - u_y_m)
        VecSetValue(b_u,i1,-diffusion_entry*u_y_avg,ADD_VALUES);
        a_p_y[ip] += -diffusion_entry;

        //===================== Develop convection entry
        double convection_entry_x = rho*u_x_avg*A_f*n.x;
        double convection_entry_y = rho*u_y_avg*A_f*n.y;

        VecSetValue(b_u,i0,-convection_entry_x*u_x_avg,ADD_VALUES);
        VecSetValue(b_u,i1,-convection_entry_y*u_y_avg,ADD_VALUES);

        //===================== Develop pressure entry
        // -1/rho A_f n dot P_avg
        VecSetValue(b_u,i0,-A_f*n.x*p_avg,ADD_VALUES);
        VecSetValue(b_u,i1,-A_f*n.y*p_avg,ADD_VALUES);
      }
    }//for face

    //=========================================== Set matrix values
    f=-1;
    for (auto& face : cell.faces)
    {
      MatSetValue(A_u,i0,i_n_ux[f],a_n_ux[f],ADD_VALUES);
      MatSetValue(A_u,i1,i_n_uy[f],a_n_uy[f],ADD_VALUES);
    }

    MatSetValue(A_u,i0,i0,a_p_x[ip]*inv_alpha,ADD_VALUES);
    MatSetValue(A_u,i1,i1,a_p_y[ip]*inv_alpha,ADD_VALUES);

    VecSetValue(b_u,i0,(1.0-alpha_u)*inv_alpha*a_p_x[ip]*u_x_m,ADD_VALUES);
    VecSetValue(b_u,i1,(1.0-alpha_u)*inv_alpha*a_p_y[ip]*u_y_m,ADD_VALUES);

  }

  log.Log(LOG_0VERBOSE_1) << "Assembling velocity system globally..." << std::endl;

  MatAssemblyBegin(A_u,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A_u,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin(x_u);
  VecAssemblyEnd(x_u);

  VecAssemblyBegin(b_u);
  VecAssemblyEnd(b_u);

  VecSet(x_u,0.0);
}