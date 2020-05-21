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

void AssembleUC2D(chi_mesh::MeshContinuum* grid,
                  SpatialDiscretization_FV& sdm,
                  chi_math::UnknownManager* uk_man_u,
                  chi_math::UnknownManager* uk_man_p,
                  const unsigned int VELOCITY,
                  const unsigned int PRESSURE,
                  std::vector<double>& boundary_pc,
                  Vec x_u, Vec x_uc, Vec x_pc,
                 std::vector<double>& a_p_x_terms,
                 std::vector<double>& a_p_y_terms)
{
    VecSet(x_uc,0.0);
    
    auto& log = ChiLog::GetInstance();

    const unsigned int U_X = 0;
    const unsigned int U_Y = 1;

    //================================= Compute values
    int global_neighbor_count=-1;
    int boundary_counter=-1;
    for (auto& cell : grid->local_cells)
    {
        auto cell_fv_view = sdm.MapFeView(cell.local_id);

        double V = cell_fv_view->volume;
        double inv_V = 1.0/cell_fv_view->volume;

        int i0            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_X);
        int i1            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_Y);
        int ip            = sdm.MapDOF(&cell,uk_man_p,PRESSURE);

        double u_x_m; VecGetValues(x_u,1,&i0,&u_x_m);
        double u_y_m; VecGetValues(x_u,1,&i1,&u_y_m);       
        double p_m; VecGetValues(x_pc,1,&ip  ,&p_m);

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
                int ju_x_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_X);
                int ju_x_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_X);   

                int ju_y_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_Y);
                int ju_y_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_Y);

                int jp_m   = sdm.MapDOF(&cell   ,uk_man_p,PRESSURE);
                int jp_p   = sdm.MapDOF(adj_cell,uk_man_p,PRESSURE);  

                //===================== Get neighbor pressure
                double u_x_p; VecGetValues(x_u,1,&ju_x_p,&u_x_p);
                double u_y_p; VecGetValues(x_u,1,&ju_y_p,&u_y_p);
                double p_p; VecGetValues(x_pc,1,&jp_p,&p_p);
                
                double u_x_avg = 0.5*(u_x_p + u_x_m);
                double u_y_avg = 0.5*(u_y_p + u_y_m);
                double p_avg = 0.5*(p_p + p_m);

                //===================== Compute Area vector
                auto ds = adj_cell->centroid - cell.centroid;
                auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

                VecSetValue(x_uc,i0,-inv_V*(1.0/a_p_x_terms[ip])*A_f*n.x*p_avg,ADD_VALUES);
                VecSetValue(x_uc,i1,-inv_V*(1.0/a_p_y_terms[ip])*A_f*n.y*p_avg,ADD_VALUES);
            }
            else
            {
                ++boundary_counter;
                //===================== Map column/row indices of unknowns
                int ju_x_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_X);  
                int ju_y_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_Y);
                int jp_m   = sdm.MapDOF(&cell   ,uk_man_p,PRESSURE); 

                double u_x_avg = 0.0;
                double u_y_avg = 0.0;

                //===================== Compute Area vector
                auto ds = face.centroid - cell.centroid;
                auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

                //===================== Get neighbor pressure
                double p_p = boundary_pc[boundary_counter];
                // p_p = 0.0;
                double p_avg = 0.5*(p_p+p_m);
                p_avg = p_m;

                VecSetValue(x_uc,i0,-inv_V*(1.0/a_p_x_terms[ip])*A_f*n.x*p_avg,ADD_VALUES);
                VecSetValue(x_uc,i1,-inv_V*(1.0/a_p_y_terms[ip])*A_f*n.y*p_avg,ADD_VALUES);
            }
        }
    }

    log.Log(LOG_0VERBOSE_1) << "Assembling velocity correction system globally..." << std::endl;

    VecAssemblyBegin(x_uc);
    VecAssemblyEnd(x_uc);
}