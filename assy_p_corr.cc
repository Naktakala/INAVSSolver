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

void AssembleP2D(chi_mesh::MeshContinuum* grid,
                 SpatialDiscretization_FV& sdm,
                 chi_math::UnknownManager* uk_man_u,
                 chi_math::UnknownManager* uk_man_p,
                 const unsigned int VELOCITY,
                 const unsigned int PRESSURE,
                 std::vector<double>& boundary_pc,
                 Mat A_p, Vec x_pc, Vec b_p, Vec x_u, Vec x_p,
                 std::vector<double>& a_p_x_terms,
                 std::vector<double>& a_p_y_terms)
{
    MatZeroEntries(A_p);
    VecSet(b_p,0.0);

    auto& log = ChiLog::GetInstance();

    const unsigned int U_X = 0;
    const unsigned int U_Y = 1;

    //====================================== Computing pressure gradients
    std::vector<double> p_grad_x(grid->local_cells.size(),0.0);
    std::vector<double> p_grad_y(grid->local_cells.size(),0.0);

    for (auto& cell : grid->local_cells)
    {
        auto cell_fv_view = sdm.MapFeView(cell.local_id);

        double V = cell_fv_view->volume;

        int ip = sdm.MapDOF(&cell,uk_man_p,PRESSURE);
        double p_m;   VecGetValues(x_p,1,&ip,&p_m);

        int f=-1;
        for (auto& face : cell.faces)
        {
            ++f;
            double A_f = cell_fv_view->face_area[f];
            chi_mesh::Vector3& n = face.normal;

            if (not grid->IsCellBndry(face.neighbor))
            {
                auto adj_cell = grid->cells[face.neighbor];
                int jp_p = sdm.MapDOF(adj_cell,uk_man_p,PRESSURE);

                double p_p; VecGetValues(x_p,1,&jp_p,&p_p);
                double p_avg = 0.5*(p_p + p_m);

                p_grad_x[ip] += (1.0/V)*A_f*n.x*p_avg;
                p_grad_y[ip] += (1.0/V)*A_f*n.y*p_avg;
            }
            else
            {
                double p_avg = 0.5*(0.0 + p_m);

                p_grad_x[ip] += (1.0/V)*A_f*n.x*p_avg;
                p_grad_y[ip] += (1.0/V)*A_f*n.y*p_avg;
            }
            
        }
    }

    //====================================== Assembling
    MPI_Barrier(MPI_COMM_WORLD);
    log.Log(LOG_0VERBOSE_1) << "Assembling pressure system locally..." << std::endl;


    int global_neighbor_count=-1;
    int boundary_counter=-1;
    for (auto& cell : grid->local_cells)
    {
        auto cell_fv_view = sdm.MapFeView(cell.local_id);

        double V = cell_fv_view->volume;

        //======================================= Map row indices of unknowns
        int i0            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_X);
        int i1            = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_Y);
        int ip            = sdm.MapDOF(&cell,uk_man_p,PRESSURE);

        double u_x_m;  VecGetValues(x_u,1,&i0,&u_x_m);
        double u_y_m;  VecGetValues(x_u,1,&i1,&u_y_m);
        double p_m;    VecGetValues(x_pc,1,&ip,&p_m);
        double p_mold; VecGetValues(x_p,1,&ip,&p_mold);
        double p_grad_x_m = p_grad_x[ip];
        double p_grad_y_m = p_grad_y[ip];

        // if (cell.local_id == 0)
        // {
        //     MatSetValue(A_p,ip,ip, 1.0,ADD_VALUES);
        //     VecSetValue(b_p,ip, 0.0,ADD_VALUES);
        //     continue;
        // }

        int f=-1;
        for (auto& face : cell.faces)
        {
            ++f;
            double A_f = cell_fv_view->face_area[f];
            chi_mesh::Vector3& n = face.normal;

            if (not grid->IsCellBndry(face.neighbor))
            {
                auto adj_cell = grid->cells[face.neighbor];

                //===================== Map column/row indices of unknowns
                int ju_x_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_X);
                int ju_x_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_X);   

                int ju_y_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_Y);
                int ju_y_p = sdm.MapDOF(adj_cell,uk_man_u,VELOCITY, U_Y);

                int jp_m = sdm.MapDOF(&cell   ,uk_man_p,PRESSURE);
                int jp_p = sdm.MapDOF(adj_cell,uk_man_p,PRESSURE);

                //===================== Compute Area vector
                auto ds = adj_cell->centroid - cell.centroid;
                auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

                //===================== Get neighbor velocities
                double u_x_p; VecGetValues(x_u,1,&ju_x_p,&u_x_p);
                double u_y_p; VecGetValues(x_u,1,&ju_y_p,&u_y_p);
                double p_pold;   VecGetValues(x_p,1,&jp_p,&p_pold);
                double p_grad_x_p = p_grad_x[jp_p];
                double p_grad_y_p = p_grad_y[jp_p];
                
                double u_x_avg = 0.5*(u_x_p + u_x_m);
                double u_y_avg = 0.5*(u_y_p + u_y_m);

                // u_x_avg += (-A_f*n.x/a_p_x_terms[ip])*
                //            (ds_inv.x*(p_pold-p_mold) 
                //            +0.5*(p_grad_x_p + p_grad_x_m));
                // u_x_avg += (-A_f*n.x/a_p_x_terms[ip])*
                //            (ds_inv.x*(p_pold-p_mold) 
                //            +0.5*(p_grad_y_p + p_grad_y_m));                           


                //===================== Develop diffusion entry
                double diffusion_entry = -(A_f*n).Dot(ds_inv);

                MatSetValue(A_p,ip,jp_p, diffusion_entry,ADD_VALUES);
                MatSetValue(A_p,ip,jp_m,-diffusion_entry,ADD_VALUES);

                //===================== Develop RHS
                VecSetValue(b_p,ip,-A_f*n.x*a_p_x_terms[ip]*u_x_avg,ADD_VALUES);
                VecSetValue(b_p,ip,-A_f*n.y*a_p_y_terms[ip]*u_y_avg,ADD_VALUES);
            }
            else
            {
                //===================== Map column/row indices of unknowns
                int ju_x_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_X);
                int ju_y_m = sdm.MapDOF(&cell   ,uk_man_u,VELOCITY, U_Y);
                int jp_m   = sdm.MapDOF(&cell   ,uk_man_p,PRESSURE);
                
                //===================== Get neighbor velocities
                ++boundary_counter;
                double u_x_avg = 0.0;
                double u_y_avg = 0.0;
                // double p_p = boundary_pc[boundary_counter];

                if (face.normal.Dot(chi_mesh::Vector3(0.0,1.0,0.0))>0.999)
                {
                    u_x_avg = U;
                    u_y_avg = 0.0;
                }
                

                //===================== Compute Area vector
                auto ds = 2.0*(face.centroid - cell.centroid);
                auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

                double n_dot_grad = n.Dot(ds_inv);

                double n_dot_apu = a_p_x_terms[ip]*u_x_m*n.x + 
                                   a_p_y_terms[ip]*u_y_m*n.y;

                double p_p = (n_dot_apu/n_dot_grad);
                // p_p = 0.0;

                //===================== Develop diffusion entry
                // double diffusion_entry = -((A_f*n).Dot(ds_inv));

                // VecSetValue(b_p,ip,-diffusion_entry*p_p,ADD_VALUES);
                // MatSetValue(A_p,ip,jp_m,-diffusion_entry,ADD_VALUES);



                //===================== Add to rhs
                VecSetValue(b_p,ip,-A_f*n.x*a_p_x_terms[ip]*u_x_avg,ADD_VALUES);
                VecSetValue(b_p,ip,-A_f*n.y*a_p_y_terms[ip]*u_y_avg,ADD_VALUES);
            }
            
        }
    }

    log.Log(LOG_0VERBOSE_1) << "Assembling pressure system globally..." << std::endl;

    MatAssemblyBegin(A_p,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_p,MAT_FINAL_ASSEMBLY);

    VecAssemblyBegin(x_pc);
    VecAssemblyEnd(x_pc);

    VecAssemblyBegin(b_p);
    VecAssemblyEnd(b_p);

    VecSet(x_pc,0.0);
}

void UpdateBoundaryPC(chi_mesh::MeshContinuum* grid,
                      SpatialDiscretization_FV& sdm,
                      chi_math::UnknownManager* uk_man_u,
                      chi_math::UnknownManager* uk_man_p,
                      const unsigned int VELOCITY,
                      const unsigned int PRESSURE,
                      std::vector<double>& boundary_pc,
                      Vec x_pc, Vec x_u,
                      std::vector<double>& a_p_x_terms,
                      std::vector<double>& a_p_y_terms)
{
    int boundary_counter=-1;
    for (auto& cell : grid->local_cells)
    {
        const unsigned int U_X = 0;
        const unsigned int U_Y = 1;

        int i0 = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_X);
        int i1 = sdm.MapDOF(&cell,uk_man_u,VELOCITY, U_Y);
        int ip = sdm.MapDOF(&cell,uk_man_p,PRESSURE);

        double u_x_m; VecGetValues(x_u,1,&i0,&u_x_m);
        double u_y_m; VecGetValues(x_u,1,&i1,&u_y_m);
        double p_m;   VecGetValues(x_pc,1,&ip,&p_m);

        for (auto& face : cell.faces)
            if (grid->IsCellBndry(face.neighbor))
            {
                chi_mesh::Vector3& n = face.normal;
                auto ds = face.centroid - cell.centroid;
                auto ds_inv = ds.InverseZeroIfGreater(1.0e-10);

                double n_dot_grad = n.Dot(ds_inv);

                double n_dot_apu = a_p_x_terms[ip]*u_x_m*n.x + 
                                   a_p_y_terms[ip]*u_y_m*n.y;

                double p_p = p_m - (n_dot_apu/n_dot_grad);

                ++boundary_counter;
                boundary_pc[boundary_counter] = p_p;
            }
    }
}