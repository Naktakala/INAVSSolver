#include "chi_runtime.h"

#include "chi_log.h"

#include "ChiMesh/chi_mesh.h"
#include "ChiMesh/MeshHandler/chi_meshhandler.h"
#include "ChiMesh/VolumeMesher/chi_volumemesher.h"
#include "ChiMesh/MeshContinuum/chi_meshcontinuum.h"
#include "ChiMesh/Cell/cell.h"

#include "ChiMath/SpatialDiscretization/FiniteVolume/fv.h"
#include "ChiMath/chi_math.h"
#include "ChiMath/UnknownManager/unknown_manager.h"

#include "ChiPhysics/chi_physics_namespace.h"
#include "ChiPhysics/FieldFunction/fieldfunction.h"

#include "ChiMath/PETScUtils/petsc_utils.h"

int main(int argc, char* argv[])
{
    auto& log = ChiLog::GetInstance();
    ChiTech::Initialize(argc,argv);

    //================================= Setup Mesh
    auto mesh_handler = chi_mesh::GetNewHandler();

    std::vector<double> verts;

    double L=10.0;
    int N=100;

    double ds = L/N;
    for (int i=0; i<=N; ++i)
        verts.push_back(i*ds);

    //int n = N;
    // chi_mesh::Create1DSlabMesh(verts); 
    int n = N*N;
    chi_mesh::Create2DOrthoMesh(verts,verts);
    // int n = N*N*N;
    // chi_mesh::Create3DOrthoMesh(verts,verts,verts);
    // mesh_handler->volume_mesher->options.partition_z = 4;
    mesh_handler->volume_mesher->Execute();

    auto grid = mesh_handler->GetGrid();

    //================================= Setup unknowns
    auto* uk_man = new chi_math::UnknownManager;

    auto FLUX        = uk_man->AddUnknown(chi_math::UnknownType::SCALAR);
    auto TEMPERATURE = uk_man->AddUnknown(chi_math::UnknownType::SCALAR);

    //================================= Setup Spatial Discretization
    SpatialDiscretization_FV sdm;
    sdm.AddViewOfLocalContinuum(grid);

    const int ndof_local = sdm.GetNumLocalDOFs(grid,uk_man);
    const int ndof_globl = sdm.GetNumGlobalDOFs(grid,uk_man);

    log.Log(LOG_0) << "Number of global unknowns = " << ndof_globl;

    //================================= Create matrices and vectors
    Mat A = chi_math::PETScUtils::CreateSquareMatrix(ndof_local,
                                                     ndof_globl);
    std::vector<int> nodal_nnz_in_diag;
    std::vector<int> nodal_nnz_off_diag;
    sdm.BuildSparsityPattern(grid, 
                             nodal_nnz_in_diag, 
                             nodal_nnz_off_diag,
                             uk_man);

    chi_math::PETScUtils::InitMatrixSparsity(A,nodal_nnz_in_diag,
                                               nodal_nnz_off_diag);

    Vec x,b;
    x = chi_math::PETScUtils::CreateVector(ndof_local,ndof_globl);
    b = chi_math::PETScUtils::CreateVector(ndof_local,ndof_globl);
    
    VecSet(x,0.0);
    VecSet(b,0.0);



    //================================= Assemble matrix and rhs
    MPI_Barrier(MPI_COMM_WORLD);
    log.Log(LOG_0) << "Assembling system locally..." << std::endl;
    double D = 1.0;
    double q = 5.0;
    double h = 1.0;
    double C = 1.0;
    for (auto& cell : grid->local_cells)
    {
        // log.Log(LOG_0) << "Cell " << cell.local_id;
        int i0            = sdm.MapDOF(&cell,uk_man,FLUX);
        int i1            = sdm.MapDOF(&cell,uk_man,TEMPERATURE);

        int jT_m = sdm.MapDOF(&cell,uk_man,TEMPERATURE);

        auto cell_fv_view = sdm.MapFeView(cell.local_id);


        VecSetValue(b,i0,q*cell_fv_view->volume,ADD_VALUES);
        VecSetValue(b,i1,h*cell_fv_view->volume,ADD_VALUES);

        // MatSetValue(A,i1,jT_m,1.0*cell_fv_view->volume,ADD_VALUES);
        MatSetValue(A,i1,i0,-1.0*cell_fv_view->volume,ADD_VALUES);

        int f=-1;
        for (auto& face : cell.faces)
        {
            ++f;
            if (grid->IsCellBndry(face.neighbor))
            {
                int jphi_m = sdm.MapDOF(&cell,uk_man,FLUX);
                int jT_m = sdm.MapDOF(&cell,uk_man,TEMPERATURE);

                auto ds = cell.centroid - face.centroid;
                auto A_f_D = cell_fv_view->face_area[f]*D;
                auto A_f_D_div_ds = chi_mesh::Vector3(
                    (std::fabs(ds.x)>1.0e-8)? A_f_D/ds.x : 0.0, 
                    (std::fabs(ds.y)>1.0e-8)? A_f_D/ds.y : 0.0, 
                    (std::fabs(ds.z)>1.0e-8)? A_f_D/ds.z : 0.0);

                MatSetValue(A,i0,jphi_m,-face.normal.Dot(A_f_D_div_ds),ADD_VALUES);

                MatSetValue(A,i1,jT_m  ,-face.normal.Dot(A_f_D_div_ds),ADD_VALUES);
            }
            else 
            {
                auto adj_cell = grid->cells[face.neighbor];
                int jphi_m = sdm.MapDOF(&cell   ,uk_man,FLUX);
                int jphi_p = sdm.MapDOF(adj_cell,uk_man,FLUX);

                int jT_m   = sdm.MapDOF(&cell   ,uk_man,TEMPERATURE);
                int jT_p   = sdm.MapDOF(adj_cell,uk_man,TEMPERATURE);

                auto ds = cell.centroid - face.centroid + 
                          face.centroid - adj_cell->centroid;
                auto A_f_D = cell_fv_view->face_area[f]*D;
                auto A_f_D_div_ds = chi_mesh::Vector3(
                    (std::fabs(ds.x)>1.0e-8)? A_f_D/ds.x : 0.0, 
                    (std::fabs(ds.y)>1.0e-8)? A_f_D/ds.y : 0.0, 
                    (std::fabs(ds.z)>1.0e-8)? A_f_D/ds.z : 0.0);

                double mat_entry = face.normal.Dot(A_f_D_div_ds);

                MatSetValue(A,i0,jphi_m,-mat_entry,ADD_VALUES);
                MatSetValue(A,i0,jphi_p, mat_entry,ADD_VALUES);

                MatSetValue(A,i1,jT_m,-mat_entry,ADD_VALUES);
                MatSetValue(A,i1,jT_p, mat_entry,ADD_VALUES);
            }
        }
    }

    log.Log(LOG_0) << "Assembling system globally..." << std::endl;

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    auto lin_solver = chi_math::PETScUtils::CreateCommonKrylovSolverSetup(
        A,            // Reference matrix
        "FV_solver",  // Solver name
        KSPGMRES,        // Solver type
        PCGAMG,       // Preconditioner type
        1.0e-6,       // Residual tolerance 
        1000);         // Maximum number of iterations

    log.Log(LOG_0) << "Solving...\n";
    KSPSolve(lin_solver.ksp,b,x);

    //================================= Post-solving
    KSPConvergedReason reason;
    KSPGetConvergedReason(lin_solver.ksp,&reason);
    log.Log(LOG_0)
        << "Done. Convergence code: " 
        << reason << " " 
        << chi_physics::GetPETScConvergedReasonstring(reason);

    //================================= Copy petsc vector to local
    std::vector<double> data;
    chi_math::PETScUtils::CopyVecToSTLvector(x,data,ndof_local);

    //================================= Attach to field function  
    auto ffphi = new chi_physics::FieldFunction(
        std::string("Phi"),              //Text name
        0,                                        //Number id
        chi_physics::FieldFunctionType::FV,       //Field function sd-method
        grid,                                     //Grid
        &sdm,                                      //Spatial DM
        1,                                        //Num components per set
        2,                                        //Num sets
        0,0,                                      //Ref component and set
        nullptr,                                  //DOF block address
        &data);                                   //Data

    ffphi->ExportToVTKFV(std::string("Zphi"),std::string("Phi"));

    auto ffT = new chi_physics::FieldFunction(
        std::string("Teemporature"),              //Text name
        0,                                        //Number id
        chi_physics::FieldFunctionType::FV,       //Field function sd-method
        grid,                                     //Grid
        &sdm,                                      //Spatial DM
        1,                                        //Num components per set
        2,                                        //Num sets
        0,1,                                      //Ref component and set
        nullptr,                                  //DOF block address
        &data);                                   //Data

    ffT->ExportToVTKFV(std::string("ZT"),std::string("Temp"));

    //================================= Finalize
    ChiTech::Finalize();

    return 0;
}
