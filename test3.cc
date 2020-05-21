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

double U=10.0;
double mu=0.01;
double rho=1.0;
double dt=0.0001;
double L=0.1;
int N=40;
double alpha_p = 0.3;
double alpha_u = 1.0;

void AssembleU2D(chi_mesh::MeshContinuum* grid,
                 SpatialDiscretization_FV& sdm,
                 chi_math::UnknownManager* uk_man_u,
                 chi_math::UnknownManager* uk_man_p,
                 const unsigned int VELOCITY,
                 const unsigned int PRESSURE,
                 std::vector<double>& boundary_p,
                 Mat A_u, Vec x_u, Vec b_u, Vec x_p,
                 std::vector<double>& a_p_x,
                 std::vector<double>& a_p_y);

void AssembleP2D(chi_mesh::MeshContinuum* grid,
                 SpatialDiscretization_FV& sdm,
                 chi_math::UnknownManager* uk_man_u,
                 chi_math::UnknownManager* uk_man_p,
                 const unsigned int VELOCITY,
                 const unsigned int PRESSURE,
                 std::vector<double>& boundary_pc,
                 Mat A_p, Vec x_pc, Vec b_p, Vec x_u, Vec x_p,
                 std::vector<double>& a_p_x_terms,
                 std::vector<double>& a_p_y_terms);

void UpdateBoundaryPC(chi_mesh::MeshContinuum* grid,
                      SpatialDiscretization_FV& sdm,
                      chi_math::UnknownManager* uk_man_u,
                      chi_math::UnknownManager* uk_man_p,
                      const unsigned int VELOCITY,
                      const unsigned int PRESSURE,
                      std::vector<double>& boundary_pc,
                      Vec x_pc, Vec x_u,
                      std::vector<double>& a_p_x_terms,
                      std::vector<double>& a_p_y_terms);

void AssembleUC2D(chi_mesh::MeshContinuum* grid,
                  SpatialDiscretization_FV& sdm,
                  chi_math::UnknownManager* uk_man_u,
                  chi_math::UnknownManager* uk_man_p,
                  const unsigned int VELOCITY,
                  const unsigned int PRESSURE,
                  std::vector<double>& boundary_pc,
                  Vec x_u, Vec x_uc, Vec x_pc,
                 std::vector<double>& a_p_x_terms,
                 std::vector<double>& a_p_y_terms);

int main(int argc, char* argv[])
{
    auto& log = ChiLog::GetInstance();
    ChiTech::Initialize(argc,argv);

    //================================= Setup Mesh
    auto mesh_handler = chi_mesh::GetNewHandler();

    std::vector<double> verts;
    double ds = L/N;
    for (int i=0; i<=N; ++i)
        verts.push_back(i*ds);

    int n = N*N;
    chi_mesh::Create2DOrthoMesh(verts,verts);
    mesh_handler->volume_mesher->Execute();

    auto grid = mesh_handler->GetGrid();

    //================================= Setup unknowns
    auto* uk_man_u = new chi_math::UnknownManager;
    auto* uk_man_p = new chi_math::UnknownManager;


    auto VELOCITY = uk_man_u->AddUnknown(chi_math::UnknownType::VECTOR_2);
    auto PRESSURE = uk_man_p->AddUnknown(chi_math::UnknownType::SCALAR);

    int num_bndry_pc=0;
    for (auto& cell : grid->local_cells)
        for (auto& face : cell.faces)
            if (grid->IsCellBndry(face.neighbor))
                ++num_bndry_pc;
                
    std::vector<double> boundary_pc(num_bndry_pc,0.0);
    std::vector<double> boundary_p(num_bndry_pc,0.0);

    //================================= Setup Spatial Discretization
    SpatialDiscretization_FV sdm;
    sdm.AddViewOfLocalContinuum(grid);

    const int ndof_local_u = sdm.GetNumLocalDOFs(grid ,uk_man_u);
    const int ndof_globl_u = sdm.GetNumGlobalDOFs(grid,uk_man_u);

    const int ndof_local_p = sdm.GetNumLocalDOFs(grid ,uk_man_p);
    const int ndof_globl_p = sdm.GetNumGlobalDOFs(grid,uk_man_p);

    log.Log(LOG_0) << "Number of velocity unknowns = " << ndof_globl_u;
    log.Log(LOG_0) << "Number of pressure unknowns = " << ndof_globl_p;

    //================================= Create matrices and vectors
    Mat A_u = chi_math::PETScUtils::CreateSquareMatrix(ndof_local_u,
                                                       ndof_globl_u);                                                                                                       
    Mat A_p = chi_math::PETScUtils::CreateSquareMatrix(ndof_local_p,
                                                       ndof_globl_p);                                                       
    std::vector<int> nodal_nnz_in_diag;
    std::vector<int> nodal_nnz_off_diag;
    sdm.BuildSparsityPattern(grid, 
                             nodal_nnz_in_diag, 
                             nodal_nnz_off_diag,
                             uk_man_u);

    chi_math::PETScUtils::InitMatrixSparsity(A_u,nodal_nnz_in_diag,
                                                 nodal_nnz_off_diag);                                                                                               

    sdm.BuildSparsityPattern(grid, 
                             nodal_nnz_in_diag, 
                             nodal_nnz_off_diag,
                             uk_man_p);

    chi_math::PETScUtils::InitMatrixSparsity(A_p,nodal_nnz_in_diag,
                                                 nodal_nnz_off_diag);

    Vec x_u,b_u;
    x_u = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);
    b_u = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);

    Vec x_uc;
    x_uc = chi_math::PETScUtils::CreateVector(ndof_local_u,ndof_globl_u);

    Vec x_p,x_pc,b_p;
    x_p = chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_globl_p);
    x_pc= chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_local_p);
    b_p = chi_math::PETScUtils::CreateVector(ndof_local_p,ndof_globl_p);
    
    VecSet(x_u,0.0);
    VecSet(b_u,0.0);

    VecSet(x_uc,0.0);

    VecSet(x_p ,0.0);
    VecSet(x_pc,0.0);
    VecSet(b_p ,0.0);

    std::vector<double> a_p_x_terms(ndof_globl_u,0.0);
    std::vector<double> a_p_y_terms(ndof_globl_u,0.0);

    //================================= Initialize linear solvers
    auto lin_solver_u = chi_math::PETScUtils::CreateCommonKrylovSolverSetup(
        A_u,            // Reference matrix
        "Momentum_solver",  // Solver name
        KSPGMRES,        // Solver type
        PCNONE,       // Preconditioner type
        1.0e-6,       // Residual tolerance 
        200);         // Maximum number of iterations 

    auto lin_solver_p = chi_math::PETScUtils::CreateCommonKrylovSolverSetup(
        A_p,            // Reference matrix
        "P_solver",  // Solver name
        KSPCG,        // Solver type
        PCGAMG,       // Preconditioner type
        1.0e-6,       // Residual tolerance 
        200);         // Maximum number of iterations   


    if (log.GetVerbosity() == LOG_0VERBOSE_0)
    {
        KSPMonitorCancel(lin_solver_u.ksp);
        KSPMonitorCancel(lin_solver_p.ksp);
    }

    //================================= Start iterative process
    for (int i=0; i<20; ++i)
    {
        AssembleU2D(grid, sdm, uk_man_u, uk_man_p, VELOCITY, PRESSURE, 
                    boundary_p,
                    A_u, x_u, b_u,x_p,
                    a_p_x_terms,a_p_y_terms);  

        log.Log(LOG_0VERBOSE_1) << "Solving Velocity...\n";
        KSPSetOperators(lin_solver_u.ksp,A_u,A_u);
        KSPSolve(lin_solver_u.ksp,b_u,x_u);

        AssembleP2D(grid, sdm, uk_man_u,uk_man_p,VELOCITY,PRESSURE,
                    boundary_pc, A_p, x_pc, b_p, x_u, x_p,
                    a_p_x_terms,a_p_y_terms);

        log.Log(LOG_0VERBOSE_1) << "Solving Pressure correction...\n";
        KSPSetOperators(lin_solver_p.ksp,A_p,A_p);
        KSPSolve(lin_solver_p.ksp,b_p,x_pc); 

        UpdateBoundaryPC(grid,sdm,uk_man_u,uk_man_p,VELOCITY,PRESSURE,
                         boundary_pc,x_pc,x_u,
                         a_p_x_terms,a_p_y_terms);

        // VecScale(x_pc,alpha_p);
        double ref_p; int ip=0; VecGetValues(x_pc,1,&ip,&ref_p);
        VecShift(x_pc,-ref_p);
        VecAXPY(x_p,alpha_p,x_pc);

        for (int i=0; i<boundary_pc.size(); ++i)
            boundary_p[i] += boundary_pc[i]*alpha_p;

        //================================= Compute velocity correction
        AssembleUC2D(grid,sdm,uk_man_u,uk_man_p,VELOCITY,PRESSURE,
                    boundary_pc,x_u,x_uc,x_pc,
                    a_p_x_terms,a_p_y_terms);
        VecAXPY(x_u,alpha_u,x_uc);

        //================================= Logs
        {
            double max_v = 0.0; VecMax(x_u,NULL,&max_v);
            double max_p = 0.0; VecMax(x_p,NULL,&max_p);
            double min_p = 0.0; VecMin(x_p,NULL,&min_p);
            double norm_v = 0.0; VecNorm(x_u,NORM_2,&norm_v);
            double norm_vc = 0.0; VecNorm(x_uc, NORM_2, &norm_vc);

            char buf[200];
            sprintf(buf,"Iteration %4d max_v=%.7f max_p=%+.6f "
                        "min_p=%+.6f norm_v=%.6f norm_vc=%.6f", -1,
                        max_v,max_p,min_p,norm_v,norm_vc);
            log.Log(LOG_0) << buf;
        }
        
    }

    //================================= Copy petsc vector to local
    std::vector<double> data;
    chi_math::PETScUtils::CopyVecToSTLvector(x_u,data,ndof_local_u);
    std::vector<double> data_p;
    chi_math::PETScUtils::CopyVecToSTLvector(x_p,data_p,ndof_local_p);

    //================================= Attach to field function  
    auto ffu = new chi_physics::FieldFunction(
        std::string("U"),              //Text name
        0,                                        //Number id
        chi_physics::FieldFunctionType::FV,       //Field function sd-method
        grid,                                     //Grid
        &sdm,                                      //Spatial DM
        2,                                        //Num components per set
        1,                                        //Num sets
        0,0,                                      //Ref component and set
        nullptr,                                  //DOF block address
        &data);                                   //Data

    ffu->ExportToVTKFVG(std::string("ZU_X"),std::string("U_X"));


    auto ffP = new chi_physics::FieldFunction(
        std::string("Pressure"),              //Text name
        0,                                        //Number id
        chi_physics::FieldFunctionType::FV,       //Field function sd-method
        grid,                                     //Grid
        &sdm,                                      //Spatial DM
        1,                                        //Num components per set
        1,                                        //Num sets
        0,0,                                      //Ref component and set
        nullptr,                                  //DOF block address
        &data_p);                                   //Data

    ffP->ExportToVTKFV(std::string("ZP"),std::string("PressureCorrection"));

    log.Log(LOG_0) 
        << "Cell 0 location: " 
        << grid->local_cells[0].centroid.PrintS(); 
    //================================= Finalize
    ChiTech::Finalize();

    return 0;
}
