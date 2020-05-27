#include "chi_runtime.h"

#include "chi_log.h"

#include "ChiMesh/chi_mesh.h"
#include "ChiMesh/MeshHandler/chi_meshhandler.h"
#include "ChiMesh/SurfaceMesher/surfacemesher.h"
#include "ChiMesh/VolumeMesher/chi_volumemesher.h"
#include "chi_mpi.h"

extern ChiMPI& chi_mpi;

#include "Solver/solver.h"

double U=100.0;
double mu=0.01;
double rho=1.0;
double dt=0.01;
double L=0.1;
int N=50;
double alpha_p = 0.3;
double alpha_u = 0.7;



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

  if (chi_mpi.process_count == 4)
  {
    mesh_handler->surface_mesher->partitioning_x = 2;
    mesh_handler->surface_mesher->partitioning_y = 2;
    mesh_handler->surface_mesher->xcuts.push_back(L/2.0);
    mesh_handler->surface_mesher->ycuts.push_back(L/2.0);
  }
  if (chi_mpi.process_count == 2)
  {
    mesh_handler->surface_mesher->partitioning_x = 2;
    mesh_handler->surface_mesher->xcuts.push_back(L/2.0);
  }
  mesh_handler->volume_mesher->Execute();


  //================================= Setup Solver
  INAVSSolver* solver = new INAVSSolver;
  solver->options.steady = true;
  solver->Initialize();

  //================================= Execute Solver
  solver->Execute();

  //================================= Finalize
  ChiTech::Finalize();

  return 0;
}
