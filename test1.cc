#include "chi_runtime.h"

#include "ChiMesh/chi_mesh.h"
#include "ChiMesh/MeshHandler/chi_meshhandler.h"
#include "ChiMesh/VolumeMesher/chi_volumemesher.h"
#include "ChiMesh/MeshContinuum/chi_meshcontinuum.h"
#include "ChiMesh/Cell/cell.h"

#include "ChiMath/SpatialDiscretization/FiniteVolume/fv.h"
#include "ChiMath/chi_math.h"

int main(int argc, char* argv[])
{
    ChiTech::Initialize(argc,argv);

    //================================= Create a mesh
    auto mesh_handler = chi_mesh::GetNewHandler();

    std::vector<double> verts;

    double L=10.0;
    int N=5;

    double ds = L/N;
    for (int i=0; i<=N; ++i)
        verts.push_back(i*ds);

    //int n = N;
    // chi_mesh::Create1DSlabMesh(verts); 
    // int n = N*N;
    // chi_mesh::Create2DOrthoMesh(verts,verts);
    int n = N*N*N;
    chi_mesh::Create3DOrthoMesh(verts,verts,verts);
    mesh_handler->volume_mesher->Execute();

    auto grid = mesh_handler->GetGrid();

    //================================= Set material IDs
    for (auto& cell : grid->local_cells)
        cell.material_id = 0;

    //================================= Setup Spatial Discretization
    SpatialDiscretization_FV sdm;
    sdm.AddViewOfLocalContinuum(grid);

    //================================= Create matrices and vectors
    typedef std::vector<double> VecDbl;
    typedef std::vector<VecDbl> MatDbl;

    VecDbl x(n,0.0);
    VecDbl b(n,0.0);
    MatDbl A(n,VecDbl(n,0.0));

    //================================= Assemble matrix and rhs
    std::cout << "Assembling system..." << std::endl;
    double D = 1.0;
    double q = 1.0;
    for (auto& cell : grid->local_cells)
    {
        int i = cell.local_id;

        // std::cout << "Cell " << i << std::endl;

        auto cell_fv_view = sdm.MapFeView(cell.local_id);

        b[i] = q*cell_fv_view->volume;

        int f=-1;
        for (auto& face : cell.faces)
        {
            ++f;
            // std::cout << "Face " << f << " " << face.neighbor <<std::endl;
            if (grid->IsCellBndry(face.neighbor))
            {
                auto ds = cell.centroid - face.centroid;
                auto A_f_D = cell_fv_view->face_area[f]*D;
                auto A_f_D_div_ds = chi_mesh::Vector3(
                    (std::fabs(ds.x)>1.0e-8)? A_f_D/ds.x : 0.0, 
                    (std::fabs(ds.y)>1.0e-8)? A_f_D/ds.y : 0.0, 
                    (std::fabs(ds.z)>1.0e-8)? A_f_D/ds.z : 0.0);

                A[i][i] += -face.normal.Dot(A_f_D_div_ds);
            }
            else 
            {
                int j = face.neighbor;
                auto adj_cell = grid->cells[face.neighbor];

                auto ds = cell.centroid - face.centroid + 
                          face.centroid - adj_cell->centroid;
                auto A_f_D = cell_fv_view->face_area[f]*D;
                auto A_f_D_div_ds = chi_mesh::Vector3(
                    (std::fabs(ds.x)>1.0e-8)? A_f_D/ds.x : 0.0, 
                    (std::fabs(ds.y)>1.0e-8)? A_f_D/ds.y : 0.0, 
                    (std::fabs(ds.z)>1.0e-8)? A_f_D/ds.z : 0.0);

                A[i][i] += -face.normal.Dot(A_f_D_div_ds);
                A[i][j] +=  face.normal.Dot(A_f_D_div_ds);
            }
        }
    }

    std::cout << "Solving..." << std::endl;
    chi_math::GaussElimination(A,b,n);

    for (auto val : b)
        std::cout << val << "\n";

    //================================= Finalize
    ChiTech::Finalize();
    return 0;
}