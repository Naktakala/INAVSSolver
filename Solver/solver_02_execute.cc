#include "solver.h"

#include "ChiPhysics/FieldFunction/fieldfunction.h"

#include "chi_log.h"

#include "ChiTimer/chi_timer.h"
extern ChiTimer chi_program_timer;

//###################################################################
/**Executes the solver.*/
void INAVSSolver::Execute()
{
  auto& log = ChiLog::GetInstance();
  log.Log(LOG_0) << "Executing Incompressible Navier Stokes Solver";

  size_t tag_gradP_gg = log.GetRepeatingEventTag("GradP_GG           ");
  size_t tag_gradU    = log.GetRepeatingEventTag("GradP_U            ");
  size_t tag_mom_assy = log.GetRepeatingEventTag("Momentum Assembly  ");
  size_t tag_mom_slv0 = log.GetRepeatingEventTag("Momentum solve - 0 ");
  size_t tag_mom_slv1 = log.GetRepeatingEventTag("Momentum solve - 1 ");
  size_t tag_comp_mf  = log.GetRepeatingEventTag("Mass flux          ");
  size_t tag_pc_assy  = log.GetRepeatingEventTag("PCorr Assembly     ");
  size_t tag_pc_slv0  = log.GetRepeatingEventTag("PCorr solve - 0    ");
  size_t tag_pc_slv1  = log.GetRepeatingEventTag("PCorr solve - 1    ");
  size_t tag_gradP_pc = log.GetRepeatingEventTag("GradP_PC           ");
  size_t tag_corr     = log.GetRepeatingEventTag("Corrections        ");

  for (int i=0; i<1000; ++i)
  {
    log.LogEvent(tag_gradP_gg,ChiLog::EventType::EVENT_BEGIN);
    ComputeGradP_GG(x_gradp, x_p);
    log.LogEvent(tag_gradP_gg,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_gradU,ChiLog::EventType::EVENT_BEGIN);
    if (i>0)
      ComputeGradU2();
    log.LogEvent(tag_gradU,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_mom_assy,ChiLog::EventType::EVENT_BEGIN);
    AssembleMomentumSystem();
    log.LogEvent(tag_mom_assy,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_mom_slv0,ChiLog::EventType::EVENT_BEGIN);
    for (int dim : dimensions)
      KSPSetOperators(lin_solver_u[dim].ksp,A_u[dim],A_u[dim]);
    log.LogEvent(tag_mom_slv0,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_mom_slv1,ChiLog::EventType::EVENT_BEGIN);
    for (int dim : dimensions)
      KSPSolve(lin_solver_u[dim].ksp,b_u[dim],x_u[dim]);
    log.LogEvent(tag_mom_slv1,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_comp_mf,ChiLog::EventType::EVENT_BEGIN);
    for (int dim : dimensions)
      VecGhostUpdateBegin(x_u[dim],INSERT_VALUES,SCATTER_FORWARD);
    for (int dim : dimensions)
      VecGhostUpdateEnd  (x_u[dim],INSERT_VALUES,SCATTER_FORWARD);
    ComputeMassFlux();
    log.LogEvent(tag_comp_mf,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_pc_assy,ChiLog::EventType::EVENT_BEGIN);
    AssemblePressureCorrectionSystem();
    log.LogEvent(tag_pc_assy,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_pc_slv0,ChiLog::EventType::EVENT_BEGIN);
    KSPSetOperators(lin_solver_p.ksp,A_pc,A_pc);
    log.LogEvent(tag_pc_slv0,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_pc_slv1,ChiLog::EventType::EVENT_BEGIN);
    VecSet(x_pc,0.0);
    KSPSolve(lin_solver_p.ksp, b_pc, x_pc);

    VecGhostUpdateBegin(x_pc,INSERT_VALUES,SCATTER_FORWARD);
    VecGhostUpdateEnd  (x_pc,INSERT_VALUES,SCATTER_FORWARD);
    log.LogEvent(tag_pc_slv1,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_gradP_pc,ChiLog::EventType::EVENT_BEGIN);
    VecSet(x_gradp,0.0);
//    int iref=0;
//    double pref=0.0;
//    VecGetValues(x_pc,1,&iref,&pref);
//    VecShift(x_pc,-pref);

    ComputeGradP_GG(x_gradp, x_pc);
    log.LogEvent(tag_gradP_pc,ChiLog::EventType::EVENT_END);

    log.LogEvent(tag_corr,ChiLog::EventType::EVENT_BEGIN);
    ComputeCorrections();
    log.LogEvent(tag_corr,ChiLog::EventType::EVENT_END);

    //================================= Logs
    {
      double max_v = 0.0; VecMax(x_u[U_X],NULL,&max_v);
      double max_p = 0.0; VecMax(x_p,NULL,&max_p);
      double min_p = 0.0; VecMin(x_p,NULL,&min_p);
      std::vector<double> res_mom_norm(num_dimensions,0.0);

      std::vector<Vec> res_mom(num_dimensions);
      for (int dim : dimensions)
      {
        KSPBuildResidual(lin_solver_u[dim].ksp,NULL,NULL,&res_mom[dim]);
        VecNorm(res_mom[dim],NORM_2,&res_mom_norm[dim]);
        VecDestroy(&res_mom[dim]);
      }


      char buf[200];
      if (num_dimensions == 2)
        sprintf(buf,"Iteration %4d max_v=%.7f max_p=%+.6f "
                    "min_p=%+.6f "
                    "Residuam Momentum-X=%.6e "
                    "Residuam Momentum-Y=%.6e",
                    i, max_v, max_p, min_p,
                    res_mom_norm[U_X],
                    res_mom_norm[U_Y]);
      if (num_dimensions == 3)
        sprintf(buf,"Iteration %4d max_v=%.7f max_p=%+.6f "
                    "min_p=%+.6f "
                    "Residuam Momentum-X=%.6e "
                    "Residuam Momentum-Y=%.6e "
                    "Residuam Momentum-Z=%.6e",
                i, max_v, max_p, min_p,
                res_mom_norm[U_X],
                res_mom_norm[U_Y],
                res_mom_norm[U_Z]);
      log.Log(LOG_0) << chi_program_timer.GetTimeString() << " " << buf;
    }

  }

  //================================= Print time summaries
  log.Log(LOG_0) << "GradP_GG           " << log.ProcessEvent(tag_gradP_gg,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "GradP_U            " << log.ProcessEvent(tag_gradU   ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "Momentum Assembly  " << log.ProcessEvent(tag_mom_assy,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "Momentum solve - 0 " << log.ProcessEvent(tag_mom_slv0,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "Momentum solve - 1 " << log.ProcessEvent(tag_mom_slv1,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "Mass flux          " << log.ProcessEvent(tag_comp_mf ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "PCorr Assembly     " << log.ProcessEvent(tag_pc_assy ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "PCorr solve - 0    " << log.ProcessEvent(tag_pc_slv0 ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "PCorr solve - 1    " << log.ProcessEvent(tag_pc_slv1 ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "GradP_PC           " << log.ProcessEvent(tag_gradP_pc,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;
  log.Log(LOG_0) << "Corrections        " << log.ProcessEvent(tag_corr    ,ChiLog::EventOperation::TOTAL_DURATION)/1.0e6;

  log.Log(LOG_0) << "Exporting visualization data";
  //================================= Copy petsc vector to local
  std::vector<std::vector<double>> data_xyz(num_dimensions);
  std::vector<double> data;
  chi_math::PETScUtils::CopyVecToSTLvector(x_u[U_X],data_xyz[U_X],ndof_local_u);
  chi_math::PETScUtils::CopyVecToSTLvector(x_u[U_Y],data_xyz[U_Y],ndof_local_u);

  data.reserve(num_dimensions*ndof_local_u);
  for (int j=0; j<ndof_local_u; ++j)
    for (int i=0; i<num_dimensions; ++i)
      data.push_back(data_xyz[i][j]);

  std::vector<double> data_p;
  chi_math::PETScUtils::CopyVecToSTLvector(x_p,data_p,ndof_local_p);

  log.Log(LOG_0) << "Exporting visualization data2";
  //================================= Attach to field function
  auto ffu = new chi_physics::FieldFunction(
    std::string("U"),              //Text name
    0,                                        //Number id
    chi_physics::FieldFunctionType::FV,       //Field function sd-method
    grid,                                     //Grid
    &fv_sdm,                                  //Spatial DM
    2,                           //Num components per set
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
    &fv_sdm,                                      //Spatial DM
    1,                                        //Num components per set
    1,                                        //Num sets
    0,0,                                      //Ref component and set
    nullptr,                                  //DOF block address
    &data_p);                                   //Data

  ffP->ExportToVTKFV(std::string("ZP"),std::string("PressureCorrection"));

  log.Log(LOG_0) << "Done exporting visualization data";

  for (int i=0; i<num_dimensions; ++i)
    MatDestroy(&A_u[i]);
  MatDestroy(&A_pc);

  for (int i=0; i<num_dimensions; ++i)
  {
    VecDestroy(&x_u   [i]);
    VecDestroy(&x_uold[i]);
    VecDestroy(&x_umim[i]);
    VecDestroy(&b_u   [i]);
  }

  VecDestroy(&x_p);

  VecDestroy(&x_pc);
  VecDestroy(&b_pc);

  for (int i=0; i<num_dimensions; ++i)
    KSPDestroy(&lin_solver_u[i].ksp);
  KSPDestroy(&lin_solver_p.ksp);

//  PCDestroy(&lin_solver_u.pc);
//  PCDestroy(&lin_solver_p.pc);


  log.Log(LOG_0) << "Done executing Incompressible Navier Stokes Solver";
}