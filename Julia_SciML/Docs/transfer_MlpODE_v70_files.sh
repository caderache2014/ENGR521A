cd /Users/cmason83/Desktop/UWash_AIML_Eng/ENGR521/ENGR521A

# Copy the new MlpODE_v70 outputs
WS=/Users/cmason83/Desktop/sciml_workspaces/wang_model/wang_model_workspace

cp $WS/mlpODE_checkpoint_v70.bson                   Julia_SciML/
cp $WS/predictions_MlpODE_v70.csv                   Julia_SciML/plots_results/
cp $WS/MlpODE_predictions_hard_shot_v70.csv         Julia_SciML/plots_results/
cp $WS/fullshot_mape_TEST_MlpODE_v70.csv            Julia_SciML/plots_results/
cp $WS/endpoint_mape_TEST_MlpODE_v70.csv            Julia_SciML/plots_results/
cp $WS/loss_history_MlpODE_v70.csv                  Julia_SciML/plots_results/
cp $WS/val_loss_history_MlpODE_v70.csv              Julia_SciML/plots_results/
cp $WS/vdot_diagnostic_summary_MlpODE_v70.csv       Julia_SciML/plots_results/
cp $WS/src/vdot_diagnostic_MlpODE_v70.jl            Julia_SciML/src/

git add -A
git status