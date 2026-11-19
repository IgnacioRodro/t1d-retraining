#!/usr/bin/env bash
# Runs the study in order with one code release (HANDOFF section 2, D90): the data (scripts 02-05) and the patients'
# sex or gender and age per cohort (06, D108), stage 0 (10-13),
# stage 1 (20-21), stage 2 (30-31) and stage 3 (40-43, with the context sweep of the foundation models, D99); with a
# later end, the confirmation (50-51, D92), Q4, the regularity of the patients' habits (60-61, D94), women against men
# in the same outcomes (62, D109), and the analyses
# on the confirmation's machinery: the forecast lows (70-71, D95, D96), the recovery after real changes (72, D97), the
# wearables (73, D98), the families against each other (74) and the closest precedent, IS-LSTM (75; D103); last, the
# paper's figures and tables and the supplement's from the tables of all of them (90, D104; 91, D106; 92, D111), and
# the validation of the method for the MethodsX article (93, D112). The confirmation is a separate, deliberate run: it
# uses
# the hold-out, so it starts only once the selections of scripts 31 and 41 are final; 60 to 75 read its plan or
# results. A full run (from script 02) first clears the result caches
# of scripts 20, 30, 40, 42, 43, 50, 71-73, 75 and 93 (tools/clear_caches.py), so that no result comes from an earlier
# release; a run resumed from a later script reuses what this run already computed. Each script writes its console
# output to logs/<script>.txt and the chain stops at the first script that fails (logs/run_all.txt has the times).
# Script 00 is left out: it can change the environment.
#
# Usage: bash tools/run_all.sh [FROM [TO]]   (script numbers; default 02 to 43)
# On a remote server, start it detached so that a dropped connection does not stop it:
#     cd t1d-retraining                                    # the project folder
#     tmux new -d -s t1d 'bash tools/run_all.sh'          # the screening, from script 02 to 43
#     tmux new -d -s t1d 'bash tools/run_all.sh 20'       # from script 20 to 43 (e.g. after a failure)
#     tmux new -d -s t1d 'bash tools/run_all.sh 50 51'    # the confirmation
#     tmux new -d -s t1d 'bash tools/run_all.sh 60 61'    # Q4, after the confirmation
#     tmux new -d -s t1d 'bash tools/run_all.sh 70 75'    # lows, recovery, wearables, families, IS-LSTM
#     bash tools/run_all.sh 06 06                         # sex or gender per cohort, for table 1 (seconds)
#     bash tools/run_all.sh 62 62                         # women against men (after 06 and 61; minutes, CPU)
#     bash tools/run_all.sh 90 92                         # the paper's figures and tables, and the supplement's (CPU)
#     tmux new -d -s t1d 'bash tools/run_all.sh 93 93'    # validation of the method, MethodsX (CPU, about an hour)
#     tail -f logs/run_all.txt                            # progress; tmux ls lists the session while it runs
# The scripts run inside the project environment on their own (t1dretrain.envguard).

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
FROM=${1:-02}
TO=${2:-43}
PYTHON=${PYTHON:-python3}
STEPS=(
  "02_harmonize_datasets"
  "03_data_quality"
  "04_define_splits"
  "05_build_model_inputs"
  "06_patient_characteristics"
  "10_stage0_baselines_noise"
  "11_pretrain_population"
  "12_stage0_population_transfer"
  "13_stage0_foundation"
  "20_stage1_ageing_runs"
  "21_stage1_ageing_analysis"
  "30_stage2_policy_runs"
  "31_stage2_policy_analysis"
  "40_stage3_mechanism_runs"
  "41_stage3_mechanism_analysis"
  "42_stage3b_physiology"
  "43_stage3_context"
  "50_confirmation_runs"
  "51_confirmation_analysis"
  "60_regularity_indices"
  "61_regularity_analysis"
  "62_sex_differences"
  "70_low_glucose_analysis"
  "71_low_glucose_loss"
  "72_change_recovery"
  "73_wearables"
  "74_family_comparison"
  "75_precedent_is_lstm"
  "90_paper_figures"
  "91_paper_tables"
  "92_supplement"
  "93_method_validation"
)
if [[ "$FROM" == "02" ]]; then
  "$PYTHON" tools/clear_caches.py | tee -a logs/run_all.txt || exit 1
fi
for step in "${STEPS[@]}"; do
  read -r script options <<< "$step"
  if [[ "${script:0:2}" < "$FROM" || "${script:0:2}" > "$TO" ]]; then
    continue
  fi
  echo "$(date '+%F %T') start $script $options" | tee -a logs/run_all.txt
  # shellcheck disable=SC2086
  if ! "$PYTHON" -u "scripts/$script.py" $options > "logs/$script.txt" 2>&1; then
    echo "$(date '+%F %T') FAILED $script: see logs/$script.txt; continue with: bash tools/run_all.sh ${script:0:2} $TO" \
      | tee -a logs/run_all.txt
    exit 1
  fi
  echo "$(date '+%F %T') done $script" | tee -a logs/run_all.txt
done
echo "$(date '+%F %T') all done: upload results/" | tee -a logs/run_all.txt
