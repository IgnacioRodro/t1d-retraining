# t1d-retraining

How often, with which data window and by which mechanism should a personalised
glucose forecasting model be re-adapted to the patient's own data? This project
studies model ageing, retraining policies and adaptation mechanisms on public
type 1 diabetes datasets.

The code cites its design decisions as D<number>, the research questions as
Q1–Q4, the models as M00–M19 and sections of the authors' working log, which is
written in Spanish and is not part of the repository; `docs/decisions.md` gives
all of them in English. The methods are described in the companion articles
(Artificial Intelligence in Medicine and MethodsX).
No patient data are included: see *Data sources and attribution* for where to
request each dataset.

## Setup (one command)

On the machine where the project will run (the compute server), with conda,
mamba or micromamba available (e.g. `module load anaconda` on a cluster):

```bash
cp config/paths.example.yaml config/paths.yaml        # then set the dataset folders
python scripts/00_check_environment.py
```

Script 00 creates the conda environment `t1d` (Python 3.12, conda-forge) if it
does not exist, tries to import every package of `environment/requirements.txt`
inside it, installs whatever is missing or broken (torch as a CUDA build when
an NVIDIA GPU is visible), installs this project, and reports the machine.
Run it on a GPU node so that the right torch build is chosen. Useful options:
`--recreate`, `--no-install`, `--torch cu128`, `--env-prefix /path/to/env`.

You never need to activate the environment: every script detects whether it
runs inside `t1d` and, if not, re-launches itself with the environment's Python
(recorded in `config/environment.local.json`).

## Updating the code

Each release zip contains the whole code. Unzip it over the project folder
(`unzip -o t1d-retraining.zip` from the folder that contains `t1d-retraining/`);
`config/paths.yaml` and `config/environment.local.json` are not in the zip, so
they are kept. Scripts 01 and later compare every Python file with
`src/t1dretrain/code_manifest.json` (written by `tools/build_code_manifest.py`
when a release is packed) and stop with the list of out-of-date files if a copy
was partial. Use `--skip-code-check` only for deliberate local edits. The
release number and the result of the check are saved in every
`run_manifest.json`.

## Run order

Scripts are numbered and must run in order. Each one writes CSV tables, PNG
figures, `summary.md` and `run_manifest.json` to `results/<script_id>/`;
zip that folder to share the results.

```bash
python scripts/00_check_environment.py
python scripts/01_inventory_raw_data.py
python scripts/02_harmonize_datasets.py      # raw datasets -> common schema in <derived_data>/processed/
python scripts/03_data_quality.py            # one clean CGM series per patient, day table, eligibility (<derived_data>/clean/)
python scripts/04_define_splits.py           # cohorts, core groups (config/splits/), day 0 and training anchors
python scripts/05_build_model_inputs.py      # native-grid glucose, IOB, COB, calendar, wearables (<derived_data>/features/)
python scripts/06_patient_characteristics.py # sex or gender and age of every cohort, for table 1 (SAGER; seconds)
python scripts/10_stage0_baselines_noise.py  # stage 0: reference and learned models on the screen group, noise floor
python scripts/11_pretrain_population.py     # tuning and population models on the pool group (<derived_data>/models/)
python scripts/12_stage0_population_transfer.py  # stage 0 again: scratch, population and fine-tuned starting points
python scripts/13_stage0_foundation.py       # stage 0 for zero-shot foundation models (Chronos-2, TimesFM 2.5, TTM)
python scripts/20_stage1_ageing_runs.py      # stage 1: frozen, rolling and growing versions from each anchor
python scripts/21_stage1_ageing_analysis.py  # stage 1 analysis: mixed models, useful life, screening 1 (CPU)
python scripts/30_stage2_policy_runs.py      # stage 2: refit windows and frequencies, successive halving, drift triggers
python scripts/31_stage2_policy_analysis.py  # stage 2 analysis: optimal interval, growth of the gain, later anchors (CPU)
python scripts/40_stage3_mechanism_runs.py   # stage 3: restart, warm start, leaf refit, scratch; ARX forgetting factor
python scripts/41_stage3_mechanism_analysis.py  # stage 3 analysis: efficiency, insulin and carbohydrates, forgetting (CPU)
python scripts/42_stage3b_physiology.py       # stage 3b: insulin action and carbohydrate absorption, for all and per patient
python scripts/43_stage3_context.py         # stage 3: zero-shot context of 1-14 days against refitting; ageing against it
python scripts/50_confirmation_runs.py       # confirmation: what the screening chose, on the hold-out and other datasets
python scripts/51_confirmation_analysis.py   # confirmation analysis: which findings of the screening hold, per cohort
python scripts/60_regularity_indices.py      # Q4: regularity of every confirmation patient's habits at startup
python scripts/61_regularity_analysis.py     # Q4: regularity against ageing and the gain of refitting, with the twins
python scripts/62_sex_differences.py          # women minus men in the outcomes of 61 (SAGER; supplement; CPU)
python scripts/70_low_glucose_analysis.py    # forecast lows: shrinkage, Clarke zone D by range, alarm curves (CPU)
python scripts/71_low_glucose_loss.py        # forecast lows: LightGBM and ARX with a loss that weighs the lows
python scripts/72_change_recovery.py         # recovery after real changes (closed loop, randomisation, sensors)
python scripts/73_wearables.py               # wearables as extra inputs, exploratory (few patients)
python scripts/74_family_comparison.py       # every model family against every other, frozen and refitted (CPU)
python scripts/75_precedent_is_lstm.py       # the closest precedent, IS-LSTM, against the recommended refitting
python scripts/90_paper_figures.py           # the paper's figures from the tables above, at the printed size (CPU)
python scripts/91_paper_tables.py            # the paper's tables 1 to 4 as LaTeX, from the tables above (CPU)
python scripts/92_supplement.py              # the supplement's tables S1-S4 and figure S1, from the tables above (CPU)
python scripts/93_method_validation.py       # validation of the method (MethodsX): designs, ageing, verdicts, pairing, runs
```

### The whole study in one run

`tools/run_all.sh` runs scripts 02 to 43 in order with one release (script 00
is left out, as it can change the environment). A full run first clears the
result caches of scripts 20, 30, 40, 42, 43, 50, 71-73, 75 and 93 (`tools/clear_caches.py`), so
that no result comes from an earlier release; each script writes its console
output to `logs/<script>.txt`, and the chain stops at the first failure.
`bash tools/run_all.sh FROM TO` runs the scripts from `FROM` to `TO`.

A script started from an SSH console stops when the connection drops, so on a
remote server start the chain in a detached `tmux` session, with a single
command (pasting several lines into a new `tmux` session can leave the later
lines in the outer shell):

```bash
cd t1d-retraining              # the project folder
tmux new -d -s t1d 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate t1d && bash tools/run_all.sh'
tail -f logs/run_all.txt      # Ctrl-C stops following the log, not the run
tmux ls                       # the session ends when the chain ends
```

After a failure, `bash tools/run_all.sh 30` resumes from script 30 and reuses
what the run had already computed. A single script can be run the same way
(`... && python -u scripts/40_stage3_mechanism_runs.py > logs/40.txt 2>&1`).

**Confirmation (scripts 50 and 51, D92).** It is a separate, deliberate run
(`bash tools/run_all.sh 50 51`): it uses the confirm group of the core dataset,
the hold-out, so it starts only once the selections of the screening are
final (`confirmation.yaml` of script 31 and `mechanisms.yaml` of script 41).
Nothing is chosen in it. Script 50 runs, on every cohort of
`confirmation.cohorts` (the confirm group, DCLP3, Loop, BrisT1D and T1D-UOM),
never, the refit at each model's optimal interval and every day, and the
mechanism of script 41, with the glucose-only twins; script 51 sets every
finding of the screening against its confirmation. Per cohort, `patients`
draws fewer patients, `test_days` sets the follow-up and `daily` whether the
every-day refit runs. Script 50 caches per cohort, model, arm and patient, so a
run that stops can go on later; `--cohorts replace_bg/core` runs some cohorts
only.

**Q4, regularity (scripts 60 and 61, D94).** `bash tools/run_all.sh 60 61`
after the confirmation. Script 60 measures, on the usable days of each
confirmation patient's startup window (the 14 days before the anchor, known
when a refit policy would be chosen), how regular the meals, boluses and
glucose profile are (timing jitter, day-to-day profile stability, daily
amounts, intermittency; steps and sleep where there are wearables), and the
glycaemic control. Script 61 relates them to what happened to each patient in
the confirmation, for every model and its glucose-only twin: the skill of the
frozen model, how fast it ages, the gain of refitting at the optimal interval
and what that interval loses against every day (H3), adjusted for time in
range, CV and mean glucose and for the cohort. Settings in `regularity` of
`config/project.yaml`; the primary indices are fixed there.

**After the confirmation (scripts 70-75, D95-D98, D103).** `bash tools/run_all.sh 70 75`.
Nothing is chosen in them; 71-73 and 75 run on the confirmation's machinery
(`sim.cohort_runs`: the same jobs, arms and caches as script 50).

- 70, forecast lows: from the Clarke histograms of scripts 50, 12 and 13 and
  the day metrics of 50 (no model runs): errors and Clarke zones by glucose
  range, how far the forecasts are pulled towards the middle in the lows, and
  the alarm curves (sensitivity against precision for every alarm threshold),
  which tell a worse detector from a more conservative alarm.
- 71, a loss that weighs the lows: LightGBM (with its population model trained
  again on the pool) and ARX with training targets below 70 mg/dL weighed 5
  and 20 times, on the confirmation's patients and arms of REPLACE-BG and
  DCLP3 Control-IQ, against script 50 (`low_glucose` in the config).
- 72, recovery after real changes: anchored on the first day after a change
  (DCLP3's switch to closed loop, with DCLP3's SAP arm as the control;
  REPLACE-BG's randomisation; BrisT1D's loop and sensor changes), models
  trained on the days before it, the confirmation's arms after it
  (`recovery`).
- 73, wearables: LightGBM and ARX, full and glucose-only, with and without
  heart rate, steps and sleep at the origin, on the patients that have them
  (`wearables`).
- 74, families against each other: every full model against every other on
  the same patients and days, frozen and refitted as recommended, in stage 3
  and in every cohort of the confirmation (no model runs; `family_comparison`).
- 75, the closest precedent: the incrementally retrained stacked LSTM of Shen
  and Kleinberg (2025), the LSTM continued every 7 days on the last 7 days,
  with and without its t-test that skips a week without change, against the
  study's arms of script 50 on the same patients and days (`precedent`).

**Paper figures (script 90, D104).** `bash tools/run_all.sh 90 90`, or
`python scripts/90_paper_figures.py --only 3 7` for some of them. It reads the
CSV tables that the other scripts wrote (nothing is recomputed) and draws the
figures of the paper at the printed size of the journal (Elsevier: 190 mm full
width, 7 pt text, never below 6 pt), with panel letters and no title (the
conclusion goes in the caption), one column per horizon and one colour and
marker per model family (`plotting.PAPER_FAMILIES`). Each figure is written as
a vector PDF with embedded fonts and a 300 dpi PNG, with the data it shows
(`<figure>_data.csv`). `--results DIR` reads the tables of another results
folder. The list is `paper_figures.figures` in the config; a figure whose
inputs are missing is left out and named in the summary, and one that fails
does not stop the others (the run then ends with an error that names it). The
font, Liberation Sans (the metrics of Arial; SIL Open Font License 1.1), ships
with the project in `src/t1dretrain/fonts/`, so the figures come out the same
on every machine.

**Sex or gender per cohort (script 06, D108).** `bash tools/run_all.sh 06 06`.
The journal asks to report sex and gender (SAGER). Each dataset records it its
own way: REPLACE-BG and DCLP3 a Gender column (M, F), Loop a baseline survey
question on gender (1 male, 2 female, 3 non-binary; `DataGlossary.rtf`),
BrisT1D a free-text answer on gender; T1D-UOM and HUPA-UCM publish none. The
script reads only the patients table of every dataset with its loader (the
harmonised data of script 02 are not touched), normalises the answers to female,
male or other (`data.demographics`) and counts them per cohort of script 04 and
per REPLACE-BG group, with the age. Script 91 takes the share of women into
table 1 (`n.r.` where the dataset records none).

**Paper tables (script 91, D106).** `bash tools/run_all.sh 91 91`. It reads
the tables of scripts 04, 06, 11, 43, 51, 73 and 74 (nothing is recomputed;
the share of the daily refit's gain an interval keeps is the ratio of two
gains that script 51 published) and writes the body of tables 1 (datasets and
cohorts), 2 (models, with the starting points the pool chose in script 11), 3
(the practical guide per model) and 4 (replication per cohort) as LaTeX
(`tabularx`, booktabs), with their CSV and a PNG preview. The captions stay in
the manuscript, which wraps each body with `\input`. The text no result file
holds (therapy, sensor, inputs and role of each cohort; class, inputs, refits
tested and stages of each model; recommended use) is in `paper_tables` of the
config. `\pend` marks a value still to add.

**Supplementary material (script 92, D111).** `bash tools/run_all.sh 92 92`.
From the tables of scripts 51, 61 and 62 it writes the body of tables S1
(the primary regularity indices against the four per-patient outcomes), S2
(patients by sex or gender), S3 (RMSE in mg/dL, frozen and refitted, and the
Clarke zones of the refitted model per confirmation cohort) and S4 (detection
of the forecast lows per cohort), and figure S1 (women minus men) at its printed
size. Nothing is recomputed; the settings are in `supplement` of the config.

**Validation of the method (script 93, D112).** `bash tools/run_all.sh 93 93`
(CPU; about an hour on the server, most of it part 1). The evidence of the
MethodsX article, in five parts (`--parts` runs some of them):
(1) a random split of the samples, a random split of the days and one
chronological split, each fitted on the patient's whole record and scored on
its held-out origins together with the simulator's frozen and refitted models
on the same origins (LightGBM fine-tuned and ARX, screening patients of stage
3), and a few patients run twice to check that a run is deterministic; (2) the
ageing estimator of script 43 on synthetic cohorts with a known loss, with the
day-to-day noise calibrated on the day metrics of scripts 40 and 43, and the
width of its intervals on the real data; (3) the tie rule with the minimum
detectable difference on comparisons of every stage-0 model with itself (other
seeds), with and without known differences added; (4) paired against unpaired
designs, in stage 0 and in the confirmation (also a crossover in time); (5) the
cost of every stage from the run manifests and fit logs and, with `--compare
<other results folder>`, whether two runs wrote the same tables. Part 1 caches
per patient and model; settings in `method_validation` of the config.

**Glucose-only twins (D90).** Every model that reads insulin and
carbohydrates and goes on to stages 1-3 (LightGBM, N-HiTS, TiDE, TSMixerx and
the LSTM fine-tuned, and ARX) has a twin that reads the glucose only
(`*_noexog`), and the twin runs exactly what its full model runs: the same
versions in stage 1, the policies its full model keeps in stage 2, and the
same arms and interval in stage 3. Selections are made on the full models.
`glucose_only_twins` in `config/project.yaml` lists the families with twins
(`enabled: false` leaves every twin out, for quick runs only).

Script 11 trains the glucose-only population models (no IOB, COB or time of
day) of the families in `glucose_only_twins` with the full ones;
`--twins-only` adds them later without tuning again.

Script 11 accepts `--families` to train only some model families (the others
keep their saved models). Script 20 caches each model's results per anchor
under `<derived_data>/runs/20_stage1_ageing_runs/cache/`, so a later run only
computes new or changed models (`--recompute` ignores the cache); script 30
does the same per model, policy and patient
(`runs/30_stage2_policy_runs/cache/`), `--round1-only` stops it after the first
round of successive halving, and `--skip-later-anchors` leaves out the runs
from anchors 42 and 70 and `--skip-twins` the glucose-only twins (quick runs
only). Script 40 caches per model, arm and patient in the same way, and script
42 per patient (`--no-lightgbm` runs ARX only). Script 43 caches per context
length, model and patient; it reads the results of script 40 to compare reading
more context with refitting. Script 42 reads the delivered insulin and logged
carbohydrates of script 02 (`<derived_data>/processed/`) to compute the inputs
again with other curves. Script 13 downloads the foundation-model weights from
the Hugging Face Hub the first time; `--patients 2` is a quick check that they
load (scripts 12 and 20 take `--patients` too).

Script 11 uses only the pool group, which is never evaluated: candidates are
trained on 80 % of the pool and scored on the other 20 % as new patients. The
chosen settings and the population models (one per seed) are saved in
`<derived_data>/models/<core>/population/` with `choices.yaml`; script 12
reads them.

The confirm group of the core dataset is a hold-out: `data.splits.group_patients`
refuses to return it unless the caller is a confirmation script (50-51).

**Ties (D61, D75).** A difference below the DMD (the change that two trainings
of the same model show by chance, script 12) is a tie. Each learned model has
its own DMD, drawn from its own bootstrap stream; a model without seed noise
(AR(3), ARX, the foundation models) takes the smallest DMD of the full learned
models of its stage. Glucose-only twins keep their own DMD but never set that
floor, so the results of the full models are the same with or without twins.

Script 04 draws the groups of the core dataset once and saves them in
`config/splits/<dataset>.yaml` (not in the release zip, so unzipping never
replaces them). Later runs keep the saved groups and report any patient whose
eligibility changed; `--reassign` redraws them, which is only acceptable
before any model has been run.

## Data sources and attribution

The raw datasets are not redistributed; each is requested or downloaded from
its own source:

- REPLACE-BG: https://public.jaeb.org/dataset/546
- DCLP3: https://public.jaeb.org/dataset/573
- Loop observational study: https://public.jaeb.org/dataset/560
- BrisT1D-Open: https://data.bris.ac.uk/data/dataset/33z5jc8fa6tob21ptrugzqog08
- T1D-UOM: https://zenodo.org/records/17361905
- HUPA-UCM (official release only, D16): https://data.mendeley.com/datasets/3hbcscwz44/1

Any publication, presentation or communication that uses them must carry these
texts:

- **Loop** (JAEB; literal, from the dataset's ReadMe): "The source of the data
  is the Loop Study (sponsored by the Jaeb Center for Health Research and funded
  by the Helmsley Charitable Trust), but the analyses, content and conclusions
  presented herein are solely the responsibility of the authors and have not
  been reviewed or approved by the study sponsor."
- **REPLACE-BG** (the JAEB template of the dataset page, filled in): "The
  source of the data is Jaeb Center for Health Research. (n.d.). Replace-BG
  Dataset [Data set]. Retrieved from https://public.jaeb.org/dataset/546. The
  analyses content and conclusions presented herein are solely the
  responsibility of the authors and have not been reviewed or approved by the
  REPLACE-BG Study Group or the T1D Exchange."
- **DCLP3** (likewise): "The source of the data is Jaeb Center for Health
  Research. (2022). DCLP3 Public Dataset (Release 3) [Data set]. Retrieved from
  https://public.jaeb.org/dataset/573. The analyses content and conclusions
  presented herein are solely the responsibility of the authors and have not
  been reviewed or approved by the International Diabetes Closed Loop (iDCL)
  Trial Research Group."
- **BrisT1D-Open, HUPA-UCM and T1D-UOM:** cited as their CC BY 4.0 licences ask.
- **DiaData** (CC BY-NC 4.0) is not read by the code; it must never be used to
  pretrain models evaluated on REPLACE-BG, DCLP3 or HUPA-UCM.

## Metrics

Every evaluation of a model reports RMSE and the share of forecasts in each
Clarke Error Grid zone, and saves what is needed to redraw the grid: a sparse
2-D histogram of (reference, forecast) in 5 mg/dL bins with the zone of each
cell, and a random sample of pairs (`src/t1dretrain/eval/metrics.py`,
`ClarkeAccumulator` and `plot_clarke`).

## Tests

```bash
python -m pytest tests
```

Script 02 accepts `--datasets hupa_ucm t1d_uom` to run a subset and
`--max-chunks 2` for a quick smoke test that reads only the first chunks of
the multi-GB files. The processed tables are one parquet file per patient and
table (`cgm`, `insulin`, `carbs`, `wearables`, `sleep`) plus `patients.parquet`;
the schema is documented in `src/t1dretrain/data/schema.py`.

Each script's docstring lists its inputs, outputs and options, and *Run order*
above gives the order; `docs/decisions.md` explains the decisions they cite.

## Compute policy

Every script detects the machine at start-up (`t1dretrain.compute`): it runs
in parallel on all available cores except two (`compute.reserve_cores` in
`config/project.yaml`), respects SLURM allocations, and uses the GPU (CUDA or
Apple MPS) whenever one is available. Set `T1D_N_WORKERS=1` to debug serially.

Many worker processes share the GPUs: each worker uses one GPU, and at most
`compute.gpu_slots_per_gpu` workers (6) compute on a GPU at the same time;
the environment variable `T1D_GPU_SLOTS` overrides it for one run (for example
`T1D_GPU_SLOTS=4` if a run reports GPU out-of-memory errors). Scripts 11, 12,
13 and 20 end their `summary.md` with a "Fits" table: a run is complete when
`other_failure` is 0.

## Conventions

- Code, comments, figures and file names are in English.
- Settings live in `config/*.yaml` and `environment/environment.json`; scripts take no hard-coded paths.
- Shared logic lives in `src/t1dretrain/`; scripts are thin entry points that start with the environment guard.
- Raw data is never modified; derived data goes to `storage.derived_data` (default `data/`).

## License and citation

The code is released under the MIT License (`LICENSE`); the bundled Liberation
Sans fonts keep their SIL Open Font License 1.1. To cite the software, use
`CITATION.cff` (GitHub shows it as *Cite this repository*); each release is
archived in Zenodo with its own DOI.
