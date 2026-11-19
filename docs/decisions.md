# Design decisions

The code cites its design decisions as D&lt;number&gt;, the research questions and
hypotheses as Q1–Q4 and H1–H4, the models of the catalogue as M00–M19, and
sections of the authors' working log (for example "HANDOFF 7.1"). The log is
written in Spanish, contains machine-specific paths and internal notes, and is
not part of the repository. This document gives what the code refers to, in
English. The methods and results are described in the companion articles
(*Artificial Intelligence in Medicine* and *MethodsX*).

Dates are those of the decision. "Iteration" refers to the working sessions of
the log. A decision that was later replaced keeps its row, with the decision that
superseded it.

## Research questions and hypotheses

| Code | Question |
|---|---|
| Q1 Aging | How much does a personal model degrade with each day without refitting, by family and horizon? How much is due to the recency of the data and how much to their quantity? |
| Q2 Policy | Which combination of refit interval, data window and strategy balances error and cost best? Does a refit triggered by drift beat a fixed calendar with the same number of refits? |
| Q3 Mechanism | Which way of taking in new data works best for the same computation: updating at every sample, re-estimating, retraining, or only reading a longer context? Which recovers sooner after a real change? |
| Q4 Patient | Does the regularity of a patient's habits predict the best refit interval? |

- **H1.** The benefit of personalizing and refitting grows with the horizon (small at 30 min, larger at 60–120 min).
- **H2.** A window shorter than the weekly cycle increases the error because it forgets repeating patterns.
- **H3.** More regular patients tolerate longer refit intervals.
- **H4.** Refitting only the patient-specific parameters of a frozen network performs almost as well as retraining it and costs much less.

## Model catalogue

| Code | Model | How it takes in new data |
|---|---|---|
| M00 | Persistence | Reference; no parameters |
| M01 | Linear extrapolation of the last 30 min | Reference; no parameters |
| M02 | ARX with a forgetting factor | Update at every sample (recursive least squares) |
| M03 | M02 with a change detector | Update at every sample (not run) |
| M04 | AR(3) | Re-estimation |
| M05–M07 | ARIMAX, AutoETS, AutoTheta | Re-estimation (not run) |
| M08 | LightGBM | Continued training (new trees), leaf re-estimation, or from scratch |
| M09 | LSTM (own encoder–decoder, D74) | Fine-tuning of a population network |
| M10 | N-HiTS | Fine-tuning of a population network |
| M11 | TiDE | Fine-tuning of a population network |
| M12 | TSMixerx | Fine-tuning of a population network |
| M13–M15 | xLSTM, SOFTS, RMoK | Planned, not run |
| M16 | PatchTST | Fine-tuning of a population network |
| M17 | Chronos-2 | Zero-shot; context only |
| M18 | TimesFM 2.5 | Zero-shot; context only |
| M19 | TTM r2 | Zero-shot; context only |

## Sections of the working log cited in the code

| Section | Content | Where to read it |
|---|---|---|
| 2 | Status of the project and run order | README, *Run order* |
| 5.2–5.4 | Verified facts of each dataset; common schema; quality rules | `src/t1dretrain/data/schema.py`; D20–D57 below |
| 6.1, 6.3 | Model catalogue; common model interface | Catalogue above; `src/t1dretrain/models/base.py` |
| 7.1 | Common protocol: no leakage (a fit never sees a target of a day it forecasts), same origins for every model, persistence on the same origins | *Method details* of the MethodsX article; D58, D66 |
| 7.2 | Stage 0: references and noise floor | D59–D63, D75 |
| 7.3 | Stage 1: aging | D65, D78, D101 |
| 7.4 | Stage 2: refit policy | D80–D84 |
| 7.5 | Stage 3: mechanism | D86–D89, D99 |
| 7.6 | Confirmation | D91, D92 |
| 7.7 | Q4: regularity | D94 |
| 7.8 | Metrics and statistics | D53, D61, D75 |
| 7.9 | Computation | D19, D73 |
| 7.10 | Analyses after the confirmation | D95–D98, D103 |
| 13 | Log of iterations and releases | Release notes of the repository |
| 14 | Dataset links and attribution texts | README, *Data sources and attribution* |

## Fixed rules

- **R1.** The authors keep a working log, updated at every iteration.
- **R2.** Code, comments, figures and file names are in English.
- **R3.** Every script writes its tables (CSV), figures (PNG), a summary (`summary.md`) and a run manifest (`run_manifest.json`) to `results/<script>/`.
- **R4.** Scripts are numbered and run in order; earlier scripts are updated when something changes.
- **R5.** Every script uses all available cores but two, and the GPU whenever possible (D19).
- **R6.** Every evaluation reports RMSE and the Clarke error grid zones, and saves what is needed to redraw the grid (D53).

## Decisions

| ID | Date | Decision | Rationale | Status |
|---|---|---|---|---|
| D01 | 2026-10-16 | Review of the state of the art in glucose forecasting in type 1 diabetes with machine learning and time series. | Starting point. | Done |
| D02 | 2026-10-25 | Line of research: refitting a personal model over time with the patient's own data. | Novelty. | Current |
| D03 | 2026-10-25 | One comprehensive study; explore in order and decide when the results make an article. | Authors' choice. | Current |
| D04 | 2026-10-25 | Q1–Q3 have priority; Q4 only if the data support it. | Authors' choice. | Current |
| D05 | 2026-10-25 | REPLACE-BG is the main test bed (open loop, 26 weeks, about 226 patients). | Long follow-up without the confounding of closed-loop control. | Current |
| D06 | 2026-10-25 | HUPA-UCM only for short-term confirmation and three long records as case studies, read from the raw data. | Median of 13 days per patient; the preprocessed grid interpolates glucose. | Current |
| D07 | 2026-10-25 | Main conclusions in open loop; Loop, the Control-IQ arm of DCLP3 and BrisT1D are analyzed separately. | In closed loop the basal insulin responds to glucose (reverse causality). | Current |
| D08 | 2026-10-25 | Short list of 20 models grouped by how they adapt (catalogue above). | Each family represents one way of adapting. | Current |
| D09 | 2026-10-26 | Fixed rules R1–R4 (R5 and R6 added later). | Authors' choice. | Fixed |
| D10 | 2026-10-26 | Project `t1d-retraining`, package `t1dretrain`, one environment with Python 3.12. | Simplicity; the verified dependencies are compatible. | Current |
| D11 | 2026-10-26 | The sensor's native grid; targets only on real readings; horizons of 30, 60 and 120 min; 6-h input window; 14 days of training before the anchor. | Avoids leakage through interpolation; consistent with the literature. | Current |
| D12 | 2026-10-26 | REPLACE-BG groups: pool (60), screening (40) and held-out confirmation (the rest), drawn among the eligible patients of the `core` cohort. Stratified by arm and tercile of usable days; of 5000 draws (seed 20261026), the one with the smallest maximum absolute standardized mean difference over 11 baseline variables is kept. Saved once in `config/splits/replace_bg.yaml`; later runs keep it, and only `--reassign` redraws it, before any model is run. | Select and confirm on different patients; comparable groups that cannot change by accident. | Current (final draw: max \|SMD\| 0.18) |
| D13 | 2026-10-26 | DiaData is never used to pretrain models evaluated on REPLACE-BG, DCLP3 or HUPA-UCM. | DiaData contains those datasets. | Current |
| D14 | 2026-10-26 | The loaders are written from an inventory of the raw files (script 01), not from assumptions. | Heterogeneous JAEB formats. | Current |
| D15 | 2026-10-26 | TimeGPT is left out. | Closed model that requires uploading data and cannot be pinned to a version. | Current |
| D16 | 2026-10-26 | HUPA-UCM only from its official Mendeley Data release. | Unofficial copies mix in data with other terms of use. | Current |
| D17 | 2026-10-26 | Forecast origins every 30 min in the screening and every 15 min in the confirmation. | Computing cost. | Current |
| D18 | 2026-10-26 | Paths in `config/paths.yaml` may be absolute or relative to `config/`. | Scripts work from any folder. | Current |
| D19 | 2026-10-26 | Rule R5: workers = available cores − 2 (at least 1), respecting SLURM and CPU affinity; GPU detected automatically (CUDA, then MPS); `T1D_N_WORKERS` overrides; the resources used go into every manifest. | Runs on a shared compute server. | Fixed |
| D20 | 2026-10-26 | DCLP3 has no public carbohydrate data; it is used for confirmation with glucose and insulin. | Verified in the raw tables. | Current |
| D21 | 2026-10-26 | T1D-UOM (glucose, boluses, basal, meals and Garmin data for about 12 weeks) becomes a confirmation dataset. | Inventory of the raw data. | Current |
| D22 | 2026-10-26 | Derived data and the Python environment outside cloud-synchronized folders. | Performance. | Superseded by D26 |
| D23 | 2026-10-26 | DiaData is not harmonized (50 GB, glucose only). | Cost against use; D13. | Current |
| D24 | 2026-10-26 | Day-of-week effects only where dates are real (HUPA-UCM, BrisT1D, T1D-UOM); not in REPLACE-BG, Loop or DCLP3 (relative, shifted or de-identified dates). | Dataset documentation. | Current |
| D25 | 2026-10-26 | Script 00 creates or repairs the conda environment `t1d` (Python 3.12, conda-forge), installing PyTorch first with the CUDA index the driver supports; scripts relaunch themselves inside it (`envguard`). | Controlled environment on the server. | Current |
| D26 | 2026-10-26 | All work runs on the compute server with the same folder structure and `paths.yaml`. | Authors' choice. | Current |
| D27 | 2026-10-26 | HUPA-UCM, hybrid reading: glucose from the raw files at its native rate; insulin and carbohydrates from the preprocessed 5-min grid; Fitbit data and sleep from the raw files. | The preprocessed grid interpolates glucose (D06). | Current |
| D28 | 2026-10-26 | Unit of HUPA-UCM carbohydrate entries inferred per patient: median above 12 means grams, otherwise 10-g portions. | The column mixes units. | Current |
| D29 | 2026-10-26 | T1D-UOM: day-first dates chosen per file; basal records parsed as pump rates or long-acting injections; probable closed loop flagged from the spacing of pump records. | Samples of the raw data. | Current |
| D30 | 2026-10-26 | Script 02 harmonizes without cleaning: every CGM source and the raw history outside the study window are kept; choosing sources and removing duplicates belongs to script 03. | Separate harmonization from cleaning. | Current |
| D31 | 2026-10-26 | Loop local time: the device clock where available; otherwise the nearest observed offset within 14 days; otherwise the standard offset of the roster. Glucose converted from mmol/L with the Tidepool factor (18.01559). | Following daylight saving time and travel. | Current (revised) |
| D32 | 2026-10-26 | DCLP3 pump times and basal rates (each rate lasts at most 12 h); in the closed-loop arm boluses include automatic corrections. | Output of script 02. | Current |
| D33 | 2026-10-26 | JAEB duration units (extended boluses, basal): milliseconds with Tidepool and minutes with Diasend, as documented; magnitude only as a check. | Deciding per upload misread about 500,000 Loop basal rows. | Current (revised) |
| D34 | 2026-10-26 | Patient identifiers prefixed by dataset (`RBG-`, `DCLP3-`, `LOOP-`, `HUPA…P`, `BRIS-`, `UOM-`); REPLACE-BG uses 2000-01-01 as the synthetic enrollment date. | Unique across datasets and valid as file names. | Current |
| D35 | 2026-10-26 | Harmonization in two phases, in parallel: tasks per large file or patient write per-patient pieces; one job per patient then merges them. | Bounded memory with a 20-GB dataset. | Current |
| D36 | 2026-10-26 | Mandatory attribution texts for the JAEB datasets (README). | Dataset terms. | Current |
| D37 | 2026-10-26 | CGM sources: Loop's two upload routes kept apart; BrisT1D device aliases normalized; summaries use the best-covered source of each day. | Output of script 02. | Current |
| D38 | 2026-10-26 | BrisT1D: device periods and loop status per patient from the insulin and carbohydrate rows, with the date of each change. | Device changes during follow-up. | Current |
| D39 | 2026-10-26 | Wearable clocks checked against sleep; the heart rate of T1D-UOM is not used. | It shows no nocturnal dip at any offset. | Resolved |
| D40 | 2026-10-27 | Code check at start-up: `code_manifest.json` holds the hash of every `.py` file; scripts stop if the code differs (`--skip-code-check` for deliberate changes). | A partial update once mixed old and new code. | Current |
| D41 | 2026-10-27 | Loop patients without any device-clock observation keep the roster's standard offset; analyses that use time of day carry a sensitivity check without them. | Daylight saving time cannot be placed with shifted dates. | Current |
| D42 | 2026-10-27 | Every number-to-time conversion discards and counts out-of-range values; warnings are recorded with their call chain. | Robust reading. | Current |
| D43 | 2026-10-27 | Clean CGM series (script 03): automatic readings only; each source cleaned separately; colliding readings resolved by local support; overlapping clock segments split into two tracks and the earlier one kept, with `clock_break`; one sensor per day; no reading modified or interpolated. | Duplicates and overlapping sensors in the raw data. | Current (revised twice) |
| D44 | 2026-10-27 | Reading flags without deleting: `at_limit` (sensor limits) and `spike` (implausible up-and-down jumps). Spikes are neither inputs nor targets; readings at the limit are kept, flagged. | Readings at the sensor limits. | Current |
| D45 | 2026-10-27 | Valid day: the chosen sensor delivers at least 70% of its expected readings; sensitivity from 50% to 90% reported. | Sensitivity of script 03. | Current |
| D46 | 2026-10-27 | Complete insulin in a day: basal covered and at least one bolus. | Needed for insulin on board. | Current |
| D47 | 2026-10-27 | Loop mode and period of every day, per dataset (open or closed; before or after randomization). | D07. | Current |
| D48 | 2026-10-27 | Eligibility with 42 valid days, in four variants (glucose, open loop, insulin, both). | D05, D07. | Current |
| D49 | 2026-10-27 | Loop device times more than 14 h from UTC are ignored. | Device clocks never set. | Current |
| D50 | 2026-10-27 | Cohorts (script 04): a usable day is valid in the cohort's loop mode, with complete insulin when its models need it; a cohort holds the patients with at least 42 usable days. Cohorts per dataset in `config/splits/<dataset>.yaml`. | One criterion across datasets; open and closed loop apart. | Current |
| D51 | 2026-10-27 | Day 0 and anchors: day 0 is the first day followed by at least 10 usable days of 14; an anchor 14, 42, 70 or 98 days later is available when its 14-day window has at least 10 usable days and the 84 days after it at least 28. | Comparable aging curves from every anchor. | Current |
| D52 | 2026-10-27 | Main period of REPLACE-BG: after randomization, with the full follow-up as sensitivity analysis. | Randomization would otherwise fall among the anchors. | Current |
| D53 | 2026-10-27 | Metrics (R6): RMSE, MAE, MARD, bias and Clarke zones A–E; a sparse 2-D histogram in 5-mg/dL cells with the zone in the key, and a random sample of up to 20,000 pairs that can be merged across processes. | Rule R6. | Current |
| D54 | 2026-10-27 | Insulin on board and insulin activity from the exponential curve of Loop and OpenAPS (360 min, peak at 75 min); carbohydrates on board and absorption rate with linear absorption over 180 min. | Standard starting values (varied in stage 3b, D89). | Current |
| D55 | 2026-10-27 | Native grid of each patient (script 05): each reading goes to its nearest slot; a new segment at every clock break; inputs interpolated across gaps of up to 15 min, never across segments; targets only on real, non-spike readings; forecast origins only on slots with a real reading. | D11. | Current |
| D56 | 2026-10-27 | Causal inputs: every input at a slot uses only records made up to that time; wearables count records finished in the last 15 or 60 min; flags of known insulin and carbohydrates. | No leakage. | Current |
| D57 | 2026-10-27 | Unit of REPLACE-BG carbohydrates decided record by record from the pump's bolus calculator, else by the patient's majority, else by the median per meal. | Meals of 0–6 "g" with boluses of 10 U. | Current (verified) |
| D58 | 2026-10-27 | Forecasting samples (`sim/samples.py`), the same for every model: origin on a real reading in an allowed day and on the origin stride; the 6-h window in the origin's segment, with at least 80% of its slots and the last 30 min complete; a target is the valid reading at origin + horizon in the same segment, else that horizon is skipped. | Compare models on the same origins. | Current |
| D59 | 2026-10-27 | Model interface, version 1: `fit(TrainData)` and `predict(Samples)`; forecasts clipped to 20–600 mg/dL. References: persistence, linear extrapolation, AR(3). Stage-0 protocol: frozen models fitted on 14 days and tested on all later days; skill = 1 − RMSE/RMSE of persistence per patient, with 2000 bootstrap resamples. | Common base for every stage. | Current |
| D60 | 2026-10-27 | Learned models of stage 0: direct LightGBM (one booster per horizon predicting the change from the origin) and N-HiTS with an own training loop; the last two training days for early stopping. | Same samples for every model (D58). | Current |
| D61 | 2026-10-27 | Noise floor: each learned model is trained with seeds 1, 2 and 3; the minimum detectable difference (MDD; DMD in the code) of a model and horizon is the 95th percentile of the absolute bootstrap means of the per-patient seed differences; a smaller difference is a tie. | Seed and patient-sampling variation. | Current; per pair since D75 |
| D62 | 2026-10-27 | Starting points: from scratch, population (pretrained on the pool) and fine-tuned (population fine-tuned on the patient's 14 days). Settings chosen on the pool with a fixed 20% of its patients for validation; the chosen setting is retrained on the whole pool once per seed. A fine-tuning that worsens validation is undone. | Personal models from scratch did not beat AR(3); the pool is never evaluated. | Current |
| D63 | 2026-10-27 | Comparison of starting points (script 12): tie below the MDD, better or worse when the interval excludes zero, otherwise uncertain. A family continues fine-tuned if it beats its version from scratch. | Same tie rule as the rest of the study. | Applied |
| D64 | 2026-10-27 | N-HiTS receives the glucose level as an input; fine-tuning may train the whole network or its output layers only. | Without the level it cannot learn the return to the mean. | Current |
| D65 | 2026-10-27 | Stage 1 (script 20): from every anchor, 84 days with three versions: frozen; refitted on a rolling 14-day window; refitted on the growing window. Every refit starts from the starting point. One seed; persistence on the same origins. | Separates recency from quantity of data. | Current |
| D66 | 2026-10-27 | No-leakage rule in the code: training samples whose targets reach the first forecast day are dropped (`known_before`). | No value of a test day is a training target. | Current |
| D67 | 2026-10-27 | MDD of stage 1 from the fine-tuned models only. | A model that did not continue inflated it. | Superseded by D75 |
| D68 | 2026-10-27 | More networks with the same adapter: TiDE, TSMixerx, LSTM and PatchTST; two population sizes each; fine-tuning of everything or the output layers. | Broader set of global networks. | Current; TiDE and LSTM fixed in D74 |
| D69 | 2026-10-27 | ARX with a forgetting factor: one linear model per horizon (glucose level, recent changes, insulin and carbohydrate inputs), weighted least squares with a small penalty. | A cheap and explainable update at every sample. | Current; λ = 1 since D87 |
| D70 | 2026-10-27 | Zero-shot foundation models (script 13): Chronos-2 with and without covariates, TimesFM 2.5 and TTM r2 with glucose only, 72-h context. | The newest models of the list; prediction cost only. | Current |
| D71 | 2026-10-27 | Incremental computation: per-family reruns of script 11 and result caches per model and anchor keyed by settings, starting point and `CACHE_VERSION`. | Adding models without recomputing. | Current |
| D72 | 2026-10-28 | After a CUDA failure, the fit restarts on the CPU from its starting point and the process stays on the CPU; results with non-data failures are not cached. | A broken CUDA context made later fits fail. | Current |
| D73 | 2026-10-28 | GPU sharing: each worker takes the least-used GPU; at most `compute.gpu_slots_per_gpu` processes compute at once per GPU (file lock); smaller batches and then the CPU when memory runs out. | Too many processes exhausted GPU memory. | Current |
| D74 | 2026-10-28 | Two neuralforecast networks fixed: the final layer normalization of TiDE's decoder is disabled; the LSTM is replaced by an own encoder–decoder, because the library's predicted early steps from stale hidden states. Saved networks carry a version. | Implausible results of TiDE and LSTM. | Current |
| D75 | 2026-10-28 | MDD per pair: each learned model has its own MDD; models without seed noise take the smallest MDD of the learned models; a comparison uses the larger MDD of its two members. | One model's noise should not widen the ties of the others. | Current; refined by D93 |
| D76 | 2026-10-28 | Glucose-only variants (`noexog`) of LightGBM, LSTM, TSMixerx and ARX, with the settings of their full models. | What insulin and carbohydrates add, without a new search. | Extended by D90 |
| D77 | 2026-10-28 | TTM loaded from the named checkpoint `512-96-r2`. | The automatic choice picked a checkpoint for daily data. | Current |
| D78 | 2026-10-28 | Stage-1 analysis (script 21): daily skill on days with at least 12 origins, clipped to [−1, 1]; mixed models of the frozen version per model and horizon; useful life = 7 × MDD / −slope; screening of models for stage 2 written to `shortlist.yaml`. | Rules written before seeing the results. | Current |
| D79 | 2026-10-29 | Stage-2 models: the five of the stage-1 screening plus fine-tuned TSMixerx. | The network that ages least and is best at 120 min. | Current |
| D80 | 2026-10-29 | Stage-2 design (script 30): anchor at day 14 after randomization, 84 test days, 14-day base window; 21 policies per model (never, plus growing, rolling 7/14/28 and exponential windows × every 1, 3, 7, 14 days); successive halving (all policies on 10 patients, the best third and the references on all); cost in compute seconds per test day, each policy paying every fit it uses. | Stage 1 showed that the window matters more than the frequency. | Current |
| D81 | 2026-10-29 | Drift triggers: ADWIN and Page-Hinkley (river) on the causal series of the model's absolute error minus persistence's, scaled by 50 mg/dL; refit the next day; each trigger compared with an even calendar with as many refits. | A trigger is only useful if it beats refitting as often without looking at the errors. | Current |
| D82 | 2026-10-29 | ARX also with its own mechanism in stage 2: an update at every sample by recursive least squares. | The definition of M02. | Current |
| D83 | 2026-10-29 | Stage 2 from later anchors (days 42 and 70) to test whether the gain of refitting depends on how much the data have grown since the first fit rather than on the anchor. | If the gain comes from accumulating data, it should be smaller with more history. | Current |
| D84 | 2026-10-29 | Stage-2 analysis (script 31): the optimal interval is the longest that beats never by more than the MDD and loses to daily refitting by less; the model's interval is the shortest over its horizons. Policies for the confirmation in `confirmation.yaml`. | Rule written before the results. | Current |
| D85 | 2026-10-29 | Glucose-only variants in stages 2 and 3. | Whether refitting gains more when the model reads insulin and carbohydrates. | Superseded by D90 |
| D86 | 2026-10-29 | Stage-3 design (scripts 40 and 41): each model refitted at its optimal interval on the growing window; arms: never, restart from the population model, warm start, leaf re-estimation (LightGBM), from scratch, update at every sample with several forgetting factors (ARX); shared anchor fit; cost and efficiency per arm. | How to take in the new data at the lowest cost. | Current |
| D87 | 2026-10-30 | ARX without forgetting in its batch fits (λ = 1): the window of each policy sets its memory. | λ = 1 beat 0.999 in 36–37 of 38 screening patients. | Current |
| D88 | 2026-10-30 | LightGBM leaf re-estimation with decay 0.9, 0.5 and 0. | Whether the mechanism fails or is only too cautious. | Resolved: no decay helps |
| D89 | 2026-10-30 | Stage 3b (script 42): insulin action of 240–480 min and absorption of 120–240 min (15 combinations), fixed or chosen per patient from past days only. | D54 left these curves as starting values. | Resolved: no effect; D54 kept |
| D90 | 2026-10-30 | Inputs as a factor of the whole study: every model that reads insulin and carbohydrates has a glucose-only twin (`_noexog`) that follows exactly the same path; selections are made with the full models and the twins inherit them. The whole study rerun with one release and empty caches. | Coherence of the whole work. | Current |
| D91 | 2026-10-31 | Mechanism of each model for the confirmation (`mechanisms.yaml`): among the arms that beat the restart at one horizon or more and lose at none, the one with the largest gain over never; otherwise the restart. | One mechanism per model, chosen in the screening. | Current |
| D92 | 2026-10-31 | Confirmation (scripts 50 and 51): nothing is chosen. Cohorts of REPLACE-BG (held-out group), DCLP3, Loop (120 patients, 252 test days), BrisT1D and T1D-UOM; origins every 15 min; never, the optimal interval, daily refitting, ARX at every sample and the selected mechanism. A finding replicates if what was better is better again and what was a tie is not worse. | Pre-specified confirmation. | Current |
| D93 | 2026-10-31 | The glucose-only twins do not set the MDD floor; each model draws its MDD from its own random stream. | A twin lowered the floor and changed other models' ties. | Current |
| D94 | 2026-11-02 | Q4 (scripts 60 and 61): regularity indices of meals, boluses, glucose profiles and wearables over the 14 days before the anchor, against per-patient outcomes in the confirmation patients; least squares with HC3 errors and Holm's correction; three primary indices. | H3, tested on the confirmation patients. | Current |
| D95 | 2026-11-02 | Low glucose (script 70): error and Clarke zones by glucose range, calibration and alarm curves from the saved histograms; per-patient detection. | Learned models raise fewer and more precise alarms. | Current |
| D96 | 2026-11-02 | Loss weighting low targets 5 or 20 times (script 71), as a sensitivity analysis. | Whether penalizing missed lows helps. | Current |
| D97 | 2026-11-02 | Recovery after real changes (script 72): DCLP3 switch to closed loop with the open-loop arm as control, REPLACE-BG randomization, BrisT1D changes of loop and sensor. | Q3, second part. | Current |
| D98 | 2026-11-02 | Wearables, exploratory (script 73): LightGBM and ARX with and without heart rate, steps, energy and sleep. | Descriptive, patient by patient. | Current |
| D99 | 2026-11-02 | Context-only mechanism (script 43): Chronos-2 and TimesFM 2.5 with 1, 3, 7 and 14 days of context, on the stage-3 protocol, against the learned models. | Q3 names "only reading context". | Current |
| D100 | 2026-11-02 | The Surveillance Error Grid is not used. | It needs the official risk table; Clarke and the low-glucose analyses cover clinical risk. | Current |
| D101 | 2026-11-02 | Aging against a model that cannot age (script 43): per patient, the weekly slope of the daily skill minus that of Chronos-2 with a fixed 14-day context; TimesFM 2.5 as negative control. | Weekly skill of frozen and foundation models rises and falls together. | Current |
| D102 | 2026-11-03 | Journal: *Artificial Intelligence in Medicine*; LaTeX with the Elsevier template; an IS-LSTM-like arm to compare with the closest precedent; the evaluation method in a separate *MethodsX* article. | Authors' choice. | Current |
| D103 | 2026-11-03 | Two analyses after the confirmation: every family against every other (script 74), and the IS-LSTM policy of Shen and Kleinberg (2025) as two arms of the same simulator (script 75). | Comparison with other proposals. | Current |
| D104 | 2026-11-03 | Article figures (script 90) drawn from the CSV tables of the other scripts at Elsevier's final size, with one color and marker per family. | Numbers always from the results. | Current |
| D105 | 2026-11-18 | Article length and supplement; MethodsX co-submitted with its Word template; each article stands alone. | Journal guides. | Current |
| D106 | 2026-11-18 | Article tables (script 91) written from the CSV tables of the other scripts. | As D104. | Current |
| D107 | 2026-11-18 | Code in a public GitHub repository (MIT License), archived in Zenodo with a DOI and cited in both articles; no data, results or working log. | Open code. | Current |
| D108 | 2026-11-18 | Sex or gender (SAGER): share of women per cohort among patients with the datum, as each dataset records it (script 06). | Journal policy. | Current |
| D109 | 2026-11-18 | Women against men in the per-patient outcomes of the confirmation (script 62), reported as differences with intervals. | SAGER. | Done: no differences |
| D110 | 2026-11-18 | Table 2 of the article (models) generated by script 91. | As D106. | Current |
| D111 | 2026-11-18 | Separate, contained supplement (script 92). | Journal guides. | Current |
| D112 | 2026-11-19 | Validation of the method for MethodsX (script 93): common designs against the simulator, the aging estimator on calibrated synthetic cohorts, the tie rule on null comparisons and with added differences, paired against unpaired designs, reproducibility and cost per stage. | MethodsX asks for validation of the method. | Current |
