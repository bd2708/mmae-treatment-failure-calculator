# Data Format Description

Patient-level data from the MESH Registry cannot be shared due to privacy restrictions. This document describes the required input format for researchers who wish to apply this model to their own institutional data.

## Target Variable

- **treatment_failure** - Binary (0/1). Treatment failure within 90 days, defined as surgical intervention, repeat MMAE, or SDH-related death.

## Input Features (64 total)

### Base Clinical Features (51 features)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Continuous | Patient age in years (18-110) |
| `sex` | Binary (0/1) | Biological sex (0=Female, 1=Male) |
| `ethnicity` | Categorical (integer) | Ethnicity category |
| `admission_gcs` | Ordinal (3-15) | Glasgow Coma Scale score at admission |
| `admission_premorbid__mrs` | Ordinal (0-5) | Modified Rankin Scale before SDH onset |
| `altered_mental_status` | Binary (0/1) | Altered mental status at presentation |
| `confusion` | Binary (0/1) | Confusion at presentation |
| `headache` | Binary (0/1) | Headache at presentation |
| `seizure` | Binary (0/1) | Seizure at presentation |
| `focal_weakness` | Binary (0/1) | Focal weakness at presentation |
| `facial_droop` | Binary (0/1) | Facial droop at presentation |
| `cranial_nerve_palsy` | Binary (0/1) | Cranial nerve palsy at presentation |
| `plt_count` | Continuous | Platelet count at admission (x10^9/L) |
| `hemoglobin__at_admission` | Continuous | Hemoglobin at admission (g/dL) |
| `wbc__at_admission` | Continuous | White blood cell count at admission (x10^9/L) |
| `aptt__at_admission` | Continuous | Activated partial thromboplastin time at admission (seconds) |
| `sdh_thickness__receiving_mmae` | Continuous | Maximum SDH thickness on imaging (mm) |
| `midline_shift__if_no_put_0` | Continuous | Midline shift on imaging (mm, 0 if none) |
| `sdh_laterality` | Categorical (integer) | SDH laterality (1=Unilateral, 3=Bilateral) |
| `brain_atrophy` | Binary (0/1) | Brain atrophy present on imaging |
| `membranes_or_septations` | Binary (0/1) | Membranes or septations present on imaging |
| `sdh_imaging_characteristics` | Categorical (1-4) | CT density (1=Hypodense, 2=Isodense, 3=Hyperdense, 4=Mixed) |
| `cause_of_sdh` | Categorical (integer) | Cause of subdural hematoma |
| `cardiac_disease` | Binary (0/1) | History of cardiac disease |
| `liver_disease` | Binary (0/1) | History of liver disease |
| `history_of_gastrointestinal_bleeding` | Binary (0/1) | History of gastrointestinal bleeding |
| `history_of_vte` | Binary (0/1) | History of venous thromboembolism |
| `prior_diagnosis_of_icad` | Binary (0/1) | Prior diagnosis of intracranial disease |
| `hx_of_cranial_trauma` | Binary (0/1) | History of cranial trauma |
| `hx_ventriculoperitoneal_shunt` | Binary (0/1) | History of ventriculoperitoneal shunt |
| `sickle_cell_disease` | Binary (0/1) | Sickle cell disease |
| `hivhcv` | Binary (0/1) | HIV or hepatitis C |
| `anticoagulants` | Binary (0/1) | On anticoagulation therapy |
| `antiplatelets` | Binary (0/1) | On antiplatelet therapy |
| `antithrombotics` | Binary (0/1) | On any antithrombotic therapy (anticoagulant or antiplatelet) |
| `anticoagulation_drug` | Binary (0/1) | Specific anticoagulation drug use |
| `alcohol` | Binary (0/1) | Alcohol use |
| `smoking` | Binary (0/1) | Smoking history |
| `charlson_score` | Ordinal (integer) | Charlson Comorbidity Index score |
| `charlson_age_adjusted` | Ordinal (integer) | Age-adjusted Charlson Comorbidity Index |
| `anticoag_risk_score` | Ordinal (integer) | Anticoagulation risk composite score |
| `bleeding_risk_score` | Ordinal (integer) | Bleeding risk composite score |
| `is_chronic_sdh` | Binary (0/1) | Chronic SDH (hypodense or isodense on CT) |
| `is_acute_sdh` | Binary (0/1) | Acute SDH (hyperdense on CT) |
| `is_mixed_sdh` | Binary (0/1) | Mixed-density SDH |
| `is_bilateral` | Binary (0/1) | Bilateral SDH |
| `complex_sdh` | Binary (0/1) | Complex SDH (membranes/septations or bilateral) |
| `significant_mass_effect` | Binary (0/1) | Significant mass effect on imaging |
| `severe_thrombocytopenia` | Binary (0/1) | Platelet count < 50 x10^9/L |
| `moderate_thrombocytopenia` | Binary (0/1) | Platelet count 50-99 x10^9/L |
| `severe_anemia` | Binary (0/1) | Hemoglobin < 8 g/dL |
| `moderate_anemia` | Binary (0/1) | Hemoglobin 8-9.9 g/dL |
| `prolonged_aptt` | Binary (0/1) | aPTT > 40 seconds |
| `coagulopathy_score` | Binary (0/1) | Composite coagulopathy indicator |
| `thrombocytopenia` | Binary (0/1) | Platelet count < 150 x10^9/L |

### Engineered Features (13 features)

These features are derived from the base features during preprocessing:

| Feature | Type | Derivation |
|---------|------|------------|
| `age_plt_interaction` | Continuous | (age x plt_count) / 10000 |
| `sdh_shift_product` | Continuous | (sdh_thickness x midline_shift) / 100 |
| `sdh_shift_ratio` | Continuous | midline_shift / (sdh_thickness + 1) |
| `plt_hgb_ratio` | Continuous | plt_count / (hemoglobin + 0.1) |
| `elderly` | Binary (0/1) | 1 if age > 80, else 0 |
| `age_squared` | Continuous | (age / 100)^2 |
| `age_charlson_product` | Continuous | (age x charlson_score) / 100 |
| `composite_bleeding_risk` | Ordinal (integer) | Sum of: 2*(plt<100) + (plt<150) + anticoagulants + (bleeding_risk>2) |
| `neurological_severity` | Ordinal (integer) | Sum of: (GCS<15) + 2*(GCS<13) + 2*mass_effect + altered_mental_status |

## Notes

- The ONNX model includes preprocessing (median imputation + standard scaling). Pass raw, unscaled values.
- Missing values should be left as NaN; the embedded imputer will handle them using training-set medians.
- Feature order must match the order listed in `calculator/model_params.json`.
