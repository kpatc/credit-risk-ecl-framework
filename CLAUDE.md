# CLAUDE.md — ECL Modeling Project

## Contexte
Projet ECL (Expected Credit Loss) en deux phases :
- **Phase I** : dataset Loan Default (~148k obs) — WOE/IV + logistic regression PD
- **Phase II** : Freddie Mac mortgage (113k obs mensuelles, 7563 prêts) — pipeline complet PD → EAD → Lifetime PD → ECL

---

## Problèmes à corriger lors de la restructuration

### Hardcoded paths Windows (à centraliser dans config.yaml)
Tous les scripts Phase I et II contiennent :
```python
BASE_DIR = Path(r"C:\Python\ECL modeling")
```
À remplacer par un fichier de configuration unique.

### Scripts ad hoc à trier
`Status.py`, `Status_2.py`, `quick_check.py`, `create_graphs.py`, `provera_PD.py` — exploratoires, à déplacer ou supprimer.

---

## Ordre d'exécution

### Phase I (Loan Default)
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16
```
- 08 produit : `outputs/data_preparation/train_woe.csv` + `test_woe.csv`
- 09–13 consomment ces fichiers
- 16 exporte : `outputs/final_PD_model/final_PD_logistic_model.joblib`

### Phase II (Freddie Mac)
```
17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27 → 28 → 29 → 30 → 31
```
- 20 produit : `outputs/default_analysis/pd_12m_target_feasibility.csv`
- 21 produit : `outputs/pd_model/pd_modeling_dataset.csv`
- 25 produit : `outputs/pd_model/logistic_pd_model.joblib`
- 26 produit : `outputs/pd_model/portfolio_pd_snapshot_202109.csv`
- 29 produit : `outputs/ead/ead_portfolio_202109.csv`
- 30 produit : `outputs/lifetime_pd/lifetime_pd_portfolio_202109.csv`
- 31 produit : `outputs/ecl/ecl_loan_level_202109.csv` + `ecl_scenario_summary_202109.csv`

---

## Structure cible

```
ecl_modeling/
├── config/
│   └── config.yaml              # paths, hyperparams, LGD scenarios
├── data/
│   ├── raw/                     # données source (gitignored)
│   └── processed/               # intermédiaires (gitignored)
├── src/
│   ├── data/
│   │   ├── loaders.py           # fonctions I/O
│   │   └── validators.py        # checks réutilisables
│   ├── features/
│   │   ├── woe.py               # WOETransformer class
│   │   └── engineering.py       # feature pipelines
│   ├── models/
│   │   ├── pd_model.py          # PDModel class
│   │   ├── ead.py               # EADEstimator
│   │   ├── lgd.py               # LGDScenarios
│   │   └── lifetime_pd.py       # LifetimePDProjector
│   └── ecl/
│       └── calculator.py        # ECLCalculator class
├── notebooks/                   # EDA (non-prod)
├── tests/                       # pytest
├── outputs/                     # gitignored
├── models/                      # artefacts .joblib versionnés
├── requirements.txt
└── CLAUDE.md
```

---

## Logique métier à préserver dans les classes

### WOETransformer (src/features/woe.py)
- Fit uniquement sur train, apply sur test
- Lissage alpha=0.5 (Laplace)
- Catégories inconnues → WOE neutre = 0
- Bin edges sauvegardées pour reproducibilité

### PDModel Phase I (src/models/pd_model.py)
- Input : features WOE-transformées
- `LogisticRegression(max_iter=2000, solver="lbfgs")`
- Export joblib avec metadata JSON (feature order, input format)

### PDModel Phase II (src/models/pd_model.py)
- GroupShuffleSplit par `id_loan` (anti-leakage sur données panel)
- Sample weights : `1 / count_obs_per_loan`, normalisé à mean=1
- Pipeline : median imputation + StandardScaler + OneHotEncoder(handle_unknown="ignore") + LogisticRegression

### ECLCalculator (src/ecl/calculator.py)
Entrées par prêt : `ead`, `pd_12m`, `mths_remng`, `orig_int_rt`, LGD scenarios
1. `monthly_hazard = 1 - (1 - pd_12m)^(1/12)`
2. `marginal_pd_t = (1-h)^(t-1) * h`
3. EAD projeté : amortissement standard (ou linéaire si taux ≈ 0)
4. `DF_t = (1 + r)^(-(t-0.5)/12)` (convention mid-month)
5. `ECL = Σ marginal_pd_t * EAD_t / DF_t * LGD`

Checks obligatoires après calcul :
- 12m ECL ≤ Lifetime ECL
- Ordering LGD scenarios
- `Σ marginal_pd = lifetime_pd` (atol=1e-10)

### LifetimePDProjector (src/models/lifetime_pd.py)
`lifetime_pd = 1 - (1 - pd_12m)^(T/12)` — constant hazard assumption

---

## Résultats de référence (ne pas casser)

| Composante | Valeur |
|---|---|
| Phase I PD AUC test | 0.7024 |
| Phase II PD AUC obs-weighted | 0.7669 |
| Phase II PD AUC loan-weighted | 0.7481 |
| Portfolio Sept 2021 — nb prêts | 4 729 |
| Portfolio Sept 2021 — Total EAD | 887.83M |
| Portfolio Sept 2021 — PD moyen | 3.66% |
| ECL 12m Base | 14.07M |
| ECL Lifetime Base | 91.77M |

---

## Ce qu'il faut ajouter en prod (pas encore implémenté)

- Out-of-time validation (coupure temporelle, pas GroupShuffleSplit aléatoire)
- Logging (remplacer tous les `print()`)
- Type hints sur toutes les fonctions
- Tests unitaires : WOE bins, ECL reconciliation, anti-leakage assertion
- IFRS 9 staging (Stage 1/2/3, SICR)
- Macroeconomic scenarios pour forward-looking adjustment
- EIR calculé, pas taux contractuel comme proxy discount rate
