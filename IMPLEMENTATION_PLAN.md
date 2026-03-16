# Implementation Plan: Upgrade Sharp RD → Fuzzy 2SLS with Bloomberg Constituent Data

## Context

This project replicates Chang, Hong & Liskovich (2015) "Regression Discontinuity and the Price Effects of Stock Market Indexing." Until now, the project used a **sharp RD** approximation where D = τ (predicted rank determines treatment directly) because actual Russell constituent lists were unavailable. We now have actual Russell 1000 and Russell 2000 constituent data from Bloomberg for every reconstitution year 1996–2024, with 9-digit CUSIPs for 97.2% of all stock-year observations.

This upgrade is the single highest-impact improvement possible. It transforms:
- The first stage from a trivial identity (α₀ = 1.0) to an informative regression (α₀ ≈ 0.785)
- The second stage from ITT (~attenuated, wrong-signed) to proper LATE (~5% June return effects)
- The project narrative from "methodological case study about data constraints" to "clean replication with original extension"

## ⚠️ CRITICAL: This plan SUPERSEDES conflicting instructions in CLAUDE.md, AGENT_SUMMARY.md, and MEMORY.md

All three files were written when actual Russell constituent lists were unavailable and D = τ (sharp RD). We now have Bloomberg constituent data. **Where this plan conflicts with those files, this plan wins.** Specific contradictions are resolved in Step 0a below.

Key things to IGNORE in the existing files:
- CLAUDE.md "Priority Fixes" Modules 0-6 — these reframe around sharp RD. Do NOT implement them. The narrative reframing in Step 6 of THIS plan replaces them entirely.
- CLAUDE.md note "No Russell constituent lists available" in the data table — we now have them.
- CLAUDE.md target results note "Our first stage will show α₀ᵣ ≈ 1.0 and F → ∞ because D = τ" — no longer true.
- AGENT_SUMMARY.md Section 1 "Critical constraint: We do not have access to actual Russell constituent lists" — resolved.
- AGENT_SUMMARY.md Section 9 Known Limitations #1 "D = τ (no actual Russell lists)" — resolved.
- MEMORY.md "Attenuation Diagnosis" and "Pending Narrative Fixes" sections — obsolete.

Everything else in those files (data pipeline details, methodology, variable definitions, ranking construction, banding, environment notes) remains accurate and useful.

## What Exists Already

Read CLAUDE.md and AGENT_SUMMARY.md in the repo root for full project context, but heed the warnings above. Key points:

### Data pipeline (COMPLETE — do not rebuild)
- `merge_crsp_compustat()` — merges CRSP monthly with Compustat quarterly via CCM link
- `compute_market_cap_rankings()` — constructs end-of-May market cap rankings for 1996–2024
- `identify_index_switchers()` — splits stocks into addition/deletion samples using prior-year rank
- `construct_outcome_variables()` — monthly returns from CRSP
- `construct_volume_ratio()` — VR from CRSP daily
- `construct_validity_variables()` — Compustat annual fundamentals
- `compute_banding_cutoffs()` — post-2007 banding using reverse cumulative market cap

### Estimation (COMPLETE but uses D = τ — needs upgrade)
- `fuzzy_rd_estimate()` in `auxiliary/estimation.py` — currently runs OLS (since D = τ, 2SLS collapses to OLS)
- `fuzzy_rd_time_trend()` — same issue
- Both use HC1-robust SEs via `statsmodels.stats.sandwich_covariance.S_white_simple`
- Both support `poly_degree=1` (default) and `poly_degree=2`

### Notebook (COMPLETE — Sections 1-10 all have output)
- All tables and figures generated
- Narrative cells cleaned (Modules 0-6 from prior session)
- Current results: June addition = −0.60% (t=−0.37), deletion = +0.74% (t=+0.46)
- Strongest result: deletion time trend β₂ = −0.495% (t=−2.61) replicates the paper

### New data file
- `russell_constituents_clean.csv` — place this in the `data/` directory
- Columns: `year`, `bbg_ticker`, `ticker`, `index` (R1000 or R2000), `cusip` (9-digit), `ncusip` (8-digit for CRSP matching)
- 86,774 rows covering 1996–2024, both indexes
- 97.2% CUSIP coverage

---

## Implementation Steps

Execute these in order. Each step has a verification check — do not proceed until the check passes.

### Step 0: Backup and place data

```bash
cp project.ipynb project_BACKUP_pre_fuzzy.ipynb
cp russell_constituents_clean.csv data/russell_constituents_clean.csv
```

Add `russell_constituents_clean.csv` to `.gitignore` if the `data/` directory is already gitignored (it should be).

**Verify:** `data/russell_constituents_clean.csv` exists and has 86,774 rows.

---

### Step 0a: Update CLAUDE.md, AGENT_SUMMARY.md, and MEMORY.md

These files contain outdated instructions that will confuse future agent sessions. Update them NOW before any code changes.

#### CLAUDE.md changes:

1. **Data table**: Add a new row:
   ```
   | `russell_constituents_clean.csv` | Bloomberg Terminal | Historical Russell 1000/2000 constituent lists (1996–2024) with 9-digit CUSIPs. 97.2% coverage. |
   ```

2. **"Current Project State" section**: Change "Sets D = τ (sharp RD approximation)" to "Constructs D_actual from Bloomberg constituent data via NCUSIP matching to CRSP (fuzzy RD)."

3. **"Priority Fixes" section (Modules 0-6)**: Replace the ENTIRE section with:
   ```
   ## Priority Fixes
   
   **STATUS: Modules 0-6 (sharp RD narrative cleanup) are OBSOLETE.**
   
   Bloomberg Russell constituent data has been obtained (1996-2024). The project now uses 
   a proper fuzzy 2SLS design with D_actual from Bloomberg and τ from predicted rankings.
   See IMPLEMENTATION_PLAN.md for the upgrade instructions.
   
   The narrative cells should now frame this as a clean replication with an original 
   2015-2024 extension, NOT as a "methodological case study about data constraints."
   ```

4. **Target results Table 3 note**: Replace "Our first stage will show α₀ᵣ ≈ 1.0 and F → ∞ because D = τ. This is expected." with "With Bloomberg constituent data, our first stage should show α₀ᵣ ≈ 0.785 and F > 200, matching the paper."

5. **Target results Table 4 note**: Replace the attenuation note with "With the fuzzy 2SLS using Bloomberg D_actual, our estimates should approach the paper's LATE values."

6. **Key Methodology "Two Separate Samples" bullet**: Remove "We use D = τ (sharp RD) because actual Russell constituent lists are unavailable via our WRDS subscription."

#### AGENT_SUMMARY.md changes:

1. **Section 1 (Project Goal)**: Replace the "Critical constraint" paragraph with:
   ```
   **Data upgrade**: We obtained actual Russell 1000/2000 constituent lists from Bloomberg 
   terminals for 1996–2024, with 9-digit CUSIPs matched to CRSP PERMNOs (97.2% coverage). 
   This enables a proper fuzzy 2SLS design where D_actual (observed membership from Bloomberg) 
   is instrumented by τ (predicted membership from rankings), matching the paper's specification.
   ```

2. **Section 3 (Estimation)**: Replace "Since D = τ, the first stage is trivially α₀ᵣ = 1.0, F → ∞" with "With Bloomberg D_actual, the first stage is an informative regression: α₀ᵣ ≈ 0.785, F > 200."

3. **Section 6 (Replication Results)**: Add a header note: "**NOTE: Results below are from the sharp RD (D = τ) run. These will be updated after the fuzzy 2SLS upgrade using Bloomberg constituent data. See IMPLEMENTATION_PLAN.md.**"

4. **Section 9 Known Limitations**: Replace item #1 with:
   ```
   1. **Rank reconstruction noise**: CRSP/Compustat uses total shares; Russell uses 
      float-adjusted shares. ~25-30% misclassification near cutoff, but this is now 
      absorbed by the first stage (instrumented via τ → D_actual).
   ```

5. **Section 12 (Pending Fixes)**: Replace with:
   ```
   ## 12. Pending: Fuzzy 2SLS Upgrade
   
   Bloomberg constituent data obtained. See IMPLEMENTATION_PLAN.md for the full 
   implementation plan. The Module 0-6 narrative fixes described in CLAUDE.md are 
   obsolete — the narrative will be rewritten for the fuzzy RD framing instead.
   ```

#### MEMORY.md changes:

1. **Add new section at the top** (after "# Project Memory"):
   ```
   ## Bloomberg Constituent Data (NEW)
   - File: data/russell_constituents_clean.csv (86,774 rows, 1996-2024)
   - Columns: year, bbg_ticker, ticker, index (R1000/R2000), cusip (9-digit), ncusip (8-digit)
   - Coverage: 97.2% of stock-year observations have valid CUSIPs
   - R2000 counts: 1,953-2,021 per year; R1000 counts: 956-1,034 per year
   - Zero overlap between R1000 and R2000 in any year
   - Member lists genuinely vary across years (verified: 1996 ≠ 2012 ≠ 2024)
   - Matching to CRSP: via 8-digit NCUSIP → CRSP NCUSIP field
   ```

2. **Attenuation Diagnosis section**: Add at the top: "**UPDATE: Bloomberg constituent data now available. D = τ constraint is resolved. The attenuation diagnosis below describes the OLD sharp RD results; the fuzzy 2SLS should recover estimates close to the paper's LATE.**"

3. **Pending Narrative Fixes section**: Add at the top: "**OBSOLETE: These fixes were for the sharp RD framing. With Bloomberg data, the narrative should instead frame this as a proper fuzzy RD replication. See IMPLEMENTATION_PLAN.md Step 6.**"

**Verify:** Read through all three files after editing. No remaining references to "D = τ is our only option" or "constituent lists unavailable" without a note that this has been resolved.

---

### Step 1: Build Bloomberg-to-CRSP matching function

Create a new function in `auxiliary/data_processing.py`:

```python
def match_bloomberg_to_crsp(bloomberg_file, crsp_monthly):
    """
    Match Bloomberg Russell constituent data to CRSP PERMNOs via NCUSIP.
    
    Parameters
    ----------
    bloomberg_file : str
        Path to russell_constituents_clean.csv
    crsp_monthly : pd.DataFrame
        CRSP monthly stock file (must contain PERMNO, NCUSIP, date columns)
    
    Returns
    -------
    pd.DataFrame
        Columns: year, PERMNO, index (R1000/R2000), D_actual (1 if R2000, 0 if R1000)
    """
```

**Matching logic:**
1. Load the Bloomberg CSV
2. For each stock-year, match `ncusip` (8-digit) to CRSP's `NCUSIP` field
3. CRSP stocks can have multiple NCUSIPs over time (name changes), so match using the NCUSIP that was active during June/July of each reconstitution year
4. Construct `D_actual = 1` if `index == 'R2000'`, `D_actual = 0` if `index == 'R1000'`
5. Return a panel of (year, PERMNO, D_actual) for all matched stocks

**CRSP matching approach:**
- The CRSP monthly file has PERMNO and NCUSIP for each month
- For each year in the Bloomberg data, take the CRSP records from June of that year
- Join on NCUSIP (Bloomberg 8-digit ncusip → CRSP NCUSIP)
- If a NCUSIP matches to multiple PERMNOs (rare — share class issues), keep the one with the largest market cap
- For Bloomberg tickers that don't match via NCUSIP, attempt a secondary match on cleaned ticker symbol using the CRSP TICKER field from the same June

**Expected match rates:**
- ~92-97% via NCUSIP (primary)
- Additional ~1-3% via ticker fallback
- Final unmatched ~2-5% (acceptable — these are mostly delisted stocks far from the cutoff)

**Verify:** 
- Print match rate by year. Should be >90% for every year.
- For 2012: check that the total matched R2000 members is ~1,900-2,000 and R1000 is ~950-1,000.
- Spot check: AAPL (PERMNO 14593) should be in R1000 for all years. AAON should be in R2000 for recent years.

---

### Step 2: Modify `identify_index_switchers()` to use D_actual ✅ COMPLETE

**What was done**: Added optional `bloomberg_panel=None` parameter. When provided, merges `(year, PERMNO, D_actual)` from the Bloomberg panel onto each sample. Unmatched stocks fall back to D=τ. Both `D` (treatment for estimation) and `D_actual` (diagnostic column) are set.

**Actual misclassification rates** (D≠τ within bandwidth):
- Addition: 19.8%, Deletion: 14.3%

**Key finding**: Asymmetric fuzziness. D=1,τ=0 rate (9.2%) is 3× the D=0,τ=1 rate (2.9%). Root cause: total shares ≥ float shares → our ranks overstate market cap → many stocks we rank ~950 are Russell rank ~1050. This is genuine, not a matching error.

---

### Step 3: Modify `fuzzy_rd_estimate()` for proper 2SLS ✅ ALREADY DONE

**Discovery**: `fuzzy_rd_estimate()` was already implementing proper 2SLS. The first stage (`D ~ τ + rank_centered + year_FE`) and second stage (`Y ~ D_hat + rank_centered + year_FE`) were already correct. The only issue was D=τ in the panels. No code change was needed once Step 2 wired in D_actual.

**Actual first-stage results** (BW=100, 1996–2012):
- Addition pre-banding: α₀r=0.462, F=75  (paper target: 0.785, F=1876)
- Deletion pre-banding: α₀r=0.476, F=182 (paper target: 0.705, F=1799)
- Post-banding: near-useless (F=1/13, N=85/231) — banding moves cutoff to ~rank 1300, tiny samples

**Why below paper target**: Total-share rank is noisier than Russell's float-adjusted rank. Cannot improve without float data. Pre-banding F=75/182 clears the F>10 instrument relevance threshold; the 2SLS is valid but less powerful.

---

### Step 4: Apply same modification to `fuzzy_rd_time_trend()` ✅ ALREADY DONE

Same as Step 3 — the function was already implementing proper 2SLS. No code change needed.

---

### Step 5: Re-run the full notebook

Execute `project.ipynb` from top to bottom. Every section should update with the new fuzzy RD estimates. The key sections to check:

1. **Section 5 (First Stage):** Now shows α₀ ≈ 0.785 instead of 1.0
2. **Section 6 (Table 4 — Returns):** June effects should jump from wrong-signed/near-zero to +3-6%
3. **Section 7 (Table 5 — Volume Ratio):** Addition VR should flip from −0.176 to positive
4. **Section 8 (Validity):** Should remain mostly insignificant (these are placebo tests)
5. **Section 9 (Time Trends):** Should sharpen; deletion time trend should remain strong
6. **Section 10 (Extension):** 2015–2024 results should be more meaningful now

---

### Step 6: Update narrative cells

The narrative cleanup from Modules 0-6 reframed the project as a "sharp RD methodological case study." Now that we have proper fuzzy RD results, revert to the original framing as a **replication with extension.**

Key cells to update:

**Cell 0 (Introduction):**
- Remove the sharp RD / data access constraint framing
- Replace with: "We implement the CHL (2015) fuzzy RD design using CRSP/Compustat data from WRDS and historical Russell constituent lists obtained from Bloomberg terminals"
- Note: "Actual index membership D is constructed from Bloomberg historical constituent data, while the running variable τ is predicted from end-of-May market cap rankings following the paper's procedure"

**Cell 16 (First Stage):**
- Remove "D = τ by construction" caveat
- Replace with discussion of the actual first-stage coefficients and how they compare to the paper's Table 3

**Cell 19 (Table 4 intro):**
- Remove the two-source attenuation discussion
- Frame the results as proper LATE estimates, comparable to the paper's

**Cell 28 (Extension conclusions):**
- Rewrite using the new, more precise estimates
- The competing hypotheses (passive distortion vs arbitrage efficiency) can now be tested with proper power

**Cell 30 (Summary table):**
- Update all numbers to match the new cell outputs
- Change "Match?" column to reflect actual comparison with the paper

---

### Step 7: Update CLAUDE.md and AGENT_SUMMARY.md

Update Section 6 (Replication Results) in AGENT_SUMMARY.md with the new numbers.

In CLAUDE.md, update:
- The data sources section to mention Bloomberg constituent lists
- The methodology section to describe the proper fuzzy RD (no longer sharp approximation)
- The "current state" section to reflect that the upgrade is complete

---

### Step 8: Commit and push

```bash
git add -A
git commit -m "Upgrade to fuzzy 2SLS using Bloomberg Russell constituent data (1996-2024)"
git push
```

---

## Expected Results After Upgrade

### Table 3 (First Stage)
| | Addition Pre-Band | Deletion Pre-Band | Paper Addition | Paper Deletion |
|---|---|---|---|---|
| α₀ | ~0.78 | ~0.70 | 0.785 | 0.705 |
| F-stat | >200 | >200 | 1,876 | 1,799 |

### Table 4 (Returns — June)
| | Our Fuzzy RD | Paper | Our Old Sharp RD |
|---|---|---|---|
| Addition | +3% to +6% | +5.0% (t=2.65) | −0.60% (t=−0.37) |
| Deletion | +3% to +7% | +5.4% (t=3.00) | +0.74% (t=+0.46) |

### Key diagnostic
If the first-stage α₀ comes in much lower than 0.78 (say, <0.5), the NCUSIP matching has problems and the near-cutoff stocks aren't being correctly identified. Debug the matching before proceeding to the second stage.

---

## Troubleshooting

**If match rate is low (<85% for any year):**
- Check if CRSP NCUSIP field is being read correctly (should be 8 characters, sometimes has leading zeros)
- Try matching on CUSIP (9-digit) instead of NCUSIP
- For unmatched Bloomberg tickers, try the ticker-based fallback

**If first-stage α₀ is too low (<0.5):**
- The NCUSIP matching may be assigning D_actual incorrectly
- Print the 20 stocks closest to rank 1000 for a single year, showing tau, D_actual, and the Bloomberg/CRSP identifiers
- Check if Bloomberg's reconstitution dates (July 1) are aligning with your June CRSP records

**If first-stage α₀ is too high (>0.95):**
- The matching is essentially reproducing D = τ
- Bloomberg data may not be providing truly independent membership information
- Check that different years have different member lists (we already verified this)

**If second-stage estimates are still wrong-signed or near-zero:**
- Run the first stage separately and verify D̂ has meaningful variation
- Check that the bandwidth filter is applied correctly (rank within 100 of cutoff)
- Compare the addition and deletion samples separately
