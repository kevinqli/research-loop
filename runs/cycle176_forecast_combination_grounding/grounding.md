# Cycle 176 — GROUNDING of c175 (ensemble-vs-select) against the FORECAST COMBINATION PUZZLE

**Mode:** meth-22 GROUND (literature cross-check of a fresh finding); the literature-scan cadence's "deeper read AFTER initial results". Diversifies MODE off the c175 sim. Free (WebSearch/WebFetch).

## Pre-registration (written before searching)
- PREDICT the literature CONFIRMS c175's core: the "forecast combination puzzle" (equal-weight averaging beats estimated-optimal-weight combination) is famous precisely because ESTIMATION ERROR in selection/weights defeats it — matching c175's P3 (ensemble beats select under a noisy verifier) + the P2 variance-reduction half.
- PREDICT it DIFFERS in framing: econometrics literature = point-forecast MSE accuracy, NOT c175's winner's-curse-on-the-self-REPORT (honesty) angle, nor the anti-correlated-verifier catastrophe, nor the combinability caveat — those are c175's additions.
- PREDICT a confirm-and-sharpen nuance: c175's P1 (SELECT beats ensemble in TRUE quality with a GOOD verifier) = the regime where the literature says selection CAN win; the "puzzle" (combination wins) is specifically the NOISY-selection regime.

## Sources + verbatim quotes (WebFetch / WebSearch, 2026-06-15)
1. **Chan & Pauwels 2015, "On the Forecast Combination Puzzle"** (arXiv:1505.00475, abstract, WebFetch):
   > "It is often reported in forecast combination literature that a simple average of candidate forecasts is more robust than sophisticated combining methods."
   > [the proposed strategy] "avoid[s] the heavy cost of estimation error" — i.e. "estimated optimal weights suffer from sampling variability that simple equal-weighting avoids."
2. **Lee & Lee 2023, "Solving the Forecast Combination Puzzle"** (arXiv:2308.05263, abstract, WebFetch):
   > "predictions formed by combining multiple forecasts in ways that seek to optimize forecast performance often do not out-perform more naive, e.g. equally-weighted, approaches"
   > Attributes the puzzle partly to TEST POWER / size: "tests that aim to discriminate between the predictive accuracy of competing combination strategies can have low power, and can lack size control" (non-standard asymptotic distribution). [= the honest hedge: estimation error is A leading explanation, not THE settled one.]
3. **Claeskens, Magnus, Vasnev, Wang 2016, "The forecast combination puzzle: A simple theoretical explanation"** (Int. J. Forecasting; via WebSearch summary):
   > "The additional uncertainty associated with estimating the weights explains this phenomenon, showing that if the optimal weights are close to equal, then a simple average of competing forecasts can dominate a forecast combination with weights that are estimated from the data."
   > "Equal weights are often a good choice using the tradeoff between bias (reflecting the error resulting from underfitting training data when choosing equal weights) and variance (quantifying the error resulting from the uncertainty when estimating other weights)."
4. **ML model-averaging-vs-selection** (WebSearch summary, multiple sources incl. arXiv:2110.10832):
   > "Model averaging outperforms model selection on average ... on average, 17% more observations are needed for model selection to achieve similar precision as model averaging."
   > "ensembles and averaged models comprise mainly the bias term only" / "reduces variance and improves generalization"; "decreased RMSE from averaging is accompanied by a relative increase in bias" (= regression-to-mean / loses the tail).
   Origin of forecast combination: Bates & Granger (1969) — combining two forecasts can beat either alone (standard attribution; not independently re-verified this cycle — flagged).

## Verification — agreements (CONFIRM) vs differences (c175 value-add)
**CONFIRMS c175 (4):**
1. c175's core finding (averaging beats selecting when selection is noisy) IS the canonical forecast combination puzzle — a 50+-yr, named, robust phenomenon. c175 re-derives it in the agent-fleet domain.
2. Mechanism: c175's P3 (ensemble beats select under a NOISY verifier) = the puzzle's "estimation error in the weights" (Claeskens 2016; Chan & Pauwels 2015). The verifier's noise IS the estimation error.
3. Bias-variance: c175's P1/P2 (ensemble regresses to mean = higher bias / lower variance; select keeps tail = lower bias / higher variance + the curse) = Claeskens 2016's explicit bias-vs-variance tradeoff AND the ML "averaging cuts variance, raises bias" framing.
4. Quantitative variance edge: ML "17% more observations for selection to match averaging" echoes c175's sqrt(K) variance reduction (ensemble_std = sigma_q/sqrt(K)).

**c175's VALUE-ADD beyond the literature (the c98/99 moat — novel derived synthesis):**
1. HONESTY / self-report: forecasting literature is MSE-ONLY (point-forecast accuracy). c175 adds that SELECT is winner's-CURSED in its self-REPORT (curse +0.16->+7.2) while ENSEMBLE is honest (curse ~0) — ties combination-vs-selection to the winner's-curse centerpiece. Not in the forecasting literature.
2. ANTI-correlated/broken verifier: c175's P3 (select delivers BELOW-mean -0.32 while reporting +3.6) is a sharper regime than "noisy weight estimate" — the selector is ACTIVELY wrong, not just high-variance.
3. COMBINABILITY caveat: forecasting assumes numeric averageable outputs; c175 makes explicit that for NON-combinable agent outputs (code patch/proof/essay) you MUST select -> the curse is unavoidable and held-out re-eval is the only fix. Agent-domain-specific.
4. The held-out re-eval as the HYBRID (select for the tail, re-estimate independently for honesty) — a third point on "don't condition the estimate on the selection"; forecasting frames it as binary select-vs-combine.

**Honest hedge / what could be wrong:**
- The puzzle has MULTIPLE debated explanations: estimation error (Claeskens/Chan-Pauwels) vs test power/size artifacts (Lee & Lee). c175 instantiates the ESTIMATION-ERROR mechanism cleanly (verifier noise), so it corroborates THAT explanation — it does NOT adjudicate the broader debate. Don't over-claim "estimation error is THE cause."
- Bates & Granger 1969 origin is the standard attribution (not independently re-verified this cycle).
- This is a literature grounding (meth-5: WebFetch'd the abstracts before citing); high confidence on the agreements (the puzzle is canonical + well-matched), appropriately hedged on which explanation.

## Verdict
c175 GROUNDED: ensemble-vs-select is the agent-fleet instance of the FORECAST COMBINATION PUZZLE (estimation-error explanation). Same meta-pattern as the curse's 4-domain grounding (c158) + detection-threshold=SDT (c170) + over-verification=alert-fatigue (c167): the loop's NEW-mechanism findings keep re-deriving canonical cross-field results; the value-add (moat) is consistently the agent-instantiation + the novel syntheses (here: honesty/self-report, anti-correlated verifier, combinability), NOT the core phenomenon. The Nth confirmation of c98/99 ("famous pieces free; the loop's novel derived+measured syntheses are the moat").
