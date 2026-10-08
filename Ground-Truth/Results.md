
## Tree Reconstruction Evaluation

### Precision, Recall, and F-measure

The reconstructed trees were evaluated against their corresponding ground-truth trees using edge-based Precision, Recall, and F-measure (F1-score).

| Tree | Total Edges | Common Edges (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F-measure |
|------|-------------|-------------------|----------------------|----------------------|-----------|--------|-----------|
| Tree 1 | 46 | 45 | 1 | 1 | 0.9783 | 0.9783 | 0.9783 |
| Tree 2 | 82 | 77 | 5 | 5 | 0.9390 | 0.9390 | 0.9390 |
| Tree 3 | 337 | 309 | 28 | 28 | 0.9169 | 0.9169 | 0.9169 |
| Tree 4 | 32 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9688 |

### Edge-Level Differences

Extra edges (False Positives) are present in the reconstructed tree but absent from the ground truth. Missing edges (False Negatives) are present in the ground truth but absent from the reconstructed tree.

#### Tree 1 — F-measure: 0.9783

**Extra edges (1):**
- `seq87 — seq88`

**Missing edges (1):**
- `seq1 — seq88`

#### Tree 2 — F-measure: 0.9390

**Extra edges (5):**
- `seq17 — seq18`
- `seq192 — seq47`
- `seq20 — seq39`
- `seq20 — seq40`
- `seq222 — seq223`

**Missing edges (5):**
- `naive — seq17`
- `naive — seq39`
- `naive — seq40`
- `seq192 — seq50`
- `seq223 — seq6`

#### Tree 3 — F-measure: 0.9169

**Extra edges (28, first 10 shown):**
- `seq15 — seq16`
- `seq150 — seq38`
- `seq151 — seq38`
- `seq176 — seq87`
- `seq177 — seq87`
- `seq192 — seq193`
- `seq197 — seq97`
- `seq3 — seq902`
- `seq30 — seq962`
- `seq371 — seq737`

*18 additional extra edges omitted.*

**Missing edges (28, first 10 shown):**
- `naive — seq444`
- `seq105 — seq418`
- `seq11 — seq193`
- `seq11 — seq197`
- `seq11 — seq388`
- `seq11 — seq767`
- `seq11 — seq768`
- `seq11 — seq791`
- `seq11 — seq792`
- `seq117 — seq911`

*18 additional missing edges omitted.*

#### Tree 4 — F-measure: 0.9688

**Extra edges (1):**
- `seq155 — seq156`

**Missing edges (1):**
- `naive — seq156`

### Results Summary

All four reconstructed trees achieved F-measures above 0.91, indicating strong agreement with their corresponding ground-truth trees.

- **Tree 1:** Highest F-measure (0.9783), with 1 false positive and 1 false negative.
- **Tree 2:** F-measure of 0.9390, with 5 false positives and 5 false negatives.
- **Tree 3:** F-measure of 0.9169, with 28 false positives and 28 false negatives.
- **Tree 4:** F-measure of 0.9688, with 1 false positive and 1 false negative.

Overall, the reconstructed trees show high edge-level accuracy, with relatively few discrepancies compared to the total number of edges.
