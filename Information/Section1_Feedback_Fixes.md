# Section 1 Feedback — What Went Wrong & How to Fix

**Score: 5.5 / 10**

| Deduction | Points | Grader message |
|-----------|--------|----------------|
| Invalid-row exclusion count | −1 | Incorrect |
| Rating counts (bins 1–5) | −1 | Must include correct counts for all five bins |
| Original class counts | −0.5 | Incorrect |
| Seeded sample membership | −1 | Differs from required samples |
| Split rows / order / indices / types | −1 | Differs from required split |

These often **cascade**: a wrong validity filter → wrong `excluded_rows` → wrong `rating_counts` / `class_counts` → different rows available to sample → different `random_state=42` samples → different train/test frames.

---

## High level: how our Section 1 code works

`prepare_data(path, n_per_class)` is the only public entry point. It returns `(train_df, test_df, stats)`.

```
data.tsv
   │
   ▼
(1) Read TSV → only review_body, star_rating → rename to review, rating
   │
   ▼
(2) Drop invalid rows (blank review OR rating not in 1..5)
       record excluded_rows
   │
   ▼
(3) On remaining valid rows (before sampling):
       rating_counts[1..5]
       class_counts: positive (4–5), negative (1–2), neutral (3)
   │
   ▼
(4) Sample n_per_class positives and n_per_class negatives (seed 42 each)
       label: pos→1, neg→0
       drop neutrals by not sampling them
   │
   ▼
(5) Concat pos then neg, reset_index → 80/20 stratified split (seed 42)
   │
   ▼
return train_df, test_df, stats
```

**Helpers involved**

| Helper | Role |
|--------|------|
| `prepare_data_read_data` | `read_csv` + rename |
| `prepare_data_convert_and_remove_invalid_rows` | numeric ratings, filter, `excluded_rows` |
| `prepare_data_count_ratings_for_stats` | `rating_counts` + `class_counts` |
| `prepare_data_train_test_split` | sample, label, concat, `train_test_split` |

`stats` keys required by the interface:

- `rating_counts` — `{1: n, 2: n, 3: n, 4: n, 5: n}` (include zeros)
- `class_counts` — `{"positive", "negative", "neutral"}`
- `excluded_rows` — int, invalid **loaded** rows (each row once)

Printed script lines only use `class_counts`. Graders still check `rating_counts` and `excluded_rows` via `prepare_data` directly.

---

## Specifics: current logic vs handout

### (a) Read & filter — where `excluded_rows` / counts come from

**What we do now**

1. Read with the required Pandas options (`sep="\t"`, `dtype=str`, `QUOTE_NONE`, `keep_default_na=False`, `on_bad_lines="skip"`, `usecols=...`).
2. `pd.to_numeric(rating, errors="coerce")`.
3. Keep rows where `review != ""` **and** `rating.isin([1, 2, 3, 4, 5])`.
4. `excluded_rows = (~valid).sum()`.
5. Cast rating to `int`.

**Likely problems**

1. **Blank reviews too narrow**  
   Handout: exclude **missing/blank** reviews. We only drop exact `""`.  
   Whitespace-only reviews (`"   "`) may still be kept. Graders often treat those as blank.  
   **Fix to try:** validity check with `review.str.strip() != ""`, but **do not** change the stored review text (preserve original text).

2. **Integer 1–5 check**  
   `isin([1,2,3,4,5])` is usually enough after `to_numeric`. If anything weird slips through, tighten with “numeric, whole number, between 1 and 5.”  
   Still preserve row order; only drop invalid rows.

3. **Count only loaded rows**  
   Lines skipped by `on_bad_lines="skip"` never enter the DataFrame — they must **not** be added to `excluded_rows`. Our “count from the mask” approach is correct for that; don’t use file line counts.

4. **`rating_counts` keys**  
   Must be Python `int` keys `1..5`, including zeros. Prefer `{int(i): int(...) for i in range(1, 6)}` so keys aren’t `np.int64`.

If (1)–(2) are wrong, **rating_counts and class_counts will also be wrong** (−1 and −0.5), even if the counting code itself looks fine.

### (b) Labels & sampling — sample membership (−1)

**What we do now**

- Positive pool: `rating >= 4` → `label = 1`
- Negative pool: `rating <= 2` → `label = 0`
- Neutral `3` counted in `class_counts`, then discarded (not sampled)
- `.sample(n=n_per_class, replace=False, random_state=42)` **separately** on each pool
- Counts are computed **before** this sampling (correct)

**Why membership can differ**

`DataFrame.sample(..., random_state=42)` depends on:

1. **Which rows** are in the pool  
2. **Row order** of that pool  

If our valid-row filter differs from the grader’s, the pools differ → the seeded sample differs even with the same `random_state=42`.

**Fix direction**

- First make exclusion match the handout exactly (blank + rating rules).
- Keep original review order when filtering (`loc` / boolean mask, no sort).
- Keep separate `random_state=42` per class (already done).
- Do not resample, shuffle the pool, or drop duplicates unless the handout says so.

### (c) Split — rows, order, indices, types (−1)

**What we do now**

- `pd.concat([pos_df, neg_df], ignore_index=True)` → positive block, then negative, new index `0 .. 2*n_per_class-1`
- Keep columns `review`, `rating`, `label` only
- `train_test_split(..., test_size=0.2, stratify=label, random_state=42)`
- Do **not** `reset_index` again after the split (important: grader wants indices from the concatenated sample)

**Likely problems**

1. **Dtypes** — interface: ratings and labels are **integers**. After split, force:
   - `rating` → `int`
   - `label` → `int`  
   on both `train_df` and `test_df`.

2. **Indices** — do not call `reset_index(drop=True)` on train/test after splitting. Keep whatever indices `train_test_split` left from the concatenated frame.

3. **Order** — do not re-sort train/test. Preserve split output order.

4. **Columns** — exactly `review`, `rating`, `label` (no extra `cleaned` etc. inside `prepare_data`).

If sample membership is already wrong, the split will fail this check even when split code is right. Fix filter + sample first, then verify dtypes/indices.

---

## Recommended fix checklist (in order)

1. **Tighten invalid-row rule**
   - Blank: empty after strip (mask only; keep original `review` text).
   - Rating: numeric and in integers `{1,2,3,4,5}`.
   - `excluded_rows =` number of loaded rows failing that rule (once each).

2. **Recompute stats on the filtered frame**
   - `rating_counts`: all five bins, Python `int` keys, include zeros.
   - `class_counts`: pos = 4–5, neg = 1–2, neutral = 3 (before sampling).

3. **Resample with unchanged seeds**
   - Pos / neg pools from the corrected valid frame, order preserved.
   - `random_state=42` each; concat pos then neg; `ignore_index=True`.

4. **Split hygiene**
   - `test_size=0.2`, `stratify=labels`, `random_state=42`.
   - Integer `rating` / `label`.
   - Keep post-split indices; no extra shuffle/sort.

5. **Sanity checks on a tiny grader-like TSV**
   - Known blank / bad rating rows → check `excluded_rows`.
   - Known star counts → check `rating_counts` and `class_counts`.
   - Run twice → identical train/test index sets (determinism).

---

## What probably does *not* need changing

- Pandas `read_csv` options / `usecols` / rename (already match the handout).
- Mapping 4–5 → positive, 1–2 → negative, drop 3 from sampling.
- Computing counts before sampling.
- Concat positive then negative before `train_test_split`.
- Using `n_per_class` for smaller grading inputs.

---

## Bottom line

Section 1 isn’t failing because the overall pipeline shape is wrong. It’s failing because the **valid-row definition (and/or dtypes/indices after split)** doesn’t match the hidden tests. Wrong exclusion is the most likely root cause; wrong samples and wrong split are the downstream symptoms. Fix the filter and count types first, then confirm seeded samples and split indices/dtypes match the interface.
