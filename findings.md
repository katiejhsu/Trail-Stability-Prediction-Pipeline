Project Summary: Trail Shoe Stability Prediction
Katie Hsu

What I Built
I designed and implemented an end-to-end data pipeline to predict Trail Shoe Stability Rating using both quantitative sensor data and qualitative runner self-report — mirroring real life multi-source human subject data pipelines.

The pipeline takes in two raw datasets (shoe_specs.csv and runner_tests.csv), joins them via a SQL query in SQLite, cleans and encodes all features, trains a RandomForestRegressor, and exports a Power BI-ready dataset alongside a feature importance visualization.

Data Sources
Quantitative — objective, measurable signals:

- Shoe specs: stack height, midsole firmness, lug depth, heel-toe drop, outsole hardness
- Biomechanics: cadence (spm), ground contact time (ms), session duration, elevation gain

Qualitative — subjective runner self-report:

- Perceived Comfort (1–5 rating)
- Confidence on Descent (1–5 rating)
- Traction Feedback (categorical: Good / Excellent / Slipped Once / Felt Unstable)
- Would Recommend (Yes / Maybe / No)


Key Findings
The top 3 most useful features were all quantitative biomechanics:

Elevation Gain (0.197) — by far the strongest signal. The harder the trail (more climbing), the more the stability rating varied. Makes sense as a trail runner — a 500ft gain on technical singletrack demands a completely different shoe than a flat fire road.
Cadence (0.148) — how many strides per minute the runner was taking. Higher cadence is generally associated with better running form, so this picking up as important suggests the model was detecting form-related stability patterns.
Session Duration (0.113) — how long the run was. Longer runs introduce fatigue, which degrades form and likely tanks stability ratings toward the end. This is the model picking up on a fatigue signal.


Qualitative data carried meaningful predictive weight. Confidence on Descent and Perceived Comfort consistently ranked among the mid-to-high importance features — comparable in signal strength to objective specs like Lug Depth and Midsole Firmness. This suggests that how a runner feels on trail is not just anecdotal — it's statistically informative for predicting stability outcomes. Both qualitative and quantitative features were useful inputs for predicting stability — but further analysis would be needed to understand the specific relationships between them

Lug Depth and Outsole Hardness were the strongest shoe-spec predictors, which aligns with trail running biomechanics: grip and ground feel are the primary mechanical drivers of stability on variable terrain like wet rock, mud, and loose scree.
Surface condition mattered, but less than expected. The Is_Rocky and Is_Wet binary flags ranked lower than runner self-report features, implying that a runner's subjective response to a surface may capture more nuance than a simple condition label — a finding that could inform how Brooks structures post-run survey instruments.

Product Implications
These findings suggest that optimizing trail shoe stability isn't purely a materials or geometry problem — runner confidence is a signal worth tracking alongside sensor data. For Brooks, this reinforces the value of pairing biomechanical testing with structured qualitative feedback collection, especially for trail-specific models like the Caldera and Cascadia lines where terrain variability is high and fit-feel tradeoffs are significant.

A logical next step would be running a correlation analysis to understand whether specific shoe measurements — like lug depth or heel-toe drop — actually drive changes in how confident or comfortable a runner feels. 

Note: The negative R² is expected on synthetic data since there's no real signal to learn. With actual Brooks runner data, the model would have genuine biomechanical patterns to pick up on and would perform meaningfully better."

Technologies Used
Python · SQL (SQLite) · scikit-learn · pandas · matplotlib · Power BI (export)

Steps Taken: 

Step 1 — Loading the CSVs
The script reads your two raw files off your computer into Python as dataframes (basically just tables in memory). It prints how many rows and columns each one has so you can confirm they loaded correctly. shoe_specs.csv has 30 rows (one per shoe model) and runner_tests.csv has 150 rows (one per test run a runner did).
Step 2 — SQL JOIN
This is where the two tables get connected. Think of it like a VLOOKUP in Excel — every test run in runner_tests.csv has a Shoe_ID, and we use that to pull in the matching shoe's specs from shoe_specs.csv. So instead of two separate tables, you now have one wide table with 18 columns containing everything about both the shoe and the runner's experience. This happens inside a temporary SQLite database that lives in memory and disappears when the script finishes.
Step 3 — Cleaning
This is the biggest step. Four things happen:

Standardising Trail Condition — the raw data has "dry", "DRY", "Dry" all meaning the same thing. This converts everything to a consistent format and replaces anything unrecognisable with the most common value.
Outlier capping — values like 999mm stack height or -50ms ground contact are clearly data errors. The IQR method calculates what a "normal" range looks like for each column and clips anything outside that range. This is also where the Stability Rating should be hard-capped at 10 (the fix mentioned above).
Median imputation — wherever a value is missing, it gets filled in with the median of that column. Better than deleting the whole row, and median is used instead of mean because it's less affected by any remaining outliers.
Encoding qualitative columns — the model can only work with numbers, so text responses get converted. "Good" and "Excellent" traction → 1, "Slipped Once" and "Felt Unstable" → 0. "Yes" to recommend → 1, "Maybe" → 0.5, "No" → 0. Likert scales (1–5) are already numbers so they stay as-is.

Step 4 — Training the model
The cleaned data gets split 80/20 — 80% is used to teach the model, 20% is held back to test it on data it's never seen. A Random Forest is essentially a large collection of decision trees (300 in this case) that each look at the data slightly differently and vote on a prediction. The final prediction is the average of all their votes. MAE tells you how far off predictions are on average (1.9 points on a 1–10 scale here). R² tells you how much better the model is than just guessing the average every time — negative means the synthetic data has no real pattern to learn, which is expected.

Step 5 — Feature importance chart
After training, the Random Forest can tell you which features it relied on most when making predictions. We fed it all 16 features and the model trained on all of them and it told us which features it relied on the most/thought was the most useful for the dataset when making predictions. The chart visualises this, with qualitative features highlighted in light teal so you can see at a glance how runner self-report compares to objective shoe specs.

Step 6 — Power BI export
The cleaned dataset gets a new column appended — Predicted_Stability — with the model's prediction for every row. That full table is saved as a CSV that you can drag straight into Power BI to build dashboards.