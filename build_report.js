const fs = require("fs");
const path = require("path");

const BASE_DIR = __dirname;
const RESULTS_DIR = path.join(BASE_DIR, "results");
const CHARTS_DIR = path.join(BASE_DIR, "charts");
const OUTPUT_DIR = path.join(BASE_DIR, "output");
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak, Header, Footer,
  PageNumber, NumberFormat
} = require("docx");

const R = (p) => JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, p), "utf8"));
const data = R("report_data.json");
const assocDia = R("assoc_diabetes.json");
const assocDep = R("assoc_depression.json");
const adjDia = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, "adjusted_diabetes_model.csv.json"), "utf8"));
const adjDep = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, "adjusted_depression_model.csv.json"), "utf8"));

const CLIENTS = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"];
const ERA = { "Client_A_2015-2016": "Hospital A (2015-2016)", "Client_B_2017-2020": "Hospital B (2017-2020)", "Client_C_2021-2023": "Hospital C (2021-2023)" };

const NAVY = "1F3864";
const ACCENT = "2E5C8A";
const LIGHT = "EDF2F7";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function img(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200, before: 100 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 260 },
  });
}

function cell(text, { header = false, width, shade } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), bold: header, color: header ? "FFFFFF" : "000000", size: 19 })],
    })],
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
  });
}

function table(headers, rows, widths) {
  const headerRow = new TableRow({
    children: headers.map((h, i) => cell(h, { header: true, width: widths[i], shade: NAVY })),
    tableHeader: true,
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((v, i) => cell(v, { width: widths[i], shade: ri % 2 === 1 ? LIGHT : undefined })),
  }));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...bodyRows],
  });
}

function fmtOR(row) {
  return `${row.OR.toFixed(2)} (${row.CI_low.toFixed(2)}\u2013${row.CI_high.toFixed(2)})`;
}
function fmtP(v) {
  return v < 0.001 ? "<0.001" : v.toFixed(3);
}

// ---------------- Client summary table ----------------
const clientRows = CLIENTS.map(c => {
  const s = data.clients[c];
  return [ERA[c], s.n_records.toLocaleString(), (s.depression_prevalence*100).toFixed(1)+"%", (s.diabetes_prevalence*100).toFixed(1)+"%"];
});
const totalN = CLIENTS.reduce((a,c) => a + data.clients[c].n_records, 0);

// ---------------- Model performance table ----------------
function perfRows(evalFed, evalLocal) {
  return CLIENTS.map(c => [
    ERA[c],
    evalFed[c].n_test,
    (evalFed[c].accuracy*100).toFixed(1)+"%",
    evalFed[c].auc.toFixed(3),
    evalLocal[c].auc.toFixed(3),
  ]).concat([[
    "Pooled", evalFed["POOLED"].n_test, (evalFed["POOLED"].accuracy*100).toFixed(1)+"%", evalFed["POOLED"].auc.toFixed(3), "\u2014"
  ]]);
}

// ---------------- Association tables ----------------
function assocRows(arr) {
  return arr.map(r => [r.label + " (" + r.code + ")", fmtOR(r), fmtP(r.p_value)]);
}

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } },
    },
    headers: {
      default: new Header({ children: [ new Paragraph({
        children: [new TextRun({ text: "Federated Learning \u2014 Depression & Diabetes (NHANES 2015\u20132023)", size: 16, color: "888888" })],
        alignment: AlignmentType.RIGHT,
      })]}),
    },
    footers: {
      default: new Footer({ children: [ new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [ new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" }) ],
      })]}),
    },
    children: [
      // ---- Title ----
      new Paragraph({
        children: [new TextRun({ text: "Federated Learning Analysis of", bold: true, size: 44, color: NAVY })],
        spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Depression\u2013Diabetes Association in NHANES", bold: true, size: 44, color: NAVY })],
        spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "2015\u20132023, Three Federated Hospitals", size: 28, color: ACCENT })],
        spacing: { after: 400 },
      }),
      p(`Prepared: ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}`, { italics: true, color: "555555" }),
      p(`Data source: NHANES cycles 2015\u20132016, 2017\u20132020, and 2021\u20132023, treated as three independent, non-pooled federated clients (hospitals) (N = ${totalN.toLocaleString()} eligible adult records combined).`, { color: "555555" }),

      new Paragraph({ children: [new PageBreak()] }),

      // ---- Executive Summary ----
      h1("Executive Summary"),
      p("This analysis applies federated learning (FedAvg) and federated statistical meta-analysis across three NHANES survey cycles \u2014 2015\u20132016, 2017\u20132020, and 2021\u20132023 \u2014 treated as three separate data-holding hospitals whose individual records are never pooled or centrally shared. Two prediction tasks were modeled: (1) depression status (PHQ-9 \u2265 10) and (2) elevated HbA1c / diabetes (\u22656.5%). In addition, item-level associations were computed for every demographic and PHQ-9 code with both outcomes."),
      h2("Key findings"),
      bullet("Depression and diabetes are significantly, bidirectionally associated even after accounting for the federated (non-pooled) structure of the data: each 1-point increase in PHQ-9 score raises the odds of diabetes by ~3.2% (OR " + data.link.PHQ9_Total_predicting_diabetes_binary.OR.toFixed(3) + ", p " + fmtP(data.link.PHQ9_Total_predicting_diabetes_binary.p_value) + "), and each 1-point increase in HbA1c raises the odds of depression by ~12.3% (OR " + data.link.LBXGH_predicting_depression_binary.OR.toFixed(3) + ", p " + fmtP(data.link.LBXGH_predicting_depression_binary.p_value) + ")."),
      bullet("The federated global model achieved test AUC of " + data.dia_eval.POOLED.auc.toFixed(3) + " for diabetes and " + data.dep_eval.POOLED.auc.toFixed(3) + " for depression, within 1\u20132 AUC points of hospital-local models trained without any federation \u2014 evidence that federation preserves predictive value while keeping raw data local."),
      bullet("Somatic/vegetative PHQ-9 items (sleep trouble, low interest, fatigue) show the strongest, most consistent links to elevated HbA1c across all three eras; the self-harm item (DPQ090) and \"feeling down\" (DPQ020) are the strongest predictors of overall depression status, as expected by PHQ-9 scoring design."),
      bullet("Depression prevalence rose steadily across cycles (9.0% \u2192 10.1% \u2192 13.1%), while diabetes prevalence rose then declined slightly (12.8% \u2192 13.8% \u2192 11.3%), a pattern discussed further in Section 6."),

      // ---- Introduction ----
      h1("1. Introduction & Objectives"),
      p("Centralizing health survey data across time periods or institutions raises privacy, governance, and logistical barriers. Federated learning (FL) addresses this by training shared models across distributed data holders without ever exchanging raw records \u2014 only model parameters or aggregate statistics are communicated. This project treats the three NHANES survey cycles supplied by the user as three federated clients (hospitals) and asks:"),
      bullet("Can a federated model predict depression and diabetes status as well as models trained on data pooled centrally?"),
      bullet("What is the federated (privacy-preserving) association between each PHQ-9 item / demographic code and each health outcome?"),
      bullet("Is there evidence of a depression\u2013diabetes link that is robust across independently-held data partitions?"),

      // ---- Data & Clients ----
      h1("2. Data & Federated Hospital Architecture"),
      p("Each of the three uploaded files (Merged_2015-2016.xlsx, Merged_2017-2020.xlsx, Merged_2021-2023.xlsx) was assigned to its own federated client — simulating one hospital per dataset — matching the natural survey-era partitioning of the data. Records were retained only if both a PHQ-9 total score and an HbA1c (LBXGH) value were present (the two modules both requiring adult examination), since both outcomes are needed for the association analysis."),
      table(
        ["Hospital (NHANES cycle)", "Eligible N", "Depression prevalence", "Diabetes prevalence"],
        clientRows,
        [3600, 2400, 2520, 2520]
      ),
      caption("Table 1. Per-hospital sample sizes and outcome prevalence. Each row corresponds to one federated client (hospital); no hospital's rows were merged with another's during preparation."),
      p("Variables used, all standard NHANES codes:"),
      bullet("Demographics: RIAGENDR (gender), RIDAGEYR (age), DMDEDUC2 (education), INDFMPIR (income-poverty ratio)"),
      bullet("Body measure: BMXBMI (BMI)"),
      bullet("Depression module: DPQ010\u2013DPQ090 (individual PHQ-9 items) and PHQ9_Total (summed score, 0\u201327)"),
      bullet("Laboratory: LBXGH (glycohemoglobin / HbA1c, %)"),

      // ---- Methodology ----
      h1("3. Methodology \u2014 Federated Rules Applied"),
      h2("3.1 Local data preparation"),
      p("Each hospital cleaned and imputed its own missing covariates (education, income-poverty ratio, BMI) using only its own median \u2014 no hospital's imputation was informed by another hospital's records."),
      h2("3.2 Federated feature normalization"),
      p("Feature scaling (z-score) requires a global mean/standard deviation. Rather than pooling raw values, each hospital shared only its local (n, mean, variance) per feature. The coordinator combined these into an exact global mean/std using the parallel-variance (Chan's) aggregation formula \u2014 mathematically identical to computing on pooled data, but without any patient row ever leaving its hospital."),
      h2("3.3 Federated Averaging (FedAvg) for prediction"),
      p("For each of the two prediction tasks, a logistic regression model was trained via FedAvg:"),
      bullet("Each hospital trains the current global model locally for 25 epochs of gradient descent on its own patient data only."),
      bullet("Each hospital sends back only its updated model weights (not data or per-record gradients)."),
      bullet("The coordinator aggregates weights via a sample-size-weighted average: w_global = \u03a3 (n_k / N) \u00d7 w_k."),
      bullet("This repeats for 15 communication rounds, after which the final global model is evaluated on each hospital's held-out local test set (never seen during training)."),
      h2("3.4 Federated association analysis (statistical meta-analysis)"),
      p("For \u201cassociation with all codes,\u201d pooling raw records would break the federated boundary. Instead, each hospital fit its own univariate logistic regression of the outcome on each code locally, producing a local coefficient and standard error. These were combined coordinator-side using fixed-effect inverse-variance meta-analysis \u2014 the standard statistical technique for combining estimates from independently-held (horizontally partitioned) datasets:"),
      p("\u03b2\u209a\u2092\u2092\u2097\u2091\u2091 = \u03a3(w\u2096\u03b2\u2096) / \u03a3(w\u2096),  where w\u2096 = 1/SE\u2096\u00b2", { italics: true }),
      p("Results are reported as odds ratios (OR) with 95% confidence intervals."),

      new Paragraph({ children: [new PageBreak()] }),

      // ---- Results ----
      h1("4. Results"),
      h2("4.1 FedAvg convergence"),
      img(path.join(CHARTS_DIR, "01_fedavg_convergence.png"), 560, 210),
      caption("Figure 1. Training loss per communication round. The bold black line is the global (federated) model; dashed lines are each hospital's local loss under that round's global weights. Both tasks converge within ~6\u20138 rounds."),

      h2("4.2 Federated vs. local-only model performance"),
      table(
        ["Hospital", "Test N", "Accuracy", "AUC (federated)", "AUC (local-only)"],
        perfRows(data.dep_eval, data.dep_local),
        [2200, 1500, 1800, 1900, 1900]
      ),
      caption("Table 2. Depression model (PHQ-9 \u2265 10) \u2014 federated vs. local-only baseline."),
      table(
        ["Hospital", "Test N", "Accuracy", "AUC (federated)", "AUC (local-only)"],
        perfRows(data.dia_eval, data.dia_local),
        [2200, 1500, 1800, 1900, 1900]
      ),
      caption("Table 3. Diabetes model (HbA1c \u2265 6.5%) \u2014 federated vs. local-only baseline."),
      img(path.join(CHARTS_DIR, "05_federated_vs_local.png"), 560, 200),
      caption("Figure 2. AUC comparison: federation achieves performance close to local-only training while never sharing raw data across hospitals."),

      h2("4.3 Prevalence trends across survey cycles"),
      img(path.join(CHARTS_DIR, "02_prevalence_by_era.png"), 480, 300),
      caption("Figure 3. Depression prevalence rose across cycles; diabetes prevalence rose then eased in the most recent cycle."),

      new Paragraph({ children: [new PageBreak()] }),
      h2("4.4 Association of all codes with Diabetes (HbA1c \u2265 6.5%)"),
      p("Federated meta-analysis of each code's univariate association with diabetes status, combining all three hospitals' local estimates."),
      img(path.join(CHARTS_DIR, "03_forest_diabetes.png"), 560, 550),
      caption("Figure 4. Forest plot \u2014 odds ratios (95% CI) for diabetes, all codes, sorted by effect size."),
      table(
        ["Code", "Odds Ratio (95% CI)", "p-value"],
        assocRows(assocDia),
        [4200, 2600, 1600]
      ),
      caption("Table 4. Full federated association results \u2014 diabetes outcome."),

      new Paragraph({ children: [new PageBreak()] }),
      h2("4.5 Association of all codes with Depression (PHQ-9 \u2265 10)"),
      img(path.join(CHARTS_DIR, "04_forest_depression.png"), 560, 550),
      caption("Figure 5. Forest plot \u2014 odds ratios (95% CI) for depression, all codes, sorted by effect size."),
      table(
        ["Code", "Odds Ratio (95% CI)", "p-value"],
        assocRows(assocDep),
        [4200, 2600, 1600]
      ),
      caption("Table 5. Full federated association results \u2014 depression outcome."),

      new Paragraph({ children: [new PageBreak()] }),
      h2("4.6 Depression \u2194 Diabetes cross-link"),
      img(path.join(CHARTS_DIR, "06_depression_diabetes_link.png"), 460, 280),
      caption("Figure 6. Federated, meta-analyzed cross-association between the two conditions, adjusted for nothing else (unadjusted univariate link, replicated independently in all 3 hospitals before pooling)."),
      p(`Per one-point increase in total PHQ-9 score, the odds of having HbA1c \u2265 6.5% increase by a factor of ${data.link.PHQ9_Total_predicting_diabetes_binary.OR.toFixed(3)} (95% CI ${data.link.PHQ9_Total_predicting_diabetes_binary.CI_low.toFixed(3)}\u2013${data.link.PHQ9_Total_predicting_diabetes_binary.CI_high.toFixed(3)}, p ${fmtP(data.link.PHQ9_Total_predicting_diabetes_binary.p_value)}).`),
      p(`Per one-point increase in HbA1c (%), the odds of PHQ-9 \u2265 10 increase by a factor of ${data.link.LBXGH_predicting_depression_binary.OR.toFixed(3)} (95% CI ${data.link.LBXGH_predicting_depression_binary.CI_low.toFixed(3)}\u2013${data.link.LBXGH_predicting_depression_binary.CI_high.toFixed(3)}, p ${fmtP(data.link.LBXGH_predicting_depression_binary.p_value)}).`),

      new Paragraph({ children: [new PageBreak()] }),
      h1("5. Adjusted (Multivariable) Association Analysis"),
      p("The associations in Section 4 are unadjusted (univariate) \u2014 each code was tested against the outcome on its own. Here, each hospital instead fits a multivariable logistic regression locally, adjusting the depression\u2013diabetes cross-link for gender, age, education, income-to-poverty ratio, and BMI simultaneously. As before, only each hospital's local coefficients (\u201cLocal \u03b2\u201d, the log-odds effect estimated purely from that hospital's own records) and standard errors are shared; the coordinator combines them into one pooled estimate via the same fixed-effect inverse-variance meta-analysis used in Section 3.4. No raw patient record ever left its hospital for this adjustment either."),

      h2("5.1 PHQ-9 total score \u2192 diabetes, adjusted"),
      table(
        ["Hospital (cohort)", "Local \u03b2", "Local OR"],
        CLIENTS.map(c => {
          const row = adjDia.find(r => r.code === "PHQ9_Total");
          const b = row[`${c}_beta`];
          return [ERA[c], b.toFixed(4), Math.exp(b).toFixed(4)];
        }).concat([[
          "Federated pooled (meta-analysis)",
          adjDia.find(r => r.code === "PHQ9_Total").beta.toFixed(4),
          fmtOR(adjDia.find(r => r.code === "PHQ9_Total")) + `, p ${fmtP(adjDia.find(r => r.code === "PHQ9_Total").p_value)}`
        ]]),
        [3200, 2400, 4400]
      ),
      caption("Table 6. PHQ-9 total predicting diabetes, adjusted for gender, age, education, income-to-poverty ratio, and BMI. Unadjusted OR was " + fmtOR(data.link.PHQ9_Total_predicting_diabetes_binary ? {OR: data.link.PHQ9_Total_predicting_diabetes_binary.OR, CI_low: data.link.PHQ9_Total_predicting_diabetes_binary.CI_low, CI_high: data.link.PHQ9_Total_predicting_diabetes_binary.CI_high} : {}) + " (Section 4.6); adjustment attenuates but does not remove the association."),

      h2("5.2 HbA1c \u2192 depression, adjusted"),
      table(
        ["Hospital (cohort)", "Local \u03b2", "Local OR"],
        CLIENTS.map(c => {
          const row = adjDep.find(r => r.code === "LBXGH");
          const b = row[`${c}_beta`];
          return [ERA[c], b.toFixed(4), Math.exp(b).toFixed(4)];
        }).concat([[
          "Federated pooled (meta-analysis)",
          adjDep.find(r => r.code === "LBXGH").beta.toFixed(4),
          fmtOR(adjDep.find(r => r.code === "LBXGH")) + `, p ${fmtP(adjDep.find(r => r.code === "LBXGH").p_value)}`
        ]]),
        [3200, 2400, 4400]
      ),
      caption("Table 7. HbA1c predicting depression, adjusted for gender, age, education, income-to-poverty ratio, and BMI. Unadjusted OR was " + fmtOR({OR: data.link.LBXGH_predicting_depression_binary.OR, CI_low: data.link.LBXGH_predicting_depression_binary.CI_low, CI_high: data.link.LBXGH_predicting_depression_binary.CI_high}) + " (Section 4.6); adjustment attenuates but does not remove the association."),

      img(path.join(CHARTS_DIR, "09_unadjusted_vs_adjusted.png"), 480, 260),
      caption("Figure 7. Direct comparison of the depression\u2194diabetes link before and after adjustment. Both links shrink toward 1.0 but remain on the same side of it."),

      h2("5.3 Full adjusted models \u2014 every covariate's own effect"),
      img(path.join(CHARTS_DIR, "07_forest_adjusted_diabetes.png"), 560, 300),
      caption("Figure 8. Adjusted odds ratios for diabetes. Orange = the key predictor (PHQ-9 total); blue = the adjustment covariates."),
      table(
        ["Code", "Adjusted OR (95% CI)", "p-value"],
        adjDia.filter(r => r.code !== "const").sort((a,b) => b.OR - a.OR).map(r => [r.label + " (" + r.code + ")", fmtOR(r), fmtP(r.p_value)]),
        [4200, 2600, 1600]
      ),
      caption("Table 8. Full adjusted diabetes model, all covariates."),

      img(path.join(CHARTS_DIR, "08_forest_adjusted_depression.png"), 560, 300),
      caption("Figure 9. Adjusted odds ratios for depression. Orange = the key predictor (HbA1c); blue = the adjustment covariates."),
      table(
        ["Code", "Adjusted OR (95% CI)", "p-value"],
        adjDep.filter(r => r.code !== "const").sort((a,b) => b.OR - a.OR).map(r => [r.label + " (" + r.code + ")", fmtOR(r), fmtP(r.p_value)]),
        [4200, 2600, 1600]
      ),
      caption("Table 9. Full adjusted depression model, all covariates."),

      h2("5.4 Interpretation"),
      bullet("Both cross-links remain statistically significant after adjustment \u2014 the association is not simply an artifact of age, sex, education, income, or BMI."),
      bullet("Both links attenuate: PHQ-9\u2192diabetes OR falls from 1.032 (unadjusted) to 1.025 (adjusted); HbA1c\u2192depression OR falls from 1.123 to 1.086. The gap is the share of the raw correlation that age/BMI/socioeconomic status were accounting for."),
      bullet("BMI and age are the strongest adjustment covariates for diabetes risk; gender and income-to-poverty ratio are the strongest for depression risk \u2014 consistent with well-established epidemiological patterns and a useful sanity check on model validity."),
      bullet("The HbA1c\u2192depression local \u03b2 is less stable across hospitals (0.050 \u2192 0.092 \u2192 0.099) than PHQ-9\u2192diabetes (0.017 \u2192 0.025 \u2192 0.033), so the pooled adjusted estimate for that direction should be treated with slightly more caution."),

      // ---- Discussion ----
      h1("6. Discussion & Final Findings"),
      bullet("Feasibility of federation: Both FedAvg models converged smoothly and lost only 1\u20132 AUC points versus local-only training. This shows a federated approach is a viable substitute for pooling these NHANES cycles when centralization is undesirable."),
      bullet("Bidirectional depression\u2013diabetes link: The association holds in both directions and is highly significant (p < 10\u207b\u00b9\u2070 in both), consistent with the broader epidemiological literature on depression as both a risk factor for, and consequence of, metabolic/glycemic dysregulation."),
      bullet("Somatic symptoms dominate the diabetes link: Sleep disturbance (DPQ030), anhedonia (DPQ010), fatigue (DPQ040), and psychomotor change (DPQ080) show the strongest links to elevated HbA1c \u2014 plausibly reflecting shared physiological pathways (inflammation, sleep, activity level) rather than purely psychological ones."),
      bullet("Socioeconomic gradient: Higher education and higher income-to-poverty ratio are protective for both depression and diabetes (OR < 1), while higher BMI and age increase risk of both \u2014 a consistent, expected pattern that also serves as a sanity check on data quality."),
      bullet("Temporal trend: Rising depression prevalence across cycles (9.0% \u2192 13.1%) aligns with nationally reported increases in depressive symptoms in the U.S. adult population over this period; the diabetes prevalence dip in 2021\u20132023 should be interpreted cautiously given pandemic-era changes to NHANES data collection and reduced/partial-response sampling in that cycle."),
      bullet("Adjustment matters but doesn't erase the link: controlling for gender, age, education, income, and BMI attenuates both directions of the depression\u2013diabetes association (PHQ-9\u2192diabetes: 1.032\u21921.025; HbA1c\u2192depression: 1.123\u21921.086) but both remain statistically significant (Section 5) \u2014 the relationship is not fully explained by shared demographic or metabolic confounders."),

      h1("7. Limitations"),
      bullet("The adjusted models (Section 5) control only for the covariates present in this dataset (gender, age, education, income, BMI); unmeasured confounders such as medication use, sleep quality, inflammation markers, or physical activity are not available here and could further attenuate the association."),
      bullet("PHQ-9 and HbA1c missingness differs by cycle (notably higher partial/non-response in 2021\u20132023), which can bias prevalence comparisons across eras."),
      bullet("Cross-sectional design: associations do not establish causality or temporal direction."),
      bullet("Logistic regression (rather than a deeper network) was used for the FedAvg model for interpretability and stability with 15 communication rounds; results should be treated as a baseline federated benchmark."),

      h1("8. Conclusion"),
      p("A fully federated pipeline \u2014 local data preparation, FedAvg model training, and meta-analytic association testing (both univariate and multivariable) \u2014 was applied across three independently-held NHANES cycles without ever pooling raw records. The federated models achieved performance comparable to non-federated local baselines, and the federated association analysis confirms a robust, statistically significant, bidirectional relationship between depression symptoms and glycemic control (HbA1c) that survives adjustment for demographics and BMI, replicated independently across all three hospital datasets."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const OUT_PATH = path.join(OUTPUT_DIR, "Federated_Depression_Diabetes_Report.docx");
  fs.writeFileSync(OUT_PATH, buf);
  console.log("Saved to:", OUT_PATH);
  console.log("Report written.");
});
