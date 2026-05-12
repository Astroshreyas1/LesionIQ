import type { CaseRecord } from "../types/lesioniq";

const REPORT_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

function confidenceLevel(confidence: number): string {
  if (confidence >= 0.75) return "high";
  if (confidence >= 0.5) return "moderate";
  return "low";
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function predictedThreshold(caseRecord: CaseRecord): number {
  return (
    caseRecord.predictionScores.find((score) => score.classCode === caseRecord.predictedClassCode)?.threshold ??
    0.5
  );
}

function closestAlternatives(caseRecord: CaseRecord): string {
  const alternatives = caseRecord.predictionScores
    .filter((score) => score.classCode !== caseRecord.predictedClassCode)
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 2)
    .map((score) => `${score.classCode} (${pct(score.probability)})`);

  if (alternatives.length === 0) return "No close alternate class probabilities were returned.";
  if (alternatives.length === 1) return alternatives[0];
  return `${alternatives[0]} and ${alternatives[1]}`;
}

function recommendationPriority(caseRecord: CaseRecord): string {
  if (caseRecord.urgency === "High concern") return "HIGH — requires immediate dermatologist review";
  if (caseRecord.urgency === "Expedited review") return "ELEVATED — prioritize clinician review";
  return "ROUTINE — clinician review still required";
}

function sanitizeSlmText(summary: string): string {
  return summary
    .replace(/\r\n/g, "\n")
    .replace(/\s+/g, " ")
    .replace(new RegExp(REPORT_DIVIDER, "g"), "")
    .replace(/LESIONIQ CLINICAL EXPLAINABILITY REPORT/g, "")
    .replace(/\bPREDICTION\b/g, "")
    .replace(/\bEVIDENCE\b/g, "")
    .replace(/\bREASONING\b/g, "")
    .replace(/\bDiagnosis\s*:[^.]*/gi, "")
    .replace(/\bConfidence\s*:[^.]*/gi, "")
    .replace(/\bThreshold\s*:[^.]*/gi, "")
    .trim();
}

function reasoningLines(caseRecord: CaseRecord): string[] {
  const slmReasoning = sanitizeSlmText(caseRecord.explainability.slmSummary);
  const lines = [
    slmReasoning ||
      `The model identified this lesion as ${caseRecord.predictedClassLabel} with ${confidenceLevel(caseRecord.calibratedConfidence)} confidence.`,
    `Primary visual evidence: ${caseRecord.explainability.gradcamSummary}`,
    `Swin Transformer attention: ${caseRecord.explainability.attentionSummary}`,
    caseRecord.recommendation
  ];

  return lines.filter(Boolean);
}

export function buildExplainabilityReport(caseRecord: CaseRecord): string {
  const threshold = predictedThreshold(caseRecord);
  const metadata = caseRecord.metadata;
  const reasoning = reasoningLines(caseRecord).map((line) => `  ${line}`).join("\n\n");

  return `${REPORT_DIVIDER}
LESIONIQ CLINICAL EXPLAINABILITY REPORT
${REPORT_DIVIDER}

PREDICTION
  Diagnosis   : ${caseRecord.predictedClassLabel} (${caseRecord.predictedClassCode})
  Confidence  : ${pct(caseRecord.calibratedConfidence)} (${confidenceLevel(caseRecord.calibratedConfidence)} confidence)
  Threshold   : ${threshold.toFixed(2)} (tuned — default 0.50)

REASONING
${reasoning}

FLAGGED REGION
  ⚑ Location  : Model-highlighted lesion evidence region
  ⚑ Feature   : ${caseRecord.explainability.gradcamSummary}
  ⚑ Area      : See Grad-CAM and attention artifacts for spatial coverage
  ⚑ Pixel     : See diagnosis.json for peak activation if generated

DIFFERENTIAL DIAGNOSIS
  Closest alternatives: ${closestAlternatives(caseRecord)}
  (Consider these if clinical presentation is ambiguous)

URGENCY & ACTION
  Priority  : ${recommendationPriority(caseRecord)}
  Recommend : ${caseRecord.recommendation}

METADATA
  Age       : ${metadata.ageYears || "Unknown"}
  Sex       : ${metadata.sex}
  Site      : ${metadata.anatomicalSite}

${REPORT_DIVIDER}
DISCLAIMER: AI decision-support tool only.
Final diagnosis must be made by a qualified clinician.
${REPORT_DIVIDER}`;
}

export function downloadExplainabilityReport(caseRecord: CaseRecord): void {
  const report = buildExplainabilityReport(caseRecord);
  const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${caseRecord.caseId || caseRecord.id}-explainability-report.txt`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
