import type { WorkMotionResult } from "./custom-motion-training";
import type { VisualActionInterval } from "./local-action-context";
import type { ActionReviewCandidate, ObservationQualityReason } from "./action-review";
import { workContextTaskType } from "./work-context-inference";

export type WorkLabelDecision = {
  candidates: ActionReviewCandidate[];
  qualityReasons: ObservationQualityReason[];
  source: "personal_sequence" | "visual_context" | "insufficient";
  independentSupport: boolean;
  autoLearningEligible: boolean;
};

export function canAutomaticallyLearnWork(decision: WorkLabelDecision | null, stage: "training" | "evaluation" | "learning" | "analysis", blocked: boolean) {
  return Boolean(decision?.autoLearningEligible && !blocked && (stage === "training" || stage === "learning"));
}

/** A boundary proposal and a zone hint are not evidence of a work label. */
export function decideWorkLabel(motion: WorkMotionResult, registeredLabels: string[], visual?: VisualActionInterval): WorkLabelDecision {
  const candidates = motion.candidates.filter(c => registeredLabels.includes(c.label)).map(c => ({
    taskType: workContextTaskType(c.label), taskLabel: c.label,
    confidence: Math.exp(-c.distance * 4.2) * motion.observedQuality,
  }));
  const usableVisual = visual?.kind === "task" && registeredLabels.includes(visual.label) ? visual : undefined;
  const qualityReasons: ObservationQualityReason[] = [];
  // Independent image evidence can enrol missing classes or resolve an uncertain pose match.
  if (usableVisual && (motion.status !== "matched" || !motion.independentSupport)) return {
    candidates: [{ taskType: workContextTaskType(usableVisual.label), taskLabel: usableVisual.label, confidence: usableVisual.similarity }],
    qualityReasons, source: "visual_context", independentSupport: true, autoLearningEligible: true,
  };
  if (candidates.length) {
    if (motion.status !== "matched") qualityReasons.push("unknown_motion");
    if (!motion.independentSupport) qualityReasons.push("unverified_reference");
    if (usableVisual && candidates[0].taskLabel !== usableVisual.label) qualityReasons.push("unknown_motion");
    return { candidates, qualityReasons, source: "personal_sequence", independentSupport: motion.independentSupport,
      autoLearningEligible: !qualityReasons.length && motion.independentSupport && usableVisual?.label === candidates[0].taskLabel };
  }
  if (usableVisual) return {
    candidates: [{ taskType: workContextTaskType(usableVisual.label), taskLabel: usableVisual.label, confidence: usableVisual.similarity }],
    qualityReasons, source: "visual_context", independentSupport: true, autoLearningEligible: true,
  };
  return { candidates: [], qualityReasons: ["unknown_motion"], source: "insufficient", independentSupport: false, autoLearningEligible: false };
}
