import type { ObservationEpisode } from "./observation-engine";

export type ActionReviewStatus = "pending" | "confirmed" | "unresolved";

export type ActionReviewCandidate = {
  taskType: string;
  taskLabel: string;
  confidence: number;
};

export type ObservationQualityReason =
  | "insufficient_frames"
  | "poor_tracking"
  | "missing_zone"
  | "missing_time_context"
  | "outside_work_context"
  | "unknown_motion";

export type ActionAmbiguityReason =
  | "no_candidates"
  | "low_confidence"
  | "close_candidates"
  | ObservationQualityReason;

export type ManualActionLabel = {
  taskType: string;
  taskLabel: string;
  reviewedAt: number;
  reviewer?: string;
  note?: string;
};

export type ActionReviewMetadata = {
  status: ActionReviewStatus;
  candidates: ActionReviewCandidate[];
  reasons: ActionAmbiguityReason[];
  topCandidateMargin: number | null;
  manualLabel: ManualActionLabel | null;
  createdAt: number;
  updatedAt: number;
};

export type ReviewableObservationEpisode = ObservationEpisode & {
  actionReview?: ActionReviewMetadata;
};

export type AmbiguityThresholds = {
  minimumConfidence: number;
  minimumCandidateMargin: number;
};

export type ActionAmbiguityAssessment = {
  ambiguous: boolean;
  reasons: ActionAmbiguityReason[];
  candidates: ActionReviewCandidate[];
  topCandidateMargin: number | null;
};

export const DEFAULT_AMBIGUITY_THRESHOLDS: AmbiguityThresholds = {
  minimumConfidence: 0.6,
  minimumCandidateMargin: 0.12,
};

export const ACTION_REVIEW_STATUS_LABELS: Record<ActionReviewStatus, string> = {
  pending: "검토 대기",
  confirmed: "라벨 확정",
  unresolved: "판별 불가",
};

export const ACTION_AMBIGUITY_REASON_LABELS: Record<ActionAmbiguityReason, string> = {
  no_candidates: "업무 후보 없음",
  low_confidence: "낮은 분류 신뢰도",
  close_candidates: "상위 후보 점수 차이 부족",
  insufficient_frames: "좌표 프레임 부족",
  poor_tracking: "스켈레톤 추적 품질 부족",
  missing_zone: "매장 구역 정보 없음",
  missing_time_context: "시간대 업무 정보 없음",
  outside_work_context: "등록된 업무 맥락과 불일치",
  unknown_motion: "학습되지 않은 동작",
};

function clampConfidence(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export function normalizeActionCandidates(
  candidates: ActionReviewCandidate[],
): ActionReviewCandidate[] {
  const unique = new Map<string, ActionReviewCandidate>();
  candidates.forEach((candidate) => {
    if (!candidate.taskType || !candidate.taskLabel) return;
    const normalized = { ...candidate, confidence: clampConfidence(candidate.confidence) };
    const current = unique.get(candidate.taskType);
    if (!current || normalized.confidence > current.confidence) {
      unique.set(candidate.taskType, normalized);
    }
  });
  return [...unique.values()].sort((a, b) => b.confidence - a.confidence);
}

export function assessActionAmbiguity(args: {
  confidence: number;
  candidates: ActionReviewCandidate[];
  qualityReasons?: ObservationQualityReason[];
  thresholds?: Partial<AmbiguityThresholds>;
}): ActionAmbiguityAssessment {
  const thresholds = { ...DEFAULT_AMBIGUITY_THRESHOLDS, ...args.thresholds };
  const candidates = normalizeActionCandidates(args.candidates);
  const topCandidateMargin = candidates.length >= 2
    ? candidates[0].confidence - candidates[1].confidence
    : null;
  const reasons: ActionAmbiguityReason[] = [];

  if (candidates.length === 0) reasons.push("no_candidates");
  if (clampConfidence(args.confidence) < thresholds.minimumConfidence) {
    reasons.push("low_confidence");
  }
  if (topCandidateMargin !== null && topCandidateMargin < thresholds.minimumCandidateMargin) {
    reasons.push("close_candidates");
  }
  (args.qualityReasons ?? []).forEach((reason) => {
    if (!reasons.includes(reason)) reasons.push(reason);
  });

  return {
    ambiguous: reasons.length > 0,
    reasons,
    candidates,
    topCandidateMargin,
  };
}

export function createActionReview(
  episode: ObservationEpisode,
  options: {
    candidates?: ActionReviewCandidate[];
    qualityReasons?: ObservationQualityReason[];
    thresholds?: Partial<AmbiguityThresholds>;
    createdAt?: number;
  } = {},
): ActionReviewMetadata | null {
  const candidates = options.candidates ?? [{
    taskType: episode.taskType,
    taskLabel: episode.taskLabel,
    confidence: episode.taskConfidence,
  }];
  const assessment = assessActionAmbiguity({
    confidence: episode.taskConfidence,
    candidates,
    qualityReasons: options.qualityReasons,
    thresholds: options.thresholds,
  });
  if (!assessment.ambiguous) return null;
  const now = options.createdAt ?? Date.now();
  return {
    status: "pending",
    candidates: assessment.candidates,
    reasons: assessment.reasons,
    topCandidateMargin: assessment.topCandidateMargin,
    manualLabel: null,
    createdAt: now,
    updatedAt: now,
  };
}

export function applyManualActionLabel(
  episode: ReviewableObservationEpisode,
  label: Omit<ManualActionLabel, "reviewedAt"> & { reviewedAt?: number },
): ReviewableObservationEpisode {
  const reviewedAt = label.reviewedAt ?? Date.now();
  const manualLabel: ManualActionLabel = { ...label, reviewedAt };
  const currentReview = episode.actionReview ?? {
    status: "pending" as const,
    candidates: [{
      taskType: episode.taskType,
      taskLabel: episode.taskLabel,
      confidence: episode.taskConfidence,
    }],
    reasons: ["low_confidence" as const],
    topCandidateMargin: null,
    manualLabel: null,
    createdAt: reviewedAt,
    updatedAt: reviewedAt,
  };

  return {
    ...episode,
    taskType: manualLabel.taskType,
    taskLabel: manualLabel.taskLabel,
    taskConfidence: 1,
    disposition: episode.mode === "learning" ? "accepted" : "analysis_only",
    dispositionReason: "개발자가 스켈레톤 리플레이를 확인하고 업무 라벨을 확정했어요.",
    actionReview: {
      ...currentReview,
      status: "confirmed",
      manualLabel,
      updatedAt: reviewedAt,
    },
  };
}

export function markActionReviewUnresolved(
  episode: ReviewableObservationEpisode,
  reviewedAt = Date.now(),
): ReviewableObservationEpisode {
  if (!episode.actionReview) return episode;
  return {
    ...episode,
    disposition: "excluded",
    dispositionReason: "개발자 검토에서도 업무를 판별할 수 없어 학습에서 제외했어요.",
    actionReview: {
      ...episode.actionReview,
      status: "unresolved",
      manualLabel: null,
      updatedAt: reviewedAt,
    },
  };
}

export function needsActionReview(episode: ReviewableObservationEpisode) {
  return episode.actionReview?.status === "pending";
}

export function listPendingActionReviews(
  episodes: ReviewableObservationEpisode[],
): ReviewableObservationEpisode[] {
  return episodes
    .filter(needsActionReview)
    .sort((a, b) => a.recordedAt - b.recordedAt);
}

