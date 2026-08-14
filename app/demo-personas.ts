// Demo/simulation personas for the "케어 기록" tab.
//
// Why this exists: a brand-new user can't realistically let the app watch
// them for a full week before they see what a care report looks like. These
// personas exist so someone can preview the feature immediately. Each
// persona is a synthetic week of daily activity, converted into the same
// `DailyLog` shape real usage produces and run through the *real*
// `computeBaseline` / `detectChangeSignal` pipeline from care-metrics.ts -
// so the "watch" / "notable" / "none" result and reasons shown here are
// genuinely computed, not hand-written copy pretending to be computed.
//
// Four personas cover four outcomes a real user could see, on purpose:
//   1. stable       -> "none"    (most weeks look like this - no alarm)
//   2. decline      -> "notable" (several indicators rise together)
//   3. busy-season   -> "notable"/"watch" + a confound note (numbers look
//                       concerning, but the period was mostly self-tagged
//                       "busy", so the report says so instead of alarming)
//   4. single-spike  -> "watch"  (one noisy day, not a persistent pattern -
//                       the system doesn't overreact to a single outlier)

import {
  computeBaseline,
  detectChangeSignal,
  emptyDailyLog,
  SLOW_DELAY_SECONDS,
  type BusyLevel,
  type ChangeSignal,
  type DailyLog,
  type MetricSummary,
} from "./care-metrics";
import type { DemoMotionType } from "./demo-motion";
import {
  detectMotionEvents,
  motionSamplesFromRawFrames,
  type DetectedMotionEvent,
} from "./motion-detection";

export type DemoEvent = {
  /** Display text, e.g. "19:14 출입문 상태 재확인". */
  label: string;
  /** Which synthetic skeleton clip (app/demo-motion.ts) represents this event. */
  motionType: DemoMotionType;
};

export type DemoDay = {
  day: string;
  date: string;
  safetyAlerts: number;
  doubleChecks: number;
  unfinishedTasks: number;
  /** 0-100, matches the "미세 지연" % shown per day. */
  microDelayRate: number;
  busy: boolean;
  note: string;
  examples: DemoEvent[];
};

export type DemoInsight = {
  icon: string;
  kicker: string;
  title: string;
  body: string;
};

type DemoPersonaDefinition = {
  id: string;
  name: string;
  avatarLabel: string;
  tagline: string;
  summary: string;
  weekRange: string;
  week: DemoDay[];
  insights: DemoInsight[];
  note: string;
};

export type DemoPersona = DemoPersonaDefinition & {
  /** Computed by the real care-metrics pipeline from `week`, not authored. */
  signal: ChangeSignal;
  baselineDayCount: number;
  /** The personal baseline `signal` was compared against (null if too little history). */
  baseline: (MetricSummary & { dayCount: number }) | null;
};

// How many discrete tasks a day's micro-delay % is approximated over when
// converting a display percentage back into raw delay samples. Only affects
// the internal signal computation, never what's shown on screen.
const TASKS_PER_DAY = 6;
const SLOW_DELAY_SAMPLE_SECONDS = 150; // >= the 120s "slow" threshold
const FAST_DELAY_SAMPLE_SECONDS = 30;
const BASELINE_DAY_COUNT = 5;

function demoDayToLog(personaId: string, day: DemoDay, index: number): DailyLog {
  const slowSamples = Math.round((day.microDelayRate / 100) * TASKS_PER_DAY);
  const fastSamples = Math.max(TASKS_PER_DAY - slowSamples, 0);
  const busyLevel: BusyLevel = day.busy ? "busy" : "normal";
  return {
    ...emptyDailyLog(`${personaId}-2000-01-${String(index + 1).padStart(2, "0")}`),
    safetyAlerts: day.safetyAlerts,
    doubleChecks: day.doubleChecks,
    tasksStarted: TASKS_PER_DAY,
    tasksCompleted: Math.max(TASKS_PER_DAY - day.unfinishedTasks, 0),
    microDelaySeconds: [
      ...Array(slowSamples).fill(SLOW_DELAY_SAMPLE_SECONDS),
      ...Array(fastSamples).fill(FAST_DELAY_SAMPLE_SECONDS),
    ],
    busyLevel,
  };
}

const PERSONA_DEFINITIONS: DemoPersonaDefinition[] = [
  // -- Persona: 안정적인 한 주 (none) --------------------------------------
  {
    id: "stable",
    name: "박준혁 사장님",
    avatarLabel: "박",
    tagline: "48세 남성 · 동네 편의점 3년차",
    summary: "혼자 편의점을 안정적으로 운영합니다. 이번 주는 안전 알림, 반복 확인, 지연 모두 평소 범위 안에서 유지되는 가장 흔한 한 주를 가정했습니다.",
    weekRange: "8월 4일–10일",
    week: [
      { day: "월", date: "8/4", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "평소와 같은 오픈·마감 흐름", examples: [{ label: "07:58 매장 오픈", motionType: "normal_task" }, { label: "12:30 진열대 정리", motionType: "high_reach" }, { label: "19:03 마감 확인 1회", motionType: "normal_task" }] },
      { day: "화", date: "8/5", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "택배 정리 후 평소처럼 마감", examples: [{ label: "13:20 택배 수령 처리", motionType: "normal_task" }, { label: "16:45 냉장고 온도 확인", motionType: "double_check" }, { label: "19:05 마감 확인 1회", motionType: "normal_task" }] },
      { day: "수", date: "8/6", safetyAlerts: 1, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 8, busy: false, note: "온열기 점검 알림 1회, 바로 정상 종료", examples: [{ label: "16:40 온열기 점검 알림 → 즉시 해제", motionType: "safety_alert" }, { label: "18:10 진열대 정리", motionType: "high_reach" }, { label: "19:01 마감 확인 1회", motionType: "normal_task" }] },
      { day: "목", date: "8/7", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 11, busy: false, note: "조용한 하루, 반복 확인 없이 마감", examples: [{ label: "11:20 재고 확인", motionType: "normal_task" }, { label: "19:10 마감 한 번에 완료", motionType: "normal_task" }] },
      { day: "금", date: "8/8", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "평소와 같은 금요일 마감", examples: [{ label: "14:00 택배 수령 처리", motionType: "normal_task" }, { label: "19:06 출입문 상태 재확인", motionType: "double_check" }] },
      { day: "토", date: "8/9", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "주말에도 평소와 비슷한 흐름", examples: [{ label: "10:15 진열대 정리", motionType: "high_reach" }, { label: "19:12 마감 확인 1회", motionType: "normal_task" }] },
      { day: "일", date: "8/10", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "휴무 준비까지 평소대로 마무리", examples: [{ label: "15:30 재고 확인", motionType: "normal_task" }, { label: "18:50 마감 확인 1회", motionType: "normal_task" }] },
    ],
    insights: [
      { icon: "≈", kicker: "1. 평소 대비 변화", title: "이번 주는 평소 범위 안이에요", body: "안전 알림, 반복 확인, 미세 지연 모두 지난 흐름과 비슷합니다. 메모리 가드는 대부분의 주가 이렇게 조용히 지나간다는 것도 함께 보여드리고 싶었어요." },
      { icon: "⌁", kicker: "2. 시간축 연결", title: "특별히 몰린 시간대가 없어요", body: "타임라인에도 특정 시간대에 여러 사건이 겹치는 구간이 보이지 않습니다. 알림이 뜨지 않는 주가 오히려 더 흔한 모습이에요." },
      { icon: "♡", kicker: "3. 케어로 연결", title: "이럴 땐 조용히 지나가요", body: "변화 신호가 없을 때는 케어 대화를 권하지 않습니다. 메모리 가드는 변화가 실제로 감지될 때만 조심스럽게 말을 건네요." },
    ],
    note: "박준혁 사장님의 데이터는 설명을 위한 가상 시나리오입니다. 대부분의 주는 이렇게 평소와 비슷하게 지나간다는 것을 보여주기 위한 예시예요.",
  },

  // -- Persona: 점진적인 변화 (notable) -------------------------------------
  {
    id: "decline",
    name: "이수진 사장님",
    avatarLabel: "이",
    tagline: "54세 여성 · 개인 카페 사장 5년차",
    summary: "평소에는 혼자 카페를 안정적으로 운영합니다. 이번 주 후반에는 깜빡함, 마감 반복 확인, 업무 중단, 미세 지연이 함께 늘어나는 패턴을 가정했습니다.",
    weekRange: "8월 4일–10일",
    week: [
      { day: "월", date: "8/4", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 8, busy: false, note: "평소와 비슷한 월요일 오픈·마감 흐름", examples: [{ label: "08:41 매장 오픈", motionType: "normal_task" }, { label: "12:15 원두 재고 정리", motionType: "high_reach" }, { label: "19:07 스마트 마감 1회 완료", motionType: "normal_task" }] },
      { day: "화", date: "8/5", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "마감 뒤 출입문 상태를 한 번 더 확인", examples: [{ label: "19:12 출입문 잠김 확인", motionType: "normal_task" }, { label: "19:14 출입문 상태 재확인", motionType: "double_check" }] },
      { day: "수", date: "8/6", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "무리 없이 하루 마감", examples: [{ label: "15:20 원두 포장 정리", motionType: "high_reach" }, { label: "19:03 마감 확인 1회", motionType: "normal_task" }] },
      { day: "목", date: "8/7", safetyAlerts: 1, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "온열기 차단 알림 1회, 곧바로 정상 마감", examples: [{ label: "14:10 손님 대기열 증가", motionType: "queue_shift" }, { label: "19:09 온열기 차단 권고 → 즉시 조치", motionType: "safety_alert" }] },
      { day: "금", date: "8/8", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 11, busy: false, note: "평소와 비슷한 금요일", examples: [{ label: "13:00 주문 대기열 증가", motionType: "queue_shift" }, { label: "19:06 출입문 상태 재확인", motionType: "double_check" }] },
      { day: "토", date: "8/9", safetyAlerts: 2, doubleChecks: 2, unfinishedTasks: 2, microDelayRate: 45, busy: false, note: "마감 반복 확인과 미완료 주문이 크게 늘어난 토요일", examples: [{ label: "15:02 주문 입력 후 9분 지연", motionType: "micro_delay" }, { label: "16:20 카드 결제 재입력", motionType: "register_tap" }, { label: "17:52 재고 박스 하단 확인", motionType: "low_bend" }, { label: "19:28 스마트 플러그 차단 권고 후 재확인", motionType: "safety_alert" }] },
      { day: "일", date: "8/10", safetyAlerts: 1, doubleChecks: 2, unfinishedTasks: 1, microDelayRate: 40, busy: false, note: "쉬는 날 없이 비슷한 흐름이 이어짐", examples: [{ label: "11:47 카드 결제 재입력", motionType: "register_tap" }, { label: "14:30 대기 손님 늘어남", motionType: "queue_shift" }, { label: "20:11 마감 항목 2회 재확인", motionType: "double_check" }] },
    ],
    insights: [
      { icon: "↗", kicker: "1. 평소 대비 변화", title: "주 후반에 신호가 겹쳐요", body: "토·일 이틀 동안 안전 알림, 반복 확인, 미완료 업무, 미세 지연이 함께 늘었습니다. 한 가지 실수보다 여러 지표가 같은 기간에 함께 움직이는지를 봅니다." },
      { icon: "⌁", kicker: "2. 시간축 연결", title: "바쁜 시간대의 흐름을 확인해요", body: "토요일 오후에는 주문 입력 중단과 스마트 플러그 차단 권고가 가까운 시간대에 기록됩니다. 타임라인은 '언제 일이 끊겼는지'를 되짚게 해줘요." },
      { icon: "♡", kicker: "3. 케어로 연결", title: "진단 대신 휴식과 점검을 권해요", body: "이 예시만으로 건강 상태를 판단하지 않습니다. 다만 변화가 며칠 더 이어진다면, 휴식이나 점검 루틴, 필요하다면 전문 상담을 조심스럽게 권할 수 있어요." },
    ],
    note: "이수진 사장님의 데이터는 설명을 위한 가상 시나리오입니다. 메모리 가드는 질환을 진단하거나 단정하지 않으며, 실제 서비스에서는 장기간의 개인 기준선과 안전·업무 변화 패턴을 함께 살펴 케어 대화를 돕는 용도로 사용합니다.",
  },

  // -- Persona: 성수기 혼잡 (confound note) ---------------------------------
  {
    id: "busy-season",
    name: "김미래 사장님",
    avatarLabel: "김",
    tagline: "45세 여성 · 베이커리 사장, 크리스마스 성수기 4일차",
    summary: "평소엔 안정적이지만, 크리스마스 성수기 주말에는 예약 주문이 몰려 안전 알림과 반복 확인이 함께 늘어나는 패턴을 가정했습니다. 두 날 모두 '바쁨'으로 직접 표시했습니다.",
    weekRange: "12월 15일–21일",
    week: [
      { day: "월", date: "12/15", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "성수기 시작 전 평소 흐름", examples: [{ label: "10:00 원재료 정리", motionType: "high_reach" }, { label: "19:04 마감 확인 1회", motionType: "normal_task" }] },
      { day: "화", date: "12/16", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "평소와 비슷한 화요일", examples: [{ label: "13:15 진열대 정리", motionType: "high_reach" }, { label: "19:02 마감 확인 1회", motionType: "normal_task" }] },
      { day: "수", date: "12/17", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 8, busy: false, note: "한산했던 수요일", examples: [{ label: "11:30 재고 확인", motionType: "normal_task" }, { label: "18:55 마감 확인 1회", motionType: "normal_task" }] },
      { day: "목", date: "12/18", safetyAlerts: 1, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 11, busy: false, note: "오후 잠깐 붐볐지만 곧 안정", examples: [{ label: "15:10 예약 주문 문의 증가", motionType: "queue_shift" }, { label: "17:00 케이크 재료 정리", motionType: "high_reach" }, { label: "19:07 마감 확인 1회", motionType: "normal_task" }] },
      { day: "금", date: "12/19", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "크리스마스 예약 주문이 늘기 시작", examples: [{ label: "11:00 포장재 정리", motionType: "high_reach" }, { label: "17:30 케이크 예약 접수 급증", motionType: "queue_shift" }, { label: "19:10 마감 확인 1회", motionType: "normal_task" }] },
      { day: "토", date: "12/20", safetyAlerts: 2, doubleChecks: 2, unfinishedTasks: 2, microDelayRate: 42, busy: true, note: "크리스마스 성수기 첫 주말, 예약 주문이 몰림", examples: [{ label: "12:40 케이크 포장 주문 폭주", motionType: "high_reach" }, { label: "13:12 케이크 픽업 대기열 발생", motionType: "queue_shift" }, { label: "14:05 주문서 재입력 지연", motionType: "micro_delay" }, { label: "19:40 마감 반복 확인 2회", motionType: "double_check" }] },
      { day: "일", date: "12/21", safetyAlerts: 1, doubleChecks: 2, unfinishedTasks: 2, microDelayRate: 38, busy: true, note: "이브 전 마지막 주말도 손님이 계속 이어짐", examples: [{ label: "12:05 결제 대기 지연", motionType: "register_tap" }, { label: "13:30 포장 손님 대기열 증가", motionType: "queue_shift" }, { label: "17:15 재고 부족으로 진열 재정리", motionType: "high_reach" }, { label: "20:02 마감 반복 확인 2회", motionType: "double_check" }] },
    ],
    insights: [
      { icon: "☂", kicker: "1. 숫자만 보면 놀랄 수 있어요", title: "하지만 이 이틀은 '바쁨'으로 직접 표시된 날이에요", body: "안전 알림과 반복 확인이 눈에 띄게 늘었지만, 사장님이 오늘 매장 분위기를 직접 '바쁨'으로 표시한 날들이라 손님이 몰려 생긴 변화일 가능성이 큽니다." },
      { icon: "⌁", kicker: "2. 성수기 전후 비교", title: "성수기 시작 전 5일은 평소와 같았어요", body: "월–금은 예약이 조금씩 늘긴 했지만 평소 범위 안이었습니다. 변화가 시작된 시점을 성수기 시작과 겹쳐보면 원인을 가늠하기 쉬워져요." },
      { icon: "♡", kicker: "3. 케어로 연결", title: "성수기가 끝난 뒤에도 이어지는지가 중요해요", body: "바쁜 기간에 지표가 오르는 건 자연스러운 일이에요. 다만 성수기가 끝난 뒤에도 비슷한 흐름이 계속된다면, 그때는 따로 살펴볼 가치가 있습니다." },
    ],
    note: "김미래 사장님의 데이터는 '바쁜 매장'과 '변화하는 패턴'을 혼동하지 않기 위한 예시입니다. 메모리 가드는 사용자가 직접 표시한 '바쁨' 태그가 있는 기간은 평소 기준 계산에서 제외하고, 리포트에도 참고 문구로 함께 표시해요.",
  },

  // -- Persona: 하루의 소란 (single-day noise, watch not notable) ----------
  {
    id: "single-spike",
    name: "정하늘 사장님",
    avatarLabel: "정",
    tagline: "61세 남성 · 동네 철물점 12년차",
    summary: "오랫동안 안정적으로 가게를 운영해왔습니다. 토요일 하루만 손님 접촉으로 인한 안전 알림이 튀고, 다음 날 곧바로 평소 흐름으로 돌아오는 '하루짜리 소란'을 가정했습니다.",
    weekRange: "8월 4일–10일",
    week: [
      { day: "월", date: "8/4", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "평소와 같은 월요일", examples: [{ label: "10:30 진열대 정리", motionType: "high_reach" }, { label: "19:01 마감 확인 1회", motionType: "normal_task" }] },
      { day: "화", date: "8/5", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "조용한 화요일", examples: [{ label: "14:00 재고 확인", motionType: "normal_task" }, { label: "18:57 마감 확인 완료", motionType: "normal_task" }] },
      { day: "수", date: "8/6", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 8, busy: false, note: "평소와 비슷한 수요일", examples: [{ label: "11:15 공구함 정리", motionType: "high_reach" }, { label: "19:04 마감 확인 1회", motionType: "normal_task" }] },
      { day: "목", date: "8/7", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 11, busy: false, note: "평소와 비슷한 목요일", examples: [{ label: "16:20 재고 확인", motionType: "normal_task" }, { label: "19:08 마감 확인 1회", motionType: "normal_task" }] },
      { day: "금", date: "8/8", safetyAlerts: 0, doubleChecks: 0, unfinishedTasks: 0, microDelayRate: 9, busy: false, note: "평소와 비슷한 금요일", examples: [{ label: "10:00 진열대 정리", motionType: "high_reach" }, { label: "18:49 마감 확인 완료", motionType: "normal_task" }] },
      { day: "토", date: "8/9", safetyAlerts: 2, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 12, busy: false, note: "손님이 실수로 온열기를 건드려 안전 알림이 두 번 울린 하루", examples: [{ label: "14:20 온열기 안전 알림 2회 (손님 접촉)", motionType: "safety_alert" }, { label: "14:45 손님 응대 후 정리", motionType: "queue_shift" }, { label: "19:05 마감 확인 1회", motionType: "normal_task" }] },
      { day: "일", date: "8/10", safetyAlerts: 0, doubleChecks: 1, unfinishedTasks: 0, microDelayRate: 10, busy: false, note: "평소처럼 돌아온 일요일", examples: [{ label: "11:00 진열대 정리", motionType: "high_reach" }, { label: "19:02 마감 확인 1회", motionType: "normal_task" }] },
    ],
    insights: [
      { icon: "•", kicker: "1. 하루의 소란, 패턴은 아니에요", title: "토요일 딱 하루만 튀었어요", body: "토요일에 안전 알림이 늘었지만, 다음 날인 일요일에는 곧바로 평소 수준으로 돌아왔습니다. 메모리 가드는 하루의 소란만으로 알리지 않고, 며칠 더 이어지는지 먼저 봐요." },
      { icon: "⌁", kicker: "2. 시간축으로 원인 찾기", title: "타임라인을 보면 이유를 알 수 있어요", body: "타임라인을 되짚어보면 손님이 온열기를 건드려 생긴 일시적인 알림이었다는 걸 확인할 수 있어요. 원인을 알 수 있는 사건은 걱정할 신호와 구분돼요." },
      { icon: "♡", kicker: "3. 케어보다 관찰이 먼저예요", title: "지금은 '지켜보는 중'이에요", body: "지표 하나가 살짝 늘어난 정도라 케어 대화를 먼저 권하지 않습니다. 이런 날이 반복되기 시작할 때 비로소 대화를 조심스럽게 제안해요." },
    ],
    note: "정하늘 사장님의 데이터는 '하루의 튐(노이즈)'과 '지속되는 패턴'을 구분하기 위한 예시입니다. 메모리 가드는 하루 이상 이어지는 변화만 이어서 살펴봐요.",
  },
];

function buildPersona(def: DemoPersonaDefinition): DemoPersona {
  const logs = def.week.map((day, index) => demoDayToLog(def.id, day, index));
  const baselineLogs = logs.slice(0, BASELINE_DAY_COUNT);
  const recentLogs = logs.slice(BASELINE_DAY_COUNT);
  const baseline = computeBaseline(baselineLogs, { lookbackDays: BASELINE_DAY_COUNT });
  const signal = detectChangeSignal(recentLogs, baseline);
  return { ...def, signal, baselineDayCount: baseline?.dayCount ?? 0, baseline };
}

export const DEMO_PERSONAS: DemoPersona[] = PERSONA_DEFINITIONS.map(buildPersona);

export function signalLevelLabel(level: ChangeSignal["level"]): string {
  switch (level) {
    case "notable":
      return "변화가 여러 지표에서 함께 보여요";
    case "watch":
      return "조금 더 지켜봐요";
    default:
      return "평소와 비슷해요";
  }
}

// -- "왜 이 신호로 잡혔나요?" - the actual detection basis, per event -------
//
// This runs the SAME motion detector (app/motion-detection.ts) used on real
// camera sessions against this clip's generated coordinates - no label or
// intended motionType is given to it. What comes back is a genuine
// measurement of this specific clip, not a lookup keyed by its category.
// (In the real app, both this motion analysis AND the existing button/timer
// actions - recordDoubleCheck on a same-day re-check, recordSafetyAlert
// when the heater's still on, a task crossing SLOW_DELAY_SECONDS - feed the
// same counted signals; see page.tsx's stopPoseTracking.)

const CATEGORY_BY_DETECTED_TYPE: Record<
  DetectedMotionEvent["type"],
  "doubleChecks" | "safetyAlerts" | "microDelay"
> = {
  double_check: "doubleChecks",
  safety_alert: "safetyAlerts",
  micro_delay: "microDelay",
};

const RULE_BY_CATEGORY: Record<"doubleChecks" | "safetyAlerts" | "microDelay", string> = {
  doubleChecks:
    "같은 날 마감 확인을 다시 실행하거나, 카메라 동작에서 시작 위치로 돌아왔다가 다시 접근하는 패턴이 반복되면 '마감 반복 확인'으로 집계돼요.",
  safetyAlerts:
    "마감 점검 때 온열기 같은 위험 요소가 켜진 상태로 확인되거나, 카메라 동작에서 급격하고 반응적인 움직임이 감지되면 '안전 알림'으로 집계돼요.",
  microDelay: `업무 완료까지 ${SLOW_DELAY_SECONDS}초(2분) 이상 걸리거나, 카메라 동작에서 손이 멈춘 채 오래 머무는 구간이 감지되면 '미세 지연'으로 집계돼요.`,
};

const SIGNAL_REASON_BY_CATEGORY: Record<"doubleChecks" | "safetyAlerts" | "microDelay", string> = {
  doubleChecks: "마감 반복 확인 횟수가 평소보다 늘었어요.",
  safetyAlerts: "최근 안전 알림 빈도가 평소보다 늘었어요.",
  microDelay: "반복 업무 처리 시간이 평소보다 늘어난 날이 많았어요.",
};

export type DemoDetectionExplanation = {
  /** The real, plain-language rule that turns an action (button, timer, OR motion) into a counted signal. */
  rule: string;
  /** What the motion detector ACTUALLY found in this clip's coordinates - empty if nothing matched. */
  detected: DetectedMotionEvent[];
  /** This day's number for that category vs. the persona's personal baseline, if known. */
  metricLine: string | null;
  /** Whether this day is one of the recent days that fed the week's signal.reasons. */
  contributesToSignal: boolean;
};

/**
 * Explains why this event's clip is (or isn't) a counted signal - by
 * actually running the motion detector on `frames`, the same coordinates
 * the replay plays back. `dayIndex` is this day's position in
 * `persona.week` (0 = Monday).
 */
export function explainDemoEvent(
  persona: DemoPersona,
  dayIndex: number,
  day: DemoDay,
  event: DemoEvent,
  frames: number[][],
): DemoDetectionExplanation {
  const detected = detectMotionEvents(motionSamplesFromRawFrames(frames));

  if (detected.length === 0) {
    return {
      rule: "이 동작에서는 마감 반복 확인·미세 지연·안전 알림에 해당하는 패턴이 감지되지 않았어요. 케어 리포트에 따로 집계되지 않는 일반 업무 장면이에요.",
      detected: [],
      metricLine: null,
      contributesToSignal: false,
    };
  }

  // A clip can (rarely) trigger more than one pattern - all are returned in
  // `detected`, but the baseline comparison below is keyed off the first
  // (strongest-priority; see detectMotionEvents) one.
  const category = CATEGORY_BY_DETECTED_TYPE[detected[0].type];
  const rule = RULE_BY_CATEGORY[category];

  const baseline = persona.baseline;
  const metricLine = !baseline
    ? "평소 기록이 아직 충분하지 않아 비교할 기준이 없어요."
    : category === "doubleChecks"
      ? `이 날 마감 반복 확인 ${day.doubleChecks}회 (평소 하루 평균 ${baseline.doubleChecks.toFixed(1)}회)`
      : category === "safetyAlerts"
        ? `이 날 안전 알림 ${day.safetyAlerts}회 (평소 하루 평균 ${baseline.safetyAlerts.toFixed(1)}회)`
        : `이 날 미세 지연 비율 ${day.microDelayRate}% (평소 평균 ${Math.round(baseline.microDelayRate * 100)}%)`;

  // The week's signal.reasons come from averaging the recent (non-baseline)
  // days, not from any single day - so "contributes" here means this day is
  // part of that recent window AND its category's reason actually appears
  // in the computed signal, not that this one day alone crossed a threshold.
  const contributesToSignal =
    dayIndex >= BASELINE_DAY_COUNT && persona.signal.reasons.includes(SIGNAL_REASON_BY_CATEGORY[category]);

  return { rule, detected, metricLine, contributesToSignal };
}
