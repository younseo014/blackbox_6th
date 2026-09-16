export type OccupationId =
  | "cafe"
  | "hair_salon"
  | "restaurant"
  | "workshop"
  | "convenience_store"
  | "clothing_store";

export type WorkPhase = "open" | "business" | "close" | "break" | "unknown";

export type PrimitiveMotionLabel =
  | "WALK"
  | "STAND"
  | "SIT"
  | "BEND_FORWARD"
  | "BEND_DOWN"
  | "REACH_UP"
  | "REACH_DOWN"
  | "REACH_FORWARD"
  | "ARM_ELEVATED"
  | "TURN_BODY"
  | "CARRY"
  | "PUSH_PULL"
  | "STATIC_PAUSE"
  | "REPETITIVE_ARM"
  | "ZONE_TRANSITION"
  | "CHECK"
  | "CLOSE"
  | "DRY"
  | "GRASP"
  | "LIFT"
  | "LOWER"
  | "MIX"
  | "OPEN"
  | "PLACE"
  | "POUR"
  | "PRESS"
  | "PULL"
  | "PUSH"
  | "REACH_SIDE"
  | "RELEASE"
  | "RINSE"
  | "ROTATE"
  | "SCOOP"
  | "SCRUB"
  | "SHAKE"
  | "SORT"
  | "STEAM"
  | "STIR"
  | "TAMP"
  | "TRANSFER"
  | "WASH"
  | "WIPE";

export type TaskTemplate = {
  id: string;
  label: string;
  phase: WorkPhase;
  zones: string[];
  motions: PrimitiveMotionLabel[];
  priority: 1 | 2;
  fallbackLabel?: string;
};

export type SequenceTemplate = {
  id: string;
  label: string;
  phases: WorkPhase[];
  zones: string[];
};

export type OccupationTemplate = {
  id: OccupationId;
  icon: string;
  label: string;
  description: string;
  zones: Array<{ id: string; label: string }>;
  tasks: TaskTemplate[];
  sequences: SequenceTemplate[];
};

const task = (
  id: string,
  label: string,
  phase: WorkPhase,
  zones: string[],
  motions: PrimitiveMotionLabel[],
  priority: 1 | 2 = 1,
  fallbackLabel?: string,
): TaskTemplate => ({ id, label, phase, zones, motions, priority, fallbackLabel });

type CafeTaskSpec = readonly [
  label: string,
  phase: WorkPhase,
  zones: string[],
  motions: PrimitiveMotionLabel[],
];

const CAFE_TASK_ID_OVERRIDES: Record<string, string> = {
  "머신 전원 켜기": "MACHINE_SETUP",
  "컵 보충": "BAR_RESTOCK",
  "고객 주문 받기": "ORDER_ZONE",
  "냉장 재료 꺼내기": "FETCH_INGREDIENT",
  "음료 재료 혼합": "DRINK_PREP_TASK",
  "고객에게 음료 전달": "SERVE_ORDER",
  "테이블 닦기": "CLEAR_TABLE",
  "식기 세척": "DISHWASH",
  "머신 표면 닦기": "MACHINE_CLEAN",
  "매장 전체 최종 점검": "CLOSING_ROUTINE",
  "앉아서 휴식": "REST",
};

const CAFE_TASK_SPECS: CafeTaskSpec[] = [
  ["머신 전원 켜기", "open", ["ESPRESSO_MACHINE"], ["STAND", "REACH_FORWARD", "GRASP", "PRESS", "RELEASE"]],
  ["머신 예열 상태 확인", "open", ["ESPRESSO_MACHINE"], ["STAND", "STATIC_PAUSE", "CHECK"]],
  ["그라인더 준비", "open", ["ESPRESSO_MACHINE"], ["STAND", "REACH_FORWARD", "GRASP", "LIFT", "PLACE", "PRESS", "RELEASE"]],
  ["원두통 확인", "open", ["ESPRESSO_MACHINE", "INGREDIENT_STORAGE"], ["REACH_FORWARD", "GRASP", "OPEN", "CHECK", "CLOSE", "RELEASE"]],
  ["포터필터 준비", "open", ["ESPRESSO_MACHINE"], ["REACH_FORWARD", "GRASP", "LIFT", "TRANSFER", "PLACE", "RELEASE"]],
  ["머신 주변 작업공간 정리", "open", ["ESPRESSO_MACHINE", "DRINK_PREP"], ["REACH_FORWARD", "GRASP", "LIFT", "TRANSFER", "PLACE", "RELEASE"]],
  ["컵 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["뚜껑 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["빨대 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["냅킨·소모품 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["원두 보충", "open", ["INGREDIENT_STORAGE", "ESPRESSO_MACHINE"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "POUR", "PLACE", "RELEASE"]],
  ["시럽·소스 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["우유·음료 재료 보충", "open", ["FRIDGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["고객 주문 받기", "business", ["POS"], ["STAND", "TURN_BODY", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["주문 내용 확인", "business", ["POS"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["POS 주문 입력", "business", ["POS"], ["STAND", "REACH_FORWARD", "PRESS", "RELEASE", "CHECK"]],
  ["결제 처리", "business", ["POS"], ["STAND", "REACH_FORWARD", "GRASP", "PRESS", "RELEASE", "CHECK"]],
  ["영수증 출력", "business", ["POS"], ["STAND", "REACH_FORWARD", "GRASP", "PULL", "LIFT", "RELEASE"]],
  ["주문 내역 확인", "business", ["POS"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["고객에게 주문 완료 안내", "business", ["POS"], ["STAND", "TURN_BODY", "REACH_FORWARD", "CHECK"]],
  ["냉장 재료 꺼내기", "business", ["FRIDGE"], ["BEND_FORWARD", "REACH_FORWARD", "GRASP", "LIFT", "CLOSE", "CARRY"]],
  ["냉장 재료 제조대로 이동", "business", ["FRIDGE", "DRINK_PREP"], ["WALK", "CARRY", "ZONE_TRANSITION", "PLACE", "RELEASE"]],
  ["상온 재료 꺼내기", "business", ["INGREDIENT_STORAGE"], ["REACH_FORWARD", "GRASP", "LIFT", "CARRY"]],
  ["상온 재료 제조대로 이동", "business", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "CARRY", "ZONE_TRANSITION", "PLACE", "RELEASE"]],
  ["원두 가져오기", "business", ["INGREDIENT_STORAGE", "ESPRESSO_MACHINE"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "ZONE_TRANSITION", "PLACE", "RELEASE"]],
  ["우유 가져오기", "business", ["FRIDGE", "DRINK_PREP"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "ZONE_TRANSITION", "PLACE", "RELEASE"]],
  ["재료를 제조대에 배치하기", "business", ["DRINK_PREP"], ["REACH_FORWARD", "GRASP", "LIFT", "TRANSFER", "PLACE", "RELEASE"]],
  ["주문 음료 확인", "business", ["POS", "DRINK_PREP"], ["WALK", "ZONE_TRANSITION", "REACH_FORWARD", "CHECK"]],
  ["컵 준비", "business", ["DRINK_PREP"], ["REACH_FORWARD", "GRASP", "LIFT", "TRANSFER", "PLACE", "RELEASE"]],
  ["컵에 얼음 담기", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "SCOOP", "TRANSFER", "PLACE", "RELEASE"]],
  ["원두 준비", "business", ["ESPRESSO_MACHINE"], ["REACH_FORWARD", "GRASP", "LIFT", "PLACE", "RELEASE"]],
  ["원두 분쇄", "business", ["ESPRESSO_MACHINE"], ["GRASP", "PLACE", "PRESS", "RELEASE", "STATIC_PAUSE"]],
  ["포터필터에 원두 담기", "business", ["ESPRESSO_MACHINE"], ["GRASP", "LIFT", "SCOOP", "TRANSFER", "PLACE", "RELEASE"]],
  ["원두 탬핑", "business", ["ESPRESSO_MACHINE"], ["GRASP", "PRESS", "TAMP", "RELEASE"]],
  ["포터필터 장착", "business", ["ESPRESSO_MACHINE"], ["GRASP", "LIFT", "REACH_FORWARD", "ROTATE", "PLACE", "RELEASE"]],
  ["에스프레소 추출 시작", "business", ["ESPRESSO_MACHINE"], ["REACH_FORWARD", "PRESS", "RELEASE"]],
  ["에스프레소 추출 대기", "business", ["ESPRESSO_MACHINE"], ["STAND", "STATIC_PAUSE", "CHECK"]],
  ["우유 준비", "business", ["FRIDGE", "DRINK_PREP"], ["REACH_FORWARD", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE"]],
  ["우유 스팀", "business", ["ESPRESSO_MACHINE"], ["GRASP", "PLACE", "PRESS", "ROTATE", "STEAM", "RELEASE"]],
  ["우유 붓기", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "POUR", "ROTATE", "LOWER", "RELEASE"]],
  ["시럽 첨가", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "REACH_FORWARD", "PRESS", "POUR", "RELEASE"]],
  ["소스 첨가", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "REACH_FORWARD", "PRESS", "POUR", "RELEASE"]],
  ["음료 재료 혼합", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "POUR", "MIX", "STIR", "PLACE", "RELEASE"]],
  ["음료 저어 섞기", "business", ["DRINK_PREP"], ["GRASP", "STIR", "ROTATE", "RELEASE"]],
  ["음료 상태 확인", "business", ["DRINK_PREP"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["컵 뚜껑 닫기", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "PLACE", "PRESS", "RELEASE"]],
  ["음료 포장", "business", ["DRINK_PREP"], ["GRASP", "LIFT", "PLACE", "PULL", "CLOSE", "RELEASE"]],
  ["완성 음료 픽업대로 이동", "business", ["DRINK_PREP", "SERVING"], ["GRASP", "LIFT", "CARRY", "WALK", "ZONE_TRANSITION", "PLACE", "RELEASE"]],
  ["완성 음료 확인", "business", ["SERVING"], ["STAND", "REACH_FORWARD", "CHECK"]],
  ["픽업대에 음료 배치", "business", ["SERVING"], ["GRASP", "LIFT", "REACH_FORWARD", "PLACE", "RELEASE"]],
  ["주문 번호 확인", "business", ["SERVING"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["고객에게 음료 전달", "business", ["SERVING", "HALL"], ["GRASP", "LIFT", "CARRY", "REACH_FORWARD", "TRANSFER", "RELEASE"]],
  ["테이크아웃 음료 전달", "business", ["SERVING"], ["GRASP", "LIFT", "CARRY", "REACH_FORWARD", "TRANSFER", "RELEASE"]],
  ["음료 전달 후 작업대로 복귀", "business", ["SERVING", "DRINK_PREP"], ["WALK", "ZONE_TRANSITION"]],
  ["빈 컵 수거", "business", ["HALL"], ["WALK", "REACH_FORWARD", "GRASP", "LIFT", "CARRY"]],
  ["사용한 식기 수거", "business", ["HALL"], ["BEND_FORWARD", "REACH_FORWARD", "GRASP", "LIFT", "CARRY"]],
  ["테이블 위 쓰레기 수거", "business", ["HALL", "WASTE"], ["BEND_FORWARD", "REACH_FORWARD", "GRASP", "LIFT", "CARRY", "ZONE_TRANSITION", "RELEASE"]],
  ["테이블 닦기", "business", ["HALL"], ["STAND", "GRASP", "REACH_FORWARD", "WIPE", "SCRUB", "REACH_SIDE", "RELEASE"]],
  ["테이블 위 물품 정리", "business", ["HALL"], ["REACH_FORWARD", "GRASP", "LIFT", "TRANSFER", "PLACE", "RELEASE"]],
  ["의자 정돈", "business", ["HALL"], ["GRASP", "PUSH", "PULL", "PLACE", "RELEASE"]],
  ["테이블 상태 확인", "business", ["HALL"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["사용 식기 분류", "business", ["SINK"], ["GRASP", "LIFT", "SORT", "PLACE", "RELEASE"]],
  ["식기 불리기", "business", ["SINK"], ["GRASP", "LIFT", "PLACE", "POUR", "RELEASE", "STATIC_PAUSE"]],
  ["컵 세척", "business", ["SINK"], ["GRASP", "LIFT", "WASH", "SCRUB", "ROTATE", "RINSE", "PLACE", "RELEASE"]],
  ["접시 세척", "business", ["SINK"], ["GRASP", "LIFT", "WASH", "SCRUB", "ROTATE", "RINSE", "PLACE", "RELEASE"]],
  ["식기 세척", "business", ["SINK"], ["GRASP", "LIFT", "WASH", "SCRUB", "ROTATE", "RINSE", "PLACE", "RELEASE"]],
  ["컵 헹구기", "business", ["SINK"], ["GRASP", "RINSE", "ROTATE", "RELEASE"]],
  ["식기 헹구기", "business", ["SINK"], ["GRASP", "RINSE", "ROTATE", "RELEASE"]],
  ["세척한 식기 물기 제거", "business", ["SINK"], ["GRASP", "SHAKE", "ROTATE", "DRY", "PLACE", "RELEASE"]],
  ["식기 건조", "business", ["SINK", "DRINK_PREP"], ["GRASP", "LIFT", "CARRY", "WALK", "PLACE", "RELEASE"]],
  ["컵·식기 제자리 정리", "business", ["DRINK_PREP", "INGREDIENT_STORAGE"], ["GRASP", "LIFT", "CARRY", "REACH_FORWARD", "PLACE", "RELEASE"]],
  ["머신 표면 닦기", "close", ["ESPRESSO_MACHINE"], ["STAND", "GRASP", "REACH_FORWARD", "WIPE", "SCRUB", "REACH_SIDE", "RELEASE"]],
  ["추출구 세척", "close", ["ESPRESSO_MACHINE", "SINK"], ["GRASP", "LIFT", "WIPE", "SCRUB", "RINSE", "PLACE", "RELEASE"]],
  ["포터필터 세척", "close", ["ESPRESSO_MACHINE", "SINK"], ["GRASP", "LIFT", "CARRY", "WASH", "SCRUB", "RINSE", "PLACE", "RELEASE"]],
  ["그룹헤드 세척", "close", ["ESPRESSO_MACHINE"], ["REACH_FORWARD", "GRASP", "WASH", "SCRUB", "RINSE", "RELEASE"]],
  ["스팀 노즐 세척", "close", ["ESPRESSO_MACHINE", "SINK"], ["GRASP", "REACH_FORWARD", "WIPE", "SCRUB", "RINSE", "RELEASE"]],
  ["스팀 노즐 닦기", "close", ["ESPRESSO_MACHINE"], ["GRASP", "REACH_FORWARD", "WIPE", "ROTATE", "RELEASE"]],
  ["머신 주변 물기 제거", "close", ["ESPRESSO_MACHINE", "DRINK_PREP"], ["GRASP", "REACH_FORWARD", "WIPE", "DRY", "REACH_SIDE", "RELEASE"]],
  ["그라인더 주변 정리", "close", ["ESPRESSO_MACHINE"], ["GRASP", "REACH_FORWARD", "WIPE", "SCRUB", "PLACE", "RELEASE"]],
  ["머신 청소 상태 확인", "close", ["ESPRESSO_MACHINE"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["남은 재료 확인", "close", ["DRINK_PREP", "INGREDIENT_STORAGE"], ["WALK", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["냉장 재료 정리", "close", ["FRIDGE"], ["GRASP", "LIFT", "CARRY", "REACH_FORWARD", "PLACE", "CLOSE", "RELEASE"]],
  ["원재료 보관", "close", ["DRINK_PREP", "INGREDIENT_STORAGE"], ["GRASP", "LIFT", "CARRY", "WALK", "PLACE", "RELEASE"]],
  ["재고 상태 확인", "close", ["INGREDIENT_STORAGE"], ["STAND", "REACH_FORWARD", "CHECK", "STATIC_PAUSE"]],
  ["설거지 마무리", "close", ["SINK"], ["GRASP", "WASH", "SCRUB", "RINSE", "DRY", "PLACE", "RELEASE"]],
  ["작업대 정리", "close", ["DRINK_PREP"], ["GRASP", "LIFT", "TRANSFER", "WIPE", "PLACE", "RELEASE"]],
  ["머신 주변 정리", "close", ["ESPRESSO_MACHINE"], ["GRASP", "LIFT", "TRANSFER", "WIPE", "PLACE", "RELEASE"]],
  ["홀 테이블 정리", "close", ["HALL"], ["WALK", "GRASP", "LIFT", "TRANSFER", "WIPE", "PLACE", "RELEASE"]],
  ["쓰레기 수거", "close", ["HALL", "WASTE"], ["GRASP", "LIFT", "CARRY", "WALK", "ZONE_TRANSITION", "RELEASE"]],
  ["분리수거", "close", ["WASTE"], ["BEND_FORWARD", "GRASP", "LIFT", "SORT", "PLACE", "RELEASE"]],
  ["바닥 상태 확인", "close", ["HALL", "DRINK_PREP"], ["WALK", "BEND_DOWN", "CHECK", "STATIC_PAUSE"]],
  ["매장 전체 최종 점검", "close", ["ENTRANCE", "POS", "ESPRESSO_MACHINE", "DRINK_PREP", "HALL"], ["WALK", "ZONE_TRANSITION", "TURN_BODY", "CHECK", "STATIC_PAUSE"]],
  ["출입구 정리", "close", ["ENTRANCE"], ["GRASP", "PUSH", "PULL", "WIPE", "PLACE", "RELEASE"]],
  ["영업 종료", "close", ["ENTRANCE", "POS"], ["WALK", "REACH_FORWARD", "GRASP", "PRESS", "CLOSE", "RELEASE"]],
  ["휴게 공간으로 이동", "break", ["REST"], ["WALK", "ZONE_TRANSITION"]],
  ["앉아서 휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"]],
  ["서서 대기", "break", ["REST"], ["STAND", "STATIC_PAUSE"]],
  ["업무 재개 전 대기", "break", ["REST", "DRINK_PREP"], ["STATIC_PAUSE", "STAND", "WALK", "ZONE_TRANSITION"]],
];

const CAFE_TASKS = CAFE_TASK_SPECS.map(([label, phase, zones, motions], index) => task(
  CAFE_TASK_ID_OVERRIDES[label] ?? `CAFE_${phase.toUpperCase()}_${String(index + 1).padStart(3, "0")}`,
  label,
  phase,
  zones,
  motions,
  label === "식기 세척" || label === "앉아서 휴식" ? 2 : 1,
));

export const OCCUPATION_TEMPLATES: OccupationTemplate[] = [
  {
    id: "cafe",
    icon: "☕",
    label: "개인 카페",
    description: "주문부터 제조·서빙·정리까지 반복되는 바 업무를 중심으로 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["POS", "주문·결제"], ["ESPRESSO_MACHINE", "커피머신"],
      ["DRINK_PREP", "음료 제조대"], ["FRIDGE", "냉장고"], ["INGREDIENT_STORAGE", "재료·재고"],
      ["SINK", "설거지·세척"], ["SERVING", "픽업·서빙대"], ["HALL", "고객 좌석"],
      ["WASTE", "쓰레기 처리"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: CAFE_TASKS,
    sequences: [
      { id: "CAFE_OPEN", label: "카페 오픈 루틴", phases: ["open"], zones: ["ENTRANCE", "POS", "INGREDIENT_STORAGE", "FRIDGE", "DRINK_PREP", "ESPRESSO_MACHINE", "HALL"] },
      { id: "CAFE_ORDER_LOOP", label: "주문 처리 루프", phases: ["business"], zones: ["POS", "FRIDGE", "INGREDIENT_STORAGE", "DRINK_PREP", "ESPRESSO_MACHINE", "SERVING", "HALL"] },
      { id: "CAFE_DRINK_PREP", label: "음료 제조 루틴", phases: ["business"], zones: ["POS", "DRINK_PREP", "ESPRESSO_MACHINE", "FRIDGE", "SERVING"] },
      { id: "CAFE_TABLE_RESET", label: "테이블 정리 루틴", phases: ["business"], zones: ["HALL", "WASTE", "SINK", "DRINK_PREP"] },
      { id: "CAFE_DISHWASH", label: "설거지 루틴", phases: ["business"], zones: ["HALL", "SINK", "DRINK_PREP"] },
      { id: "CAFE_MACHINE_CLEAN", label: "머신 청소 루틴", phases: ["close"], zones: ["ESPRESSO_MACHINE", "SINK", "DRINK_PREP"] },
      { id: "CAFE_CLOSE", label: "카페 마감 루틴", phases: ["close"], zones: ["SINK", "INGREDIENT_STORAGE", "FRIDGE", "DRINK_PREP", "ESPRESSO_MACHINE", "HALL", "WASTE", "ENTRANCE"] },
    ],
  },
  {
    id: "hair_salon",
    icon: "✂",
    label: "미용실",
    description: "고객석과 도구·세척 공간을 오가는 시술 루프를 중심으로 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["CLIENT_CHAIR", "고객 시술 의자"], ["SHAMPOO", "샴푸대"],
      ["TOOL_STATION", "도구·작업대"], ["MATERIAL", "약제·소모품"], ["WASH_CLEAN", "세척·소독"],
      ["POS_ADMIN", "계산·예약"], ["STORAGE", "수납"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: [
      task("STATION_SETUP", "작업대 준비", "open", ["TOOL_STATION", "STORAGE"], ["REACH_FORWARD", "CARRY"]),
      task("CLIENT_GREETING", "고객 맞이", "business", ["ENTRANCE", "CLIENT_CHAIR"], ["WALK", "TURN_BODY"]),
      task("SHAMPOO_TASK", "샴푸", "business", ["SHAMPOO"], ["BEND_FORWARD", "REPETITIVE_ARM"]),
      task("HAIR_SERVICE_WORK", "헤어 시술", "business", ["CLIENT_CHAIR"], ["ARM_ELEVATED", "REPETITIVE_ARM"], 1, "시술 업무"),
      task("FETCH_TOOL", "도구·재료 가져오기", "business", ["CLIENT_CHAIR", "TOOL_STATION", "MATERIAL"], ["WALK", "REACH_FORWARD"]),
      task("CLIENT_RESET", "고객 사이 자리 정리", "business", ["CLIENT_CHAIR", "WASH_CLEAN", "TOOL_STATION"], ["CARRY", "ZONE_TRANSITION"]),
      task("TOOL_CLEAN", "도구 세척·소독", "business", ["WASH_CLEAN"], ["REPETITIVE_ARM", "BEND_FORWARD"], 2),
      task("CLOSING_ROUTINE", "미용실 마감", "close", ["CLIENT_CHAIR", "TOOL_STATION", "WASH_CLEAN", "STORAGE", "ENTRANCE"], ["ZONE_TRANSITION", "CARRY"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "SALON_CLIENT_RESET", label: "고객 사이 정리", phases: ["business"], zones: ["CLIENT_CHAIR", "TOOL_STATION", "WASH_CLEAN", "CLIENT_CHAIR"] },
      { id: "SALON_CLOSE", label: "미용실 마감 루틴", phases: ["close"], zones: ["CLIENT_CHAIR", "TOOL_STATION", "WASH_CLEAN", "STORAGE", "ENTRANCE"] },
    ],
  },
  {
    id: "restaurant",
    icon: "🍽",
    label: "식당",
    description: "주문·조리·서빙·정리 사이클과 주방·홀 이동을 중심으로 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["POS", "계산대"], ["KITCHEN_PREP", "주방 준비대"],
      ["COOKING", "조리 구역"], ["COLD_STORAGE", "냉장·냉동"], ["INGREDIENT_STORAGE", "식재료 보관"],
      ["SINK", "세척"], ["SERVING", "배식·서빙"], ["HALL", "홀"], ["WASTE", "폐기물"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: [
      task("OPEN_INSPECTION", "오픈 시설 점검", "open", ["KITCHEN_PREP", "COOKING", "COLD_STORAGE", "HALL"], ["WALK", "ZONE_TRANSITION"]),
      task("INGREDIENT_PREP", "식재료 준비", "open", ["KITCHEN_PREP", "COLD_STORAGE"], ["REPETITIVE_ARM", "BEND_FORWARD"]),
      task("COOKING_TASK", "조리", "business", ["COOKING", "KITCHEN_PREP"], ["REPETITIVE_ARM", "TURN_BODY"]),
      task("SERVE_ORDER", "서빙", "business", ["SERVING", "HALL"], ["CARRY", "WALK"]),
      task("CLEAR_TABLE", "테이블 정리", "business", ["HALL", "SINK"], ["BEND_FORWARD", "CARRY"]),
      task("DISHWASH", "조리도구·식기 세척", "business", ["SINK"], ["REPETITIVE_ARM", "BEND_FORWARD"]),
      task("CLOSING_ROUTINE", "식당 마감", "close", ["INGREDIENT_STORAGE", "SINK", "HALL", "WASTE", "ENTRANCE"], ["ZONE_TRANSITION", "CARRY"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "RESTAURANT_ORDER_LOOP", label: "주문 처리 사이클", phases: ["business"], zones: ["POS", "KITCHEN_PREP", "COOKING", "SERVING", "HALL", "SINK"] },
      { id: "RESTAURANT_CLOSE", label: "식당 마감 루틴", phases: ["close"], zones: ["INGREDIENT_STORAGE", "SINK", "HALL", "WASTE", "ENTRANCE"] },
    ],
  },
  {
    id: "workshop",
    icon: "🎨",
    label: "공방",
    description: "클래스 전·중·후 흐름과 작업대·재료·세척 공간의 변화를 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["RECEPTION", "예약·접수"], ["WORKBENCH", "작업대"],
      ["DEMO_AREA", "시범 공간"], ["TOOL_STORAGE", "도구 보관"], ["MATERIAL_STORAGE", "재료 보관"],
      ["SAFETY_AREA", "안전 관리"], ["PHOTO_AREA", "촬영"], ["PACKING", "포장"], ["SINK_CLEAN", "세척"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: [
      task("CLASS_SETUP", "클래스 준비", "open", ["WORKBENCH", "TOOL_STORAGE", "MATERIAL_STORAGE"], ["CARRY", "REACH_FORWARD"]),
      task("MATERIAL_PORTIONING", "재료 계량·소분", "open", ["MATERIAL_STORAGE", "WORKBENCH"], ["REPETITIVE_ARM", "REACH_FORWARD"]),
      task("CLASS_GUIDANCE", "클래스 지도", "business", ["WORKBENCH", "DEMO_AREA"], ["STAND", "ARM_ELEVATED"], 1, "클래스 활동"),
      task("MATERIAL_SUPPORT", "재료 추가 지원", "business", ["MATERIAL_STORAGE", "WORKBENCH"], ["WALK", "CARRY"]),
      task("PACK_PRODUCT", "완성품 포장", "close", ["PACKING"], ["REPETITIVE_ARM", "REACH_FORWARD"]),
      task("CLASS_RESET", "작업대·도구 정리", "close", ["WORKBENCH", "SINK_CLEAN", "TOOL_STORAGE"], ["CARRY", "REPETITIVE_ARM"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "WORKSHOP_CLASS", label: "클래스 진행", phases: ["open", "business", "close"], zones: ["MATERIAL_STORAGE", "WORKBENCH", "DEMO_AREA", "WORKBENCH", "PACKING", "SINK_CLEAN"] },
    ],
  },
  {
    id: "convenience_store",
    icon: "🏪",
    label: "편의점",
    description: "입고·검수와 매대 보충, POS, 마감 정산의 반복 동선을 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["POS", "POS·카운터"], ["CASH_SAFE", "시재·금고"],
      ["RECEIVING", "물류 입고"], ["DISPLAY_FOOD", "식품 매대"], ["DISPLAY_GENERAL", "일반 매대"],
      ["WALK_IN_COOLER", "워크인 냉장고"], ["STORAGE", "창고"], ["AISLE", "매장 통로"], ["WASTE", "분리수거"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: [
      task("CASH_COUNT", "시재 점검", "open", ["POS", "CASH_SAFE"], ["STAND", "REPETITIVE_ARM"]),
      task("DELIVERY_CHECK", "물류 입고·검수", "open", ["RECEIVING", "STORAGE"], ["BEND_DOWN", "CARRY"]),
      task("EXPIRY_CHECK", "유통기한 확인", "business", ["DISPLAY_FOOD", "AISLE"], ["WALK", "REACH_FORWARD"]),
      task("FACE_UP", "상품 페이스업", "business", ["DISPLAY_FOOD", "DISPLAY_GENERAL"], ["REPETITIVE_ARM", "REACH_FORWARD"]),
      task("POS_CHECKOUT", "POS 결제", "business", ["POS"], ["STAND", "REPETITIVE_ARM"]),
      task("SHELF_RESTOCK", "상품 진열", "business", ["STORAGE", "DISPLAY_GENERAL"], ["CARRY", "REACH_UP"]),
      task("COOLER_RESTOCK", "냉장고 채우기", "business", ["WALK_IN_COOLER", "STORAGE"], ["CARRY", "PUSH_PULL"]),
      task("WASTE_OUT", "쓰레기·분리수거", "close", ["WASTE", "ENTRANCE"], ["CARRY", "BEND_DOWN"]),
      task("HANDOVER_CLOSE", "시재 마감·인수인계", "close", ["POS", "CASH_SAFE"], ["STAND", "REPETITIVE_ARM"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "CONVENIENCE_RESTOCK", label: "상품 보충 루프", phases: ["business"], zones: ["STORAGE", "AISLE", "DISPLAY_GENERAL", "AISLE", "STORAGE"] },
      { id: "CONVENIENCE_CLOSE", label: "편의점 마감", phases: ["close"], zones: ["POS", "WASTE", "AISLE", "POS"] },
    ],
  },
  {
    id: "clothing_store",
    icon: "👗",
    label: "1인 옷가게",
    description: "접기·걸기·다림질처럼 팔 범위가 큰 반복 업무와 피팅룸 동선을 학습해요.",
    zones: [
      ["ENTRANCE", "출입구"], ["POS_ADMIN", "계산·장부"], ["DISPLAY", "의류 진열"],
      ["FOLDING_TABLE", "폴딩 테이블"], ["FITTING_ROOM", "피팅룸"], ["STEAMING", "스팀 다림질"],
      ["WINDOW_MANNEQUIN", "쇼윈도·마네킹"], ["STOCK", "재고 보관"], ["MIRROR_CONTENT", "거울·콘텐츠"],
      ["SECURITY_SHUTTER", "보안 셔터"], ["REST", "휴식"],
    ].map(([id, label]) => ({ id, label })),
    tasks: [
      task("STEAM_IRONING", "스팀 다림질", "open", ["STEAMING"], ["ARM_ELEVATED", "REPETITIVE_ARM"]),
      task("MANNEQUIN_SETUP", "마네킹·행거 세팅", "open", ["WINDOW_MANNEQUIN", "ENTRANCE"], ["CARRY", "PUSH_PULL"]),
      task("FOLD_CLOTHES", "의류 폴딩", "business", ["FOLDING_TABLE"], ["REPETITIVE_ARM", "REACH_FORWARD"]),
      task("DISPLAY_RESTOCK", "선반·행거 진열", "business", ["DISPLAY", "STOCK"], ["CARRY", "REACH_UP"]),
      task("FITTING_SUPPORT", "피팅 도움", "business", ["FITTING_ROOM", "DISPLAY", "STOCK"], ["WALK", "ARM_ELEVATED"]),
      task("CONTENT_CAPTURE", "SNS 거울 촬영", "business", ["MIRROR_CONTENT"], ["ARM_ELEVATED", "TURN_BODY"], 2),
      task("FITTING_RESET", "피팅룸 의류 원위치", "close", ["FITTING_ROOM", "FOLDING_TABLE", "DISPLAY"], ["CARRY", "REACH_UP"]),
      task("SHUTTER_CLOSE", "보안 셔터 내리기", "close", ["SECURITY_SHUTTER"], ["REACH_UP", "BEND_DOWN", "PUSH_PULL"]),
      task("ADMIN_CLOSE", "정산·장부 작성", "close", ["POS_ADMIN"], ["SIT", "STATIC_PAUSE"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "CLOTHING_FITTING_RESET", label: "피팅룸 정리", phases: ["business", "close"], zones: ["FITTING_ROOM", "FOLDING_TABLE", "DISPLAY", "FITTING_ROOM"] },
      { id: "CLOTHING_CLOSE", label: "옷가게 마감", phases: ["close"], zones: ["FITTING_ROOM", "DISPLAY", "POS_ADMIN", "SECURITY_SHUTTER"] },
    ],
  },
];

export function getOccupationTemplate(id: OccupationId): OccupationTemplate {
  return OCCUPATION_TEMPLATES.find((item) => item.id === id) ?? OCCUPATION_TEMPLATES[0];
}

const OCCUPATION_NAME_HINTS: Array<[RegExp, OccupationId]> = [
  [/카페|커피|음료|디저트|베이커리|제과|빵집|coffee|cafe|bakery/i, "cafe"],
  [/미용|헤어|살롱|이발|바버|네일|뷰티|피부|hair|salon|barber|nail/i, "hair_salon"],
  [/식당|음식|요리|주방|분식|한식|중식|일식|밥집|덮밥|국밥|치킨|피자|레스토랑|술집|포차|restaurant|kitchen/i, "restaurant"],
  [/공방|클래스|작업실|도예|공예|꽃집|플라워|스튜디오|수리|제작|workshop|studio/i, "workshop"],
  [/편의점|마트|슈퍼|잡화|소매|무인점|convenience|mart|retail/i, "convenience_store"],
  [/옷|의류|패션|부티크|쇼핑몰|clothing|fashion|boutique/i, "clothing_store"],
];

export type OccupationContextInference = {
  template: OccupationTemplate;
  matched: boolean;
};

/** Maps a user-entered business name to the closest built-in local context. */
export function inferOccupationContext(label: string): OccupationContextInference {
  const normalized = label.trim().replace(/\s|[·\-_]/g, "");
  const direct = OCCUPATION_TEMPLATES.find((template) => {
    const templateLabel = template.label.replace(/\s|[·\-_]/g, "");
    return normalized.includes(templateLabel) || templateLabel.includes(normalized);
  });
  if (direct && normalized) return { template: direct, matched: true };

  const hintedId = OCCUPATION_NAME_HINTS.find(([pattern]) => pattern.test(normalized))?.[1];
  return {
    template: getOccupationTemplate(hintedId ?? "cafe"),
    matched: Boolean(hintedId),
  };
}

const ZONE_NAME_HINTS: Array<[RegExp, string[]]> = [
  [/출입|입구|현관|문앞/, ["ENTRANCE"]],
  [/주문|결제|계산|포스|카운터|접수|예약/, ["POS", "POS_ADMIN", "RECEPTION"]],
  [/세척|설거지|싱크|소독/, ["SINK", "SINK_CLEAN", "WASH_CLEAN", "SHAMPOO"]],
  [/창고|재고|수납|보관|재료/, ["STORAGE", "STOCK", "INGREDIENT_STORAGE", "MATERIAL_STORAGE", "TOOL_STORAGE", "MATERIAL"]],
  [/냉장|냉동|냉고/, ["FRIDGE", "COLD_STORAGE", "WALK_IN_COOLER"]],
  [/제조|준비|조리|작업/, ["DRINK_PREP", "KITCHEN_PREP", "COOKING", "WORKBENCH", "TOOL_STATION"]],
  [/포장|픽업|배식|서빙/, ["PACKING", "SERVING"]],
  [/홀|좌석|테이블|대기|테라스/, ["HALL", "CLIENT_CHAIR"]],
  [/쓰레기|폐기|분리수거/, ["WASTE"]],
  [/휴게|휴식|직원실/, ["REST"]],
  [/피팅|탈의/, ["FITTING_ROOM"]],
  [/촬영|사진|거울/, ["PHOTO_AREA", "MIRROR_CONTENT"]],
];

export function inferZoneContext(template: OccupationTemplate, label: string) {
  const normalized = label.replace(/\s|[·\-_]/g, "");
  const direct = template.zones.find((zone) => {
    const zoneLabel = zone.label.replace(/\s|[·\-_]/g, "");
    return normalized.includes(zoneLabel) || zoneLabel.includes(normalized);
  });
  if (direct) return direct;
  const ids = ZONE_NAME_HINTS.find(([pattern]) => pattern.test(normalized))?.[1];
  return ids ? template.zones.find((zone) => ids.includes(zone.id)) ?? null : null;
}

export function phaseForHour(hour: number): WorkPhase {
  if (hour < 10) return "open";
  if (hour >= 20) return "close";
  return "business";
}
