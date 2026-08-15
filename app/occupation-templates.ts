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
  | "ZONE_TRANSITION";

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
    tasks: [
      task("MACHINE_SETUP", "머신 준비", "open", ["ESPRESSO_MACHINE"], ["STAND", "REPETITIVE_ARM"]),
      task("BAR_RESTOCK", "바 재고 보충", "open", ["INGREDIENT_STORAGE", "DRINK_PREP"], ["CARRY", "REACH_FORWARD"]),
      task("ORDER_ZONE", "주문·결제", "business", ["POS"], ["STAND", "REACH_FORWARD"]),
      task("FETCH_INGREDIENT", "재료 가져오기", "business", ["FRIDGE", "INGREDIENT_STORAGE", "DRINK_PREP"], ["WALK", "CARRY"]),
      task("DRINK_PREP_TASK", "음료 제조", "business", ["ESPRESSO_MACHINE", "DRINK_PREP"], ["REPETITIVE_ARM", "REACH_FORWARD"]),
      task("SERVE_ORDER", "음료 전달", "business", ["SERVING", "HALL"], ["CARRY", "WALK"]),
      task("CLEAR_TABLE", "테이블 정리", "business", ["HALL"], ["BEND_FORWARD", "CARRY"]),
      task("DISHWASH", "설거지", "business", ["SINK"], ["REPETITIVE_ARM", "BEND_FORWARD"], 2),
      task("MACHINE_CLEAN", "머신 청소", "close", ["ESPRESSO_MACHINE", "SINK"], ["REPETITIVE_ARM", "BEND_FORWARD"]),
      task("CLOSING_ROUTINE", "카페 마감", "close", ["SINK", "INGREDIENT_STORAGE", "HALL", "WASTE", "ENTRANCE"], ["ZONE_TRANSITION", "CARRY"]),
      task("REST", "휴식", "break", ["REST"], ["SIT", "STATIC_PAUSE"], 2),
    ],
    sequences: [
      { id: "CAFE_ORDER_LOOP", label: "주문 처리 루프", phases: ["business"], zones: ["POS", "FRIDGE", "DRINK_PREP", "SERVING", "HALL", "DRINK_PREP"] },
      { id: "CAFE_CLOSE", label: "카페 마감 루틴", phases: ["close"], zones: ["SINK", "INGREDIENT_STORAGE", "HALL", "WASTE", "ENTRANCE"] },
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

export function phaseForHour(hour: number): WorkPhase {
  if (hour < 10) return "open";
  if (hour >= 20) return "close";
  return "business";
}
