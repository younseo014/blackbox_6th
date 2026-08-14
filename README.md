# vinext-starter

A clean full-stack starter running on
[vinext](https://github.com/cloudflare/vinext), with optional Cloudflare D1 and
Drizzle support.

## 메모리 가드: 참여 동의 · 실제 케어 지표 · 테스트

이 앱(`app/page.tsx`)은 vinext 스타터 위에 구축된 "메모리 가드" 매장 안전 케어
앱입니다. 아래는 제품 리뷰 이후 추가된 부분에 대한 요약입니다.

### 참여 동의와 내 데이터 관리

- 카메라를 처음 켤 때 목적(① 기억 복원, ② 선택적 장기 인지 건강 관찰)을 밝히는
  동의 모달이 뜹니다. 동의하지 않아도 타임라인·스마트 마감 같은 매장 안전
  기능은 카메라 없이 그대로 사용할 수 있어요.
- 사이드바의 "내 데이터 관리"에서 저장된 세션 수, 케어 기록 일수, 브라우저
  저장 용량, 동의 상태를 확인하고, 관찰 참여를 철회하거나 데이터를 전부 삭제할
  수 있습니다. (`app/metrics-store.ts`, `app/pose-store.ts`)

### 실제 케어 지표 파이프라인

- `app/care-metrics.ts`: 안전 알림/마감 반복 확인/업무 누락율/미세 지연율을
  계산하고, "바쁜 날"을 제외한 개인 기준선(baseline) 대비 변화를 감지합니다.
  진단 표현 없이 `none`/`watch`/`notable` 수준과 이유만 반환합니다.
- 케어 기록 탭의 "실제 리포트"는 더 이상 하드코딩된 숫자가 아니라, 스마트
  마감·세이브포인트 등 실제 사용자 조작에서 쌓인 기록으로 계산됩니다. 기록이
  부족하면 가짜 숫자 대신 빈 상태 안내를 보여줍니다.
- 카메라 세션의 손 좌표에서 "동작 변동성/매끄러움" 참고 지표를 계산합니다
  (`app/motion-analysis.ts`). 5FPS 샘플링 특성상 실제 임상적 손 떨림(4~12Hz)은
  측정할 수 없어 검증되지 않은 참고용임을 UI에 명시합니다.
- `app/motion-detection.ts`: 스켈레톤 좌표 궤적 자체에서 마감 반복
  확인(`double_check`)·미세 지연(`micro_delay`)·안전 알림(`safety_alert`)
  패턴을 판정하는 순수 계산 엔진입니다. UI 버튼이나 타이머가 아니라 좌표의
  속도·이동 거리만으로 판정합니다:
  - `micro_delay`: 실제로 움직이던 손이 `FREEZE_MIN_DURATION_MS`(1초) 이상
    멈춰 있다가 다시 움직이면(멈추기 전후 모두 real motion 확인) 감지.
  - `double_check`: 시작 위치에서 멀어졌다가(`AWAY_DISTANCE_THRESHOLD`) 다시
    가까이 돌아온 뒤(`NEAR_DISTANCE_THRESHOLD`) 다시 멀어지는, 분리된 접근이
    반복되면 감지. (순간 속도가 아니라 시작점 기준 누적 이동 거리로 판정 —
    동작의 정점에서 속도가 0에 가까워지는 raised-cosine 곡선 특성 때문에
    순간 속도 기반 판정은 오탐이 많았습니다.)
  - `safety_alert`: 프레임 간 속도 변화(jerk)가 `SAFETY_JERK_THRESHOLD`를
    넘는 급격하고 반응적인 움직임이 있으면 감지.
  - 한 구간이 `double_check`로 감지되면 그 구간과 겹치는 `micro_delay`는
    중복 집계하지 않도록 걸러냅니다(반복 확인 동작 특성상 중간에 완전히
    멈추는 구간이 자연스럽게 생기기 때문).
  - 실제 카메라 세션이 끝나면(`page.tsx`의 `recordMotionDetections`) 저장된
    좌표를 이 엔진으로 분석해, 감지된 이벤트를 **기존 마감 확인 버튼
    재실행·온열기 타이머 기반 집계와 나란히, 추가로** 같은 카운터에
    기록합니다(대체가 아니라 합산). 관찰 동의가 있고 세션 프레임 수가
    충분할 때만 실행되는 best-effort 분석이라, 분석이 실패해도 세션 종료
    자체는 막지 않습니다.
  - `tests/motion-detection.test.ts`가 손으로 만든 궤적으로 각 판정 로직을
    검증하고, `app/demo-motion.ts`의 8가지 합성 이벤트 유형 전부를 여러
    seed에 걸쳐 이 엔진에 통과시켜 의도한 유형과 정확히 일치하는지
    확인합니다(closed-loop 검증).

### 시뮬레이션 페르소나 (4종)

새 사용자가 실제로 일주일을 기다리지 않고도 케어 리포트가 어떻게 보이는지
미리 체험할 수 있도록, 케어 기록 탭 "시뮬레이션 보기"에 서로 다른 결과를
보여주는 가상 페르소나 4명을 추가했습니다 (`app/demo-personas.ts`).

- **박준혁 사장님 (안정적)** — 평소와 비슷한 한 주 → `none`. 대부분의 주는
  알림 없이 조용히 지나간다는 걸 보여줍니다.
- **이수진 사장님 (점진적 변화)** — 주 후반에 여러 지표가 함께 오름 →
  `notable`.
- **김미래 사장님 (성수기 혼잡)** — 숫자는 오르지만 해당 날짜가 사용자가
  직접 "바쁨"으로 표시한 기간이라, 리포트에 혼란 변수 안내가 함께 뜹니다 →
  `notable` + 확인 문구.
- **정하늘 사장님 (하루의 소란)** — 하루만 튀고 다음 날 바로 회복 → `watch`
  (연속된 패턴이 아니므로 `notable`까지 올라가지 않는다는 걸 보여줌).

각 페르소나의 신호(`none`/`watch`/`notable`, 이유, 혼란 변수 안내)는 미리
써둔 문구가 아니라, 합성한 7일치 로그를 실제 사용자와 동일한
`computeBaseline`/`detectChangeSignal` 파이프라인에 통과시켜 계산한
값입니다. `tests/demo-personas.test.ts`가 각 페르소나가 의도한 결과로
계산되는지 검증합니다.

### 개발용 세션 리플레이

"정확히 어떤 움직임 때문에 이렇게 측정됐는지" 확인할 수 있도록, 저장된 좌표
세션을 스켈레톤으로 그대로 재생해보는 개발/디버그 전용 도구를 추가했습니다.

- 진입 지점 2곳: (1) 타임라인 이벤트 상세 모달에서 "이 순간이 기록된 세션
  전체 리플레이 보기" 버튼, (2) "내 데이터 관리" 모달의 "개발용 · 최근 좌표
  세션 리플레이" 목록.
- `app/pose-store.ts`의 `parseSessionFrame()`이 저장된 프레임(바이너리 레이아웃)을
  구조화된 형태로 되돌리고, `app/session-replay.tsx`(`SessionReplayPanel`)가
  재생/일시정지·배속(0.5x/1x/2x)·구간 탐색(스크러버)과 함께 라이브 카메라와
  동일한 렌더러(`app/skeleton-draw.ts`)로 스켈레톤을 그립니다.
- 구간별 "동작 변동성/매끄러움" 값을 프레임마다 다시 계산해 보여주고, 스크러버
  위에 변동성 스파크라인을 그려서 어느 프레임에서 신호가 튀었는지 바로 찾아갈
  수 있게 했습니다.
- 저장된 프레임에는 귀 좌표(얼굴 인접)가 없어 정확한 머리 중심을 복원할 수
  없습니다. 어깨 중점에서 근사한 위치이며, 이는 재생 전용 근사치입니다
  (`approximateHeadCenterFromShoulders`, `app/motion-analysis.ts`).
- 얼굴·영상·음성은 애초에 저장되지 않으므로 리플레이에도 나타나지 않습니다.

### 개발용 합성 시나리오 리플레이 (가상 페르소나 이벤트별)

위 세션 리플레이는 실제로 카메라 앞에서 움직여 기록된 좌표만 재생할 수
있습니다. 하지만 시뮬레이션 페르소나 4명은 실제 녹화가 없는 가상의 한
주이므로, "이 이벤트가 감지됐다면 물리적으로 대략 어떤 동작이었을지"를
보여주는 절차적 합성 스켈레톤 클립을 별도로 추가했습니다
(`app/demo-motion.ts`). 한 주 전체를 통째로 재생하는 게 아니라, 각 날짜에
표시된 개별 이벤트("19:06 출입문 상태 재확인" 등) 옆의 **"동작 보기"**
버튼을 누르면 그 이벤트 하나에 해당하는 짧은 클립만 재생됩니다.

- `generateEventMotion(type, seed)`가 이벤트 유형 8가지에 따라
  결정론적으로(랜덤 없이) 좌표 프레임을 생성합니다 — 같은 (유형, seed)
  조합은 항상 같은 클립을 반환하므로 재생할 때마다 같은 동작이 보입니다.
  손 하나만 움직이는 게 아니라 상체 기울임(`torsoLean`), 무릎을 굽히는
  크라우치(`crouch`), 좌우 무게 중심 이동(`weightShiftX`), 양손 동시
  사용까지 조합해서, 유형마다 몸 전체의 움직임이 눈에 띄게 다르게 보이도록
  했습니다.
  - `double_check`: 서로 다른 두 지점(손잡이 → 잠금장치)을 순서대로
    확인하고 그 사이에는 완전히 멈추는 반복 확인 동작
  - `micro_delay`: 손을 뻗다가 완전히 멈춰서(속도 0) 오래 머뭇거린 뒤,
    다른 지점으로 이어서 마무리하는 동작
  - `safety_alert`: 무릎을 굽혀 낮은 곳(콘센트 등)까지 빠르게 반응하고
    바로 일어서는 급박한 동작
  - `normal_task`: 가슴 높이에서 좌우로 오가는 평범한 반복 업무 동작
  - `register_tap`: 짧게 여러 번 두드리다가 중간에 멈춘 채(재입력·결제
    지연) 정지했다가 마지막에 한 번 더 눌러 마무리하는 동작
  - `queue_shift`: 좌우로 무게 중심을 옮기며 고개를 돌려 주변을 살피는
    대기·서성임 동작
  - `high_reach`: 양팔을 동시에 높이 들어 올려 높은 선반을 정리하는 동작
  - `low_bend`: 허리와 무릎을 굽혀 바닥 가까운 곳을 확인하는 동작
- 생성된 좌표는 `app/pose-store.ts`가 저장하는 것과 동일한 프레임 레이아웃이라
  `parseSessionFrame()`과 기존 `SessionReplayPanel` 재생기를 그대로 재사용합니다
  (`session-replay.tsx`의 `source={{kind:"synthetic", frames}}`).
- 화면 상단에 **"실제 기록이 아니에요"** 문구를 항상 표시해, 실제 녹화
  리플레이와 절대 혼동되지 않도록 했습니다.
- 이 합성 클립도 진짜 `computeHandMotionVariability`/`computeMovementSmoothness`
  분석 함수를 그대로 통과하며, `tests/demo-motion.test.ts`가 각 이벤트
  유형이 의도한 동작 신호(정지 구간, 반응성 등)를 실제로 만들어내는지
  검증합니다.

#### "왜 이 신호로 잡혔을까요?" - 실제 감지 근거 표시

동작 리플레이만 보여주면 "그래서 이게 왜 이상행동으로 잡힌 거지?"라는
질문에는 답이 되지 않습니다. 그래서 각 이벤트 리플레이 상단에 **이 클립의
좌표를 실제 앱과 동일한 동작 분석 함수(`app/motion-detection.ts`)로 분석한
진짜 결과**를 보여주는 패널을 추가했습니다 (`explainDemoEvent()`,
`app/demo-personas.ts`).

- 이 패널은 미리 써둔 설명이 아니라, 리플레이 중인 클립의 좌표를
  `detectMotionEvents()`에 실제로 통과시켜 나온 결과입니다. 마감 반복
  확인·미세 지연·안전 알림 중 무엇이 감지됐는지, 그리고 그 근거가 된
  구체적인 수치(예: "1.4초간 멈춰 있었어요", "서로 떨어진 접근이 2번
  있었어요")를 그대로 보여줍니다. 아무것도 감지되지 않은 클립(일반 업무
  동작)은 "패턴이 감지되지 않았어요"로 정직하게 표시됩니다.
- 실제 서비스에서는 이 동작 분석 결과가 기존의 마감 확인 버튼
  재실행(`recordDoubleCheck`)·온열기 켜짐(`recordSafetyAlert`)·업무 소요
  시간(`SLOW_DELAY_SECONDS`, `care-metrics.ts`) 같은 버튼/타이머 기반
  규칙과 **함께** 같은 카운터에 집계됩니다(하나가 다른 하나를 대체하지
  않음) — 패널의 규칙 설명 문구에도 두 가지 경로를 모두 명시했습니다.
- 패널은 네 부분으로 구성됩니다: (1) 이 클립의 좌표에서 실제로 감지된
  구체적 근거 문장(들), (2) 그 유형이 실제로 어떤 규칙으로 집계되는지
  (버튼/타이머 경로와 동작 분석 경로를 함께 설명), (3) 이 날의 수치를
  페르소나의 평소 하루 평균과 비교한 문장(예: "이 날 마감 반복 확인 2회
  (평소 하루 평균 0.6회)"), (4) 이 날이 실제로 그 주의 케어
  신호(`signal.reasons`)에 반영된 날인지 여부.
- `tests/demo-personas.test.ts`가 일반 업무 이벤트는 "집계 안 됨"으로
  정직하게 표시되는지, 집계되는 이벤트는 실제 기준치와 비교 문구를
  포함하는지, "이번 주 신호에 반영됨" 표시가 실제 `signal.reasons`와
  정확히 일치하는지(과장도 누락도 없이) 검증합니다.
  `tests/motion-detection.test.ts`는 감지 엔진 자체와, 이 엔진이
  `app/demo-motion.ts`의 각 합성 이벤트 유형을 의도한 대로 분류하는지를
  검증합니다.

### 선택적 서버 동기화 (기본 비활성)

- `db/schema.ts`의 `dailyCareMetrics` 테이블과 `app/api/metrics/route.ts`는
  일별 요약 지표를 위한 준비된 스캐폴딩입니다. **기본적으로 꺼져 있으며 지금은
  UI에서 호출되지 않습니다.**
- 활성화하려면: `.openai/hosting.json`의 `d1` 값을 실제 바인딩 이름(예:
  `"DB"`)으로 설정 → `npm run db:generate`로 마이그레이션 생성 → 배포. 그
  전까지 `getDb()`는 명확한 안내 메시지와 함께 에러를 반환합니다.
- 원본 pose/hand 좌표 프레임은 이 서버 동기화 대상에 포함되지 않습니다. 일별
  요약 카운트만 동기화 대상이며, 브라우저 로컬(IndexedDB) 저장이 여전히
  기본입니다.

### 테스트

- `npm run test:unit`: `app/motion-analysis.ts`, `app/care-metrics.ts`,
  `app/demo-personas.ts`, `app/pose-store.ts`(`parseSessionFrame`),
  `app/demo-motion.ts`(합성 이벤트 클립), `app/motion-detection.ts`(좌표
  기반 이상행동 감지 엔진)의 순수 계산 로직 유닛 테스트
  (브라우저 불필요, node:test + tsx)
- `npm run test:e2e`: Playwright로 실제 브라우저에서 카메라 권한 거부, 관찰
  비동의, 바쁜 날 지표 기록, 세션 누적, 데이터 전체 삭제, 접근성(글자 크기),
  시뮬레이션 페르소나 전환, 세션 리플레이, 합성 시나리오 리플레이("동작 보기"),
  GPU 미가속 폴백 등 페르소나별 시나리오를 검증합니다. 최초 1회
  `npx playwright install chromium` 필요. MediaPipe WASM
  모델 로딩 때문에 전체 실행에 몇 분 걸릴 수 있습니다(멈춘 게 아니에요).
- `npm test`: 빌드 + 서버 렌더링 스모크 테스트 + 유닛 테스트를 한 번에
  실행합니다 (`test:e2e`는 시간이 걸려 별도 명령으로 분리했습니다).

## Prerequisites

- Node.js `>=22.13.0`

## Quick Start

```bash
npm install
npm run dev
npm run build
```

This starter does not use `wrangler.jsonc`.

## Included Shape

- edit site code under `app/`
- `.openai/hosting.json` declares optional Sites D1 and R2 bindings
- `vite.config.ts` simulates declared bindings for local development
- `db/schema.ts` starts intentionally empty
- `examples/d1/` contains an optional D1 example surface
- `drizzle.config.ts` supports local migration generation when needed

## Workspace Auth Headers

Signed-in visitors receive both `oai-authenticated-user-id` and `oai-authenticated-user-email`. Private Sites require every visitor to sign in; public Sites may also have anonymous visitors, for whom neither header is present.

The user ID is stable for the same user on the same Site and different across Sites. Email and name are intended for display or contact purposes.

SIWC-authenticated workspace sites may also receive
`oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty
`name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by
`oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const userId = requestHeaders.get("oai-authenticated-user-id");
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Optional Dispatch-Owned ChatGPT Sign-In

Import the ready-to-use helpers from `app/chatgpt-auth.ts` when the site needs
optional or required ChatGPT sign-in:

- Use `getChatGPTUser()` for optional signed-in UI.
- Use `requireChatGPTUser(returnTo)` for server-rendered pages that should send
  anonymous visitors through Sign in with ChatGPT.
- Use `chatGPTSignInPath(returnTo)` and `chatGPTSignOutPath(returnTo)` for
  browser links or actions.
- Pass a same-origin relative `returnTo` path for the destination after sign-in
  or sign-out. The helper validates and safely encodes it.
- Mark protected pages with `export const dynamic = "force-dynamic"` because
  they depend on per-request identity headers.

Dispatch owns `/signin-with-chatgpt`, `/signout-with-chatgpt`, `/callback`, the
OAuth cookies, and identity header injection. Do not implement app routes for
those reserved paths. Routes that do not import and call the helper remain
anonymous-compatible.

SIWC establishes identity only; it does not prove workspace membership. Use the
Sites hosting platform's access policy controls for workspace-wide restrictions,
or enforce explicit server-side membership or allowlist checks.

Use SIWC for account pages, user-specific dashboards, saved records, and write
actions tied to the current ChatGPT user. Leave public content anonymous.

## Useful Commands

- `npm run dev`: start local development
- `npm run build`: verify the vinext build output
- `npm test`: build, verify server-rendered output, and run unit tests
- `npm run test:unit`: fast unit tests for the pure motion/care-metrics logic
- `npm run test:e2e`: Playwright persona tests (needs `npx playwright install
  chromium` once; see "메모리 가드" section above)
- `npm run db:generate`: generate Drizzle migrations after schema changes

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)
