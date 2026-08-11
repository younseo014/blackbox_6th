"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type View = "today" | "timeline" | "closing" | "care";
type CameraStatus = "idle" | "requesting" | "connected" | "error";
type ClosingStatus = "idle" | "checking" | "attention" | "done";
type EventKind = "payment" | "door" | "safety" | "booking";

type TimelineEvent = {
  id: string;
  time: string;
  title: string;
  detail: string;
  kind: EventKind;
  clipUrl?: string;
  poster?: string;
  isCapturing?: boolean;
};

const initialEvents: TimelineEvent[] = [
  {
    id: "sample-1",
    time: "16:30",
    title: "손님이 나가셨어요",
    detail: "출입문 움직임 감지",
    kind: "door",
  },
  {
    id: "sample-2",
    time: "14:05",
    title: "결제가 완료됐어요",
    detail: "카드 결제 · 45,000원",
    kind: "payment",
  },
  {
    id: "sample-3",
    time: "13:52",
    title: "예약을 등록했어요",
    detail: "김하나 고객 · 커트",
    kind: "booking",
  },
  {
    id: "sample-4",
    time: "09:42",
    title: "매장 문을 열었어요",
    detail: "출입문 센서 감지",
    kind: "door",
  },
];

const navItems: Array<{ id: View; label: string; icon: string }> = [
  { id: "today", label: "오늘", icon: "⌂" },
  { id: "timeline", label: "타임라인", icon: "≡" },
  { id: "closing", label: "스마트 마감", icon: "✓" },
  { id: "care", label: "케어 기록", icon: "♡" },
];

const eventPresets: Array<{
  title: string;
  detail: string;
  kind: EventKind;
  button: string;
}> = [
  {
    title: "결제가 완료됐어요",
    detail: "로컬 카메라 테스트 · 결제 이벤트",
    kind: "payment",
    button: "결제 완료",
  },
  {
    title: "손님이 나가셨어요",
    detail: "로컬 카메라 테스트 · 퇴장 이벤트",
    kind: "door",
    button: "손님 퇴장",
  },
  {
    title: "출입문이 열렸어요",
    detail: "로컬 카메라 테스트 · 출입 이벤트",
    kind: "door",
    button: "출입문 열림",
  },
];

const kindLabel: Record<EventKind, string> = {
  payment: "결제",
  door: "출입",
  safety: "안전",
  booking: "예약",
};

function currentTime() {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date());
}

function TimelineList({
  events,
  onSelect,
  compact = false,
}: {
  events: TimelineEvent[];
  onSelect: (event: TimelineEvent) => void;
  compact?: boolean;
}) {
  const visibleEvents = compact ? events.slice(0, 4) : events;

  return (
    <div className="timeline-list">
      {visibleEvents.map((event, index) => (
        <button
          className="timeline-event"
          key={event.id}
          onClick={() => onSelect(event)}
          type="button"
        >
          <span className="event-time">{event.time}</span>
          <span className={`event-node ${event.kind}`} aria-hidden="true">
            {event.kind === "safety" ? "✓" : ""}
          </span>
          {index < visibleEvents.length - 1 && (
            <span className="event-line" aria-hidden="true" />
          )}
          <span className="event-copy">
            <span className="event-title-row">
              <strong>{event.title}</strong>
              {event.clipUrl && <span className="video-tag">영상</span>}
              {event.isCapturing && (
                <span className="saving-tag">저장 중</span>
              )}
            </span>
            <span>{event.detail}</span>
          </span>
          <span className="event-chevron" aria-hidden="true">
            ›
          </span>
        </button>
      ))}
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("today");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [cameraMessage, setCameraMessage] = useState(
    "카메라를 연결하면 오늘의 장면을 확인할 수 있어요.",
  );
  const [events, setEvents] = useState<TimelineEvent[]>(initialEvents);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [toast, setToast] = useState("");
  const [heaterOn, setHeaterOn] = useState(true);
  const [closingStatus, setClosingStatus] = useState<ClosingStatus>("idle");
  const [closingStep, setClosingStep] = useState(0);
  const [savepointOpen, setSavepointOpen] = useState(true);
  const [bookingOpen, setBookingOpen] = useState(false);
  const [bookingName, setBookingName] = useState("김하나");
  const [bookingService, setBookingService] = useState("커트");

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlsRef = useRef<string[]>([]);

  const todayEvents = useMemo(() => events, [events]);

  useEffect(() => {
    if (cameraStatus === "connected" && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => undefined);
    }
  }, [cameraStatus, view]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const objectUrls = objectUrlsRef.current;
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedEvent(null);
        setBookingOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("error");
      setCameraMessage("이 브라우저에서는 카메라를 사용할 수 없어요.");
      return;
    }

    setCameraStatus("requesting");
    setCameraMessage("카메라 연결을 기다리고 있어요…");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraStatus("connected");
      setCameraMessage("노트북 카메라가 안전하게 연결됐어요.");
      setToast("카메라가 연결됐어요");
    } catch {
      setCameraStatus("error");
      setCameraMessage(
        "카메라 권한을 허용한 뒤 다시 연결해 주세요. 영상은 외부로 전송되지 않아요.",
      );
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraStatus("idle");
    setCameraMessage("카메라 연결을 멈췄어요.");
    setToast("카메라 연결을 종료했어요");
  }

  function capturePoster() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return undefined;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return undefined;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.78);
  }

  function recordTestEvent(preset: (typeof eventPresets)[number]) {
    const stream = streamRef.current;
    if (!stream || cameraStatus !== "connected") {
      setToast("먼저 카메라를 연결해 주세요");
      return;
    }
    if (!window.MediaRecorder) {
      setToast("이 브라우저는 영상 기록을 지원하지 않아요");
      return;
    }

    const id = `camera-${crypto.randomUUID()}`;
    const poster = capturePoster();
    const event: TimelineEvent = {
      id,
      time: currentTime(),
      title: preset.title,
      detail: `${preset.detail} · 4초 장면`,
      kind: preset.kind,
      poster,
      isCapturing: true,
    };

    setEvents((previous) => [event, ...previous]);
    setIsCapturing(true);
    setToast("현재 장면을 4초 동안 기록하고 있어요");

    const chunks: Blob[] = [];
    let recorder: MediaRecorder;
    const preferredTypes = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/mp4",
      "video/webm",
    ];
    const mimeType = preferredTypes.find((type) =>
      MediaRecorder.isTypeSupported(type),
    );

    try {
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
    } catch {
      setEvents((previous) => previous.filter((item) => item.id !== id));
      setIsCapturing(false);
      setToast("영상 기록을 시작하지 못했어요");
      return;
    }

    recorder.ondataavailable = (clipEvent) => {
      if (clipEvent.data.size > 0) chunks.push(clipEvent.data);
    };

    recorder.onerror = () => {
      setEvents((previous) => previous.filter((item) => item.id !== id));
      setIsCapturing(false);
      setToast("영상 기록 중 문제가 생겼어요");
    };

    recorder.onstop = () => {
      if (chunks.length === 0) {
        setEvents((previous) => previous.filter((item) => item.id !== id));
        setToast("기록된 영상이 없어 다시 시도해 주세요");
        setIsCapturing(false);
        return;
      }
      const blob = new Blob(chunks, {
        type: recorder.mimeType || chunks[0].type || "video/webm",
      });
      const clipUrl = URL.createObjectURL(blob);
      objectUrlsRef.current.push(clipUrl);
      setEvents((previous) =>
        previous.map((item) =>
          item.id === id ? { ...item, clipUrl, isCapturing: false } : item,
        ),
      );
      setIsCapturing(false);
      setToast("장면이 메모리 타임라인에 저장됐어요");
    };

    recorder.start();
    window.setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, 4000);
  }

  function startClosingCheck() {
    setClosingStatus("checking");
    setClosingStep(0);
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      setClosingStep(step);
      if (step >= 3) {
        window.clearInterval(timer);
        if (heaterOn) {
          setClosingStatus("attention");
        } else {
          completeClosing();
        }
      }
    }, 650);
  }

  function completeClosing() {
    setClosingStatus("done");
    setClosingStep(3);
    setEvents((previous) => {
      if (previous[0]?.title === "오늘의 마감이 완료됐어요") return previous;
      return [
        {
          id: `closing-${crypto.randomUUID()}`,
          time: currentTime(),
          title: "오늘의 마감이 완료됐어요",
          detail: "가스 · 전기 · 출입문 모두 안전",
          kind: "safety",
        },
        ...previous,
      ];
    });
    setToast("모든 항목이 안전해요. 편안히 퇴근하세요");
  }

  function turnOffHeater() {
    setHeaterOn(false);
    setClosingStatus("idle");
    setToast("온열기 전원을 차단했어요");
  }

  function saveBooking(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setEvents((previous) => [
      {
        id: `booking-${crypto.randomUUID()}`,
        time: currentTime(),
        title: "예약을 등록했어요",
        detail: `${bookingName} 고객 · ${bookingService}`,
        kind: "booking",
      },
      ...previous,
    ]);
    setSavepointOpen(false);
    setBookingOpen(false);
    setToast("하던 업무를 이어서 완료했어요");
  }

  function openTimelineEvent(event: TimelineEvent) {
    if (event.isCapturing) {
      setToast("영상을 저장하고 있어요. 잠시만 기다려 주세요");
      return;
    }
    setSelectedEvent(event);
  }

  const statusText =
    cameraStatus === "connected"
      ? "카메라 연결됨"
      : cameraStatus === "requesting"
        ? "연결 중"
        : "카메라 대기";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand" aria-label="메모리 가드">
          <span className="brand-mark" aria-hidden="true">
            M
          </span>
          <span>
            <strong>메모리 가드</strong>
            <small>Memory Guard</small>
          </span>
        </div>

        <nav className="main-nav" aria-label="주요 메뉴">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "active" : ""}
              onClick={() => setView(item.id)}
              type="button"
              aria-current={view === item.id ? "page" : undefined}
            >
              <span aria-hidden="true">{item.icon}</span>
              {item.label}
              {item.id === "closing" && heaterOn && (
                <i aria-label="확인할 항목 1개">1</i>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-care">
          <span className="care-sprout" aria-hidden="true">
            ♡
          </span>
          <p>사장님의 하루를</p>
          <strong>조용히 지켜드릴게요.</strong>
        </div>
        <div className="profile">
          <span className="avatar">김</span>
          <span>
            <strong>김메모리 사장님</strong>
            <small>오늘도 좋은 하루예요</small>
          </span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">8월 11일 화요일</span>
            <h1>
              {view === "today" && "안녕하세요, 사장님"}
              {view === "timeline" && "오늘의 메모리 타임라인"}
              {view === "closing" && "스마트 마감"}
              {view === "care" && "나의 케어 기록"}
            </h1>
          </div>
          <div className={`camera-pill ${cameraStatus}`}>
            <span aria-hidden="true" />
            {statusText}
          </div>
        </header>

        {view === "today" && (
          <div className="today-view">
            <section className="safety-banner">
              <div className="safety-check" aria-hidden="true">
                ✓
              </div>
              <div>
                <strong>매장은 대체로 안전해요</strong>
                <p>온열기 한 가지만 확인하면 마음 놓고 마감할 수 있어요.</p>
              </div>
              <button type="button" onClick={() => setView("closing")}>
                확인하기 <span aria-hidden="true">›</span>
              </button>
            </section>

            <div className="dashboard-grid">
              <section className="panel camera-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">기억 복원 카메라</span>
                    <h2>지금 매장 모습</h2>
                  </div>
                  {cameraStatus === "connected" && (
                    <span className="live-badge">
                      <i aria-hidden="true" /> LIVE
                    </span>
                  )}
                </div>

                <div className={`camera-frame ${cameraStatus}`}>
                  <video ref={videoRef} muted playsInline aria-label="실시간 카메라 영상" />
                  {cameraStatus !== "connected" && (
                    <div className="camera-empty">
                      <span className="camera-symbol" aria-hidden="true">
                        ●
                      </span>
                      <strong>
                        {cameraStatus === "requesting"
                          ? "카메라를 연결하고 있어요"
                          : "노트북 카메라를 연결해 주세요"}
                      </strong>
                      <p>{cameraMessage}</p>
                      <button
                        className="primary-button"
                        type="button"
                        onClick={startCamera}
                        disabled={cameraStatus === "requesting"}
                      >
                        {cameraStatus === "requesting" ? "연결 중…" : "카메라 연결"}
                      </button>
                    </div>
                  )}
                  {cameraStatus === "connected" && (
                    <div className="camera-caption">
                      <span>{cameraMessage}</span>
                      <button type="button" onClick={stopCamera}>
                        카메라 끄기
                      </button>
                    </div>
                  )}
                </div>

                <div className="camera-test">
                  <div>
                    <strong>이벤트 연동 테스트</strong>
                    <p>아래 버튼을 누르면 현재 장면 4초가 타임라인에 저장돼요.</p>
                  </div>
                  <div className="test-buttons">
                    {eventPresets.map((preset) => (
                      <button
                        type="button"
                        key={preset.button}
                        disabled={isCapturing}
                        onClick={() => recordTestEvent(preset)}
                      >
                        {preset.button}
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              <section className="panel timeline-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">메모리 타임라인</span>
                    <h2>오늘 무슨 일이 있었나요?</h2>
                  </div>
                  <button
                    className="text-button"
                    type="button"
                    onClick={() => setView("timeline")}
                  >
                    모두 보기
                  </button>
                </div>
                <TimelineList
                  events={todayEvents}
                  onSelect={openTimelineEvent}
                  compact
                />
              </section>
            </div>

            {savepointOpen && (
              <section className="savepoint-card">
                <div className="savepoint-icon" aria-hidden="true">
                  ↗
                </div>
                <div>
                  <span className="section-kicker">업무 세이브포인트</span>
                  <h2>아까 하던 예약 입력이 남아 있어요</h2>
                  <p>김하나 고객 · 커트 · 예약 시간 확인 단계</p>
                </div>
                <button type="button" onClick={() => setBookingOpen(true)}>
                  이어서 하기 <span aria-hidden="true">›</span>
                </button>
              </section>
            )}
          </div>
        )}

        {view === "timeline" && (
          <div className="subpage timeline-page">
            <div className="subpage-intro">
              <div>
                <span className="section-kicker">하루의 기억을 한눈에</span>
                <p>중요한 순간만 모았어요. 영상을 저장한 항목은 바로 다시 볼 수 있어요.</p>
              </div>
              <span className="count-chip">오늘 {events.length}개 기록</span>
            </div>
            <section className="panel full-timeline-panel">
              <TimelineList events={events} onSelect={openTimelineEvent} />
            </section>
          </div>
        )}

        {view === "closing" && (
          <div className="subpage closing-page">
            <section className={`closing-hero ${closingStatus}`}>
              <div className="closing-hero-icon" aria-hidden="true">
                {closingStatus === "done" ? "✓" : closingStatus === "attention" ? "!" : "⌁"}
              </div>
              <div>
                <span className="section-kicker">퇴근 전 자동 점검</span>
                <h2>
                  {closingStatus === "done"
                    ? "오늘의 마감이 끝났어요"
                    : closingStatus === "checking"
                      ? "매장을 하나씩 확인하고 있어요"
                      : closingStatus === "attention"
                        ? "온열기 전원을 확인해 주세요"
                        : "마지막으로 안전을 확인할까요?"}
                </h2>
                <p>
                  {closingStatus === "done"
                    ? "가스, 전기, 출입문 모두 안전해요. 편안히 퇴근하세요."
                    : "연결된 센서가 가스, 전기, 출입문 상태를 확인해 드려요."}
                </p>
              </div>
            </section>

            <section className="panel closing-card">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">연결된 안전 기기</span>
                  <h2>마감 확인 항목</h2>
                </div>
                <span className="device-count">3개 연결됨</span>
              </div>

              <div className="device-list">
                <div className={`device-row ${closingStep >= 1 || closingStatus === "idle" ? "checked" : ""}`}>
                  <span className="device-icon gas" aria-hidden="true">G</span>
                  <span className="device-copy">
                    <strong>가스 밸브</strong>
                    <small>주방 가스 차단기</small>
                  </span>
                  <span className="device-status safe"><i /> 닫혀 있어요</span>
                </div>
                <div className={`device-row ${!heaterOn ? "checked" : "warning"}`}>
                  <span className="device-icon power" aria-hidden="true">P</span>
                  <span className="device-copy">
                    <strong>온열기 전원</strong>
                    <small>카운터 스마트 플러그</small>
                  </span>
                  {heaterOn ? (
                    <button className="device-action" type="button" onClick={turnOffHeater}>
                      전원 끄기
                    </button>
                  ) : (
                    <span className="device-status safe"><i /> 꺼져 있어요</span>
                  )}
                </div>
                <div className={`device-row ${closingStep >= 3 || closingStatus === "idle" ? "checked" : ""}`}>
                  <span className="device-icon lock" aria-hidden="true">D</span>
                  <span className="device-copy">
                    <strong>출입문</strong>
                    <small>정문 도어락 센서</small>
                  </span>
                  <span className="device-status safe"><i /> 잠겨 있어요</span>
                </div>
              </div>

              <button
                className="closing-button"
                type="button"
                onClick={startClosingCheck}
                disabled={closingStatus === "checking" || closingStatus === "done"}
              >
                {closingStatus === "checking"
                  ? `안전 확인 중 ${closingStep}/3`
                  : closingStatus === "done"
                    ? "마감 완료"
                    : "퇴근 전 자동 점검"}
              </button>
            </section>
          </div>
        )}

        {view === "care" && (
          <div className="subpage care-page">
            <section className="care-summary">
              <span className="care-summary-mark" aria-hidden="true">♡</span>
              <div>
                <span className="section-kicker">최근 2주 케어 리포트</span>
                <h2>대체로 평소와 비슷한 흐름이에요</h2>
                <p>가스·전기 차단 알림이 3회 있었어요. 이번 주말은 푹 쉬면서 컨디션을 관리해 보세요.</p>
              </div>
            </section>

            <div className="metric-grid">
              <article className="metric-card">
                <span>안전 알림</span>
                <strong>3<small>회</small></strong>
                <p><i className="up">↑ 1회</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>마감 반복 확인</span>
                <strong>2<small>회</small></strong>
                <p><i>— 같음</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>업무 누락</span>
                <strong>1<small>건</small></strong>
                <p><i className="down">↓ 1건</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>업무 흐름</span>
                <strong className="word-value">평소와 비슷</strong>
                <p><i className="steady">● 안정</i> 일상 업무 소요 시간</p>
              </article>
            </div>

            <section className="panel care-note">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">따뜻한 한마디</span>
                  <h2>이번 주의 케어 메모</h2>
                </div>
                <span>8월 11일</span>
              </div>
              <p>바쁜 날에는 마감 확인이 조금 늘었지만, 업무 흐름은 평소와 비슷했어요. 충분히 쉬는 것만으로도 다음 주가 한결 가벼워질 거예요.</p>
            </section>
          </div>
        )}
      </section>

      <nav className="mobile-nav" aria-label="모바일 주요 메뉴">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "active" : ""}
            onClick={() => setView(item.id)}
            type="button"
          >
            <span aria-hidden="true">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {selectedEvent && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSelectedEvent(null)}>
          <section
            className="video-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="video-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button className="modal-close" type="button" onClick={() => setSelectedEvent(null)} aria-label="닫기">×</button>
            <div className="modal-heading">
              <span className={`event-type ${selectedEvent.kind}`}>{kindLabel[selectedEvent.kind]}</span>
              <span>{selectedEvent.time}</span>
              <h2 id="video-title">{selectedEvent.title}</h2>
              <p>{selectedEvent.detail}</p>
            </div>
            <div className="playback-frame">
              {selectedEvent.clipUrl ? (
                <video src={selectedEvent.clipUrl} poster={selectedEvent.poster} controls autoPlay playsInline />
              ) : (
                <div className="no-clip">
                  <span className="camera-symbol" aria-hidden="true">●</span>
                  <strong>이 기록은 예시 타임라인이에요</strong>
                  <p>카메라를 연결하고 이벤트 테스트를 하면 이곳에서 실제 장면을 다시 볼 수 있어요.</p>
                </div>
              )}
            </div>
            <button className="modal-confirm" type="button" onClick={() => setSelectedEvent(null)}>확인했어요</button>
          </section>
        </div>
      )}

      {bookingOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setBookingOpen(false)}>
          <section
            className="booking-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="booking-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button className="modal-close" type="button" onClick={() => setBookingOpen(false)} aria-label="닫기">×</button>
            <span className="section-kicker">아까 하던 업무</span>
            <h2 id="booking-title">예약 입력을 마무리할까요?</h2>
            <p>작성했던 내용은 그대로 임시 저장되어 있어요.</p>
            <form onSubmit={saveBooking}>
              <label>
                고객 이름
                <input value={bookingName} onChange={(event) => setBookingName(event.target.value)} required />
              </label>
              <label>
                서비스
                <select value={bookingService} onChange={(event) => setBookingService(event.target.value)}>
                  <option>커트</option>
                  <option>염색</option>
                  <option>펌</option>
                </select>
              </label>
              <label>
                예약 시간
                <input value="8월 14일 오후 3:00" readOnly />
              </label>
              <button type="submit">예약 입력 완료</button>
            </form>
          </section>
        </div>
      )}

      {toast && (
        <div className="toast" role="status">
          <span aria-hidden="true">✓</span> {toast}
        </div>
      )}
    </main>
  );
}
