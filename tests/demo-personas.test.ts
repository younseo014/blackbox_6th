import assert from "node:assert/strict";
import test from "node:test";
import { DEMO_PERSONAS, explainDemoEvent, signalLevelLabel, type DemoEvent } from "../app/demo-personas.ts";
import { generateEventMotion } from "../app/demo-motion.ts";

// explainDemoEvent now runs the real motion detector against the clip's
// generated coordinates, so tests need actual frames. The seed doesn't
// affect which pattern gets detected (verified across seeds in
// tests/motion-detection.test.ts), so a fixed seed is fine here.
function framesFor(event: DemoEvent) {
  return generateEventMotion(event.motionType, 0);
}

// These personas exist so a brand-new user can preview what the care report
// looks like without waiting a week for real data. The key property under
// test isn't the exact copy - it's that the signal shown for each persona is
// actually *computed* by the real care-metrics pipeline from synthetic daily
// logs, and that the four personas land on four different, intentional
// outcomes (see app/demo-personas.ts for the rationale).

function findPersona(id: string) {
  const persona = DEMO_PERSONAS.find((p) => p.id === id);
  assert.ok(persona, `expected a demo persona with id "${id}"`);
  return persona!;
}

test("demo personas: exactly 4 personas, each with a full 7-day week", () => {
  assert.equal(DEMO_PERSONAS.length, 4);
  for (const persona of DEMO_PERSONAS) {
    assert.equal(persona.week.length, 7, `${persona.id} should have 7 days`);
    assert.ok(persona.insights.length >= 1, `${persona.id} should have at least one insight`);
    assert.ok(persona.note.length > 0, `${persona.id} should have a closing note`);
  }
});

test("persona 'stable': a quiet week produces 'none', no confound note", () => {
  const persona = findPersona("stable");
  assert.equal(persona.signal.level, "none");
  assert.equal(persona.signal.confoundNote, null);
});

test("persona 'decline': several indicators rising together produces 'notable'", () => {
  const persona = findPersona("decline");
  assert.equal(persona.signal.level, "notable");
  assert.ok(persona.signal.reasons.length >= 3, "expected multiple corroborating reasons");
  for (const reason of persona.signal.reasons) {
    assert.doesNotMatch(reason, /치매|진단|질환/, "must never use a diagnostic label");
  }
});

test("persona 'busy-season': elevated numbers during self-tagged busy days come with a confound note", () => {
  const persona = findPersona("busy-season");
  assert.ok(["watch", "notable"].includes(persona.signal.level));
  assert.ok(persona.signal.confoundNote, "busy days should surface a confound note instead of a bare alarm");
  // The two most recent days in this persona's week are both tagged busy.
  const recentDays = persona.week.slice(-2);
  assert.ok(recentDays.every((day) => day.busy), "the recent window for this persona should be self-tagged busy");
});

test("persona 'single-spike': one noisy day (not a persistent pattern) stays at 'watch', not 'notable'", () => {
  const persona = findPersona("single-spike");
  assert.equal(persona.signal.level, "watch");
  assert.equal(persona.signal.confoundNote, null);
});

test("signalLevelLabel: returns a distinct, non-empty label for every level", () => {
  const levels = ["none", "watch", "notable"] as const;
  const labels = levels.map(signalLevelLabel);
  assert.equal(new Set(labels).size, 3, "each level should have a distinct label");
  for (const label of labels) assert.ok(label.length > 0);
});

// -- explainDemoEvent: the "왜 이 신호로 잡혔을까요?" detection basis --------
//
// The important thing under test isn't the exact Korean copy - it's that
// (a) routine events (normal_task/queue_shift/high_reach/low_bend) are
// honestly labeled as not counted, (b) events that ARE counted cite a real
// rule and a real baseline comparison, and (c) "this contributed to the
// week's signal" is only ever claimed when it's actually true.

test("explainDemoEvent: a routine (uncounted) event is labeled as not contributing", () => {
  const persona = findPersona("stable");
  const day = persona.week[0];
  const routineEvent = day.examples.find(
    (event) => !["double_check", "safety_alert", "micro_delay", "register_tap"].includes(event.motionType),
  );
  assert.ok(routineEvent, "expected at least one routine example on this day");
  const explanation = explainDemoEvent(persona, 0, day, routineEvent, framesFor(routineEvent));
  assert.equal(explanation.metricLine, null);
  assert.equal(explanation.contributesToSignal, false);
  assert.match(explanation.rule, /집계되지 않는 일반 업무 장면이에요/);
});

test("explainDemoEvent: a counted event cites its baseline comparison", () => {
  const persona = findPersona("decline");
  const dayIndex = 1; // 화 - inside the baseline window, not the "recent" window
  const day = persona.week[dayIndex];
  const doubleCheckEvent = day.examples.find((event) => event.motionType === "double_check");
  assert.ok(doubleCheckEvent, "expected a double_check example on this day");
  const explanation = explainDemoEvent(persona, dayIndex, day, doubleCheckEvent, framesFor(doubleCheckEvent));
  assert.match(explanation.rule, /마감 반복 확인/);
  assert.ok(explanation.metricLine && /마감 반복 확인/.test(explanation.metricLine));
  // This day is part of the baseline itself, not the recent window used to
  // compute signal.reasons, so it can never be marked as contributing.
  assert.equal(explanation.contributesToSignal, false);
});

test("explainDemoEvent: contributesToSignal exactly matches whether this recent day's reason is present", () => {
  const persona = findPersona("decline");
  const dayIndex = 5; // 토 - one of the two "recent" days behind signal.reasons
  const day = persona.week[dayIndex];
  const safetyEvent = day.examples.find((event) => event.motionType === "safety_alert");
  assert.ok(safetyEvent, "expected a safety_alert example on this day");
  const explanation = explainDemoEvent(persona, dayIndex, day, safetyEvent, framesFor(safetyEvent));
  const expectedFlag = persona.signal.reasons.includes("최근 안전 알림 빈도가 평소보다 늘었어요.");
  assert.equal(explanation.contributesToSignal, expectedFlag);
});

test("explainDemoEvent: every example across every persona/day produces a well-formed explanation", () => {
  for (const persona of DEMO_PERSONAS) {
    persona.week.forEach((day, dayIndex) => {
      for (const event of day.examples) {
        const explanation = explainDemoEvent(persona, dayIndex, day, event, framesFor(event));
        assert.ok(explanation.rule.length > 0, `${persona.id} day ${dayIndex} "${event.label}" needs a rule`);
        assert.ok(
          explanation.metricLine === null || explanation.metricLine.length > 0,
          `${persona.id} day ${dayIndex} "${event.label}" metricLine must be null or non-empty`,
        );
        assert.equal(typeof explanation.contributesToSignal, "boolean");
      }
    });
  }
});
