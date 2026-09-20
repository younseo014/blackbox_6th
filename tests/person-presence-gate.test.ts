import assert from "node:assert/strict";
import test from "node:test";
import {
  EMPTY_BACKGROUND_GATE,
  personPresenceAllowsAcquisition,
  updateBackgroundGate,
} from "../app/person-presence-gate.ts";

function pixels(value: number) {
  return new Float32Array(32 * 18).fill(value);
}

test("quiet person-free frames establish a background without reporting a person", () => {
  let state = updateBackgroundGate(EMPTY_BACKGROUND_GATE, pixels(80), 100, true);
  state = updateBackgroundGate(state, pixels(80), 500, true);
  state = updateBackgroundGate(state, pixels(80), 900, true);
  assert.equal(state.learnedFrames, 3);
  assert.equal(personPresenceAllowsAcquisition({
    timestamp: 900,
    lastPersonDetectionAt: 0,
    background: state,
  }), false);
});

test("foreground change can admit a headless moving body after the room is learned", () => {
  let state = updateBackgroundGate(EMPTY_BACKGROUND_GATE, pixels(80), 100, true);
  state = updateBackgroundGate(state, pixels(80), 500, true);
  state = updateBackgroundGate(state, pixels(80), 900, true);
  state = updateBackgroundGate(state, pixels(130), 1300, true);
  assert.ok(state.foregroundRatio > 0.9);
  assert.equal(personPresenceAllowsAcquisition({
    timestamp: 1400,
    lastPersonDetectionAt: 0,
    background: state,
  }), true);
});

test("a large early scene change is not absorbed as background", () => {
  let state = updateBackgroundGate(EMPTY_BACKGROUND_GATE, pixels(60), 100, true);
  state = updateBackgroundGate(state, pixels(130), 500, true);
  assert.equal(state.learnedFrames, 1);
  assert.equal(personPresenceAllowsAcquisition({
    timestamp: 600,
    lastPersonDetectionAt: 0,
    background: state,
  }), true);
});

test("a recent person-detector hit admits a stationary person", () => {
  assert.equal(personPresenceAllowsAcquisition({
    timestamp: 2000,
    lastPersonDetectionAt: 1500,
    background: EMPTY_BACKGROUND_GATE,
  }), true);
  assert.equal(personPresenceAllowsAcquisition({
    timestamp: 3000,
    lastPersonDetectionAt: 1500,
    background: EMPTY_BACKGROUND_GATE,
  }), false);
});

test("a detected person is never learned into a new empty-room baseline", () => {
  const state = updateBackgroundGate(EMPTY_BACKGROUND_GATE, pixels(120), 100, false);
  assert.equal(state.pixels, null);
  assert.equal(state.learnedFrames, 0);
});
