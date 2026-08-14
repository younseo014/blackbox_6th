// Shared canvas skeleton-drawing helpers, extracted from page.tsx so both
// the live camera overlay/snapshot view and the recorded-session replay
// viewer (session-replay.tsx) can reuse the exact same rendering code -
// what you see live and what you see in a replay are drawn identically.

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { HeadDirection } from "./motion-analysis";
import { HAND_LANDMARK_COUNT } from "./pose-store";

export type MotionSnapshot = {
  body: number[];
  leftHand: number[] | null;
  rightHand: number[] | null;
  head: HeadDirection;
};

export const BODY_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19],
  [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20],
  [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25],
  [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31],
  [30, 32], [27, 31], [28, 32],
];

export const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

export function drawMotionSkeleton(
  canvas: HTMLCanvasElement,
  body: NormalizedLandmark[] | number[],
  leftHand: NormalizedLandmark[] | number[] | null,
  rightHand: NormalizedLandmark[] | number[] | null,
  head: HeadDirection | null,
  fullBody: boolean,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);

  const bodyPoint = (index: number) => {
    if (typeof body[0] === "number") {
      const flat = body as number[];
      const storedIndex = index - 11;
      return {
        x: flat[storedIndex * 4],
        y: flat[storedIndex * 4 + 1],
        z: flat[storedIndex * 4 + 2],
        visibility: flat[storedIndex * 4 + 3],
      };
    }
    return (body as NormalizedLandmark[])[index];
  };

  context.lineWidth = Math.max(3, canvas.width / 180);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = fullBody ? "#6ff0b7" : "#ffc36e";
  context.shadowColor = "rgba(8, 38, 32, 0.75)";
  context.shadowBlur = 8;

  for (const [from, to] of BODY_CONNECTIONS) {
    const start = bodyPoint(from);
    const end = bodyPoint(to);
    if (!start || !end || start.visibility < 0.35 || end.visibility < 0.35) continue;
    context.beginPath();
    context.moveTo(start.x * canvas.width, start.y * canvas.height);
    context.lineTo(end.x * canvas.width, end.y * canvas.height);
    context.stroke();
  }

  context.shadowBlur = 5;
  for (let index = 11; index < 33; index += 1) {
    const landmark = bodyPoint(index);
    if (!landmark || landmark.visibility < 0.35) continue;
    context.beginPath();
    context.arc(
      landmark.x * canvas.width,
      landmark.y * canvas.height,
      Math.max(3, canvas.width / 150),
      0,
      Math.PI * 2,
    );
    context.fillStyle = fullBody ? "#b9ffdc" : "#ffe0aa";
    context.fill();
  }

  if (head) {
    const leftShoulder = bodyPoint(11);
    const rightShoulder = bodyPoint(12);
    const shoulderWidth = leftShoulder && rightShoulder
      ? Math.abs(leftShoulder.x - rightShoulder.x) * canvas.width
      : canvas.width * 0.12;
    const radius = Math.max(14, shoulderWidth * 0.28);
    const centerX = head.centerX * canvas.width;
    const centerY = head.centerY * canvas.height;
    context.shadowBlur = 8;
    context.strokeStyle = "#ffffff";
    context.lineWidth = Math.max(3, canvas.width / 190);
    context.beginPath();
    context.arc(centerX, centerY, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(centerX, centerY);
    context.lineTo(
      centerX + head.yaw * radius * 1.25,
      centerY + head.pitch * radius * 1.25,
    );
    context.stroke();
  }

  const drawHand = (
    hand: NormalizedLandmark[] | number[] | null,
    color: string,
  ) => {
    if (!hand) return;
    const handPoint = (index: number) => {
      if (typeof hand[0] === "number") {
        const flat = hand as number[];
        return {
          x: flat[index * 3],
          y: flat[index * 3 + 1],
          z: flat[index * 3 + 2],
        };
      }
      return (hand as NormalizedLandmark[])[index];
    };
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = Math.max(2, canvas.width / 260);
    context.shadowBlur = 6;
    for (const [from, to] of HAND_CONNECTIONS) {
      const start = handPoint(from);
      const end = handPoint(to);
      context.beginPath();
      context.moveTo(start.x * canvas.width, start.y * canvas.height);
      context.lineTo(end.x * canvas.width, end.y * canvas.height);
      context.stroke();
    }
    for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
      const landmark = handPoint(index);
      context.beginPath();
      context.arc(
        landmark.x * canvas.width,
        landmark.y * canvas.height,
        Math.max(2, canvas.width / 230),
        0,
        Math.PI * 2,
      );
      context.fill();
    }
  };

  drawHand(leftHand, "#74d9ff");
  drawHand(rightHand, "#d9a0ff");
}
