/**
 * Runs the tracking evaluation and writes a per-model JSON result plus a
 * human-readable Markdown report.
 *
 *   npx tsx scripts/run-tracking-eval.ts --model mediapipe-heavy
 *
 * Extra scenarios are picked up automatically from `data/eval-scenarios/*.json`
 * (public datasets and developer pre-validation clips exported to the
 * `EvalScenario` shape). The process exits non-zero when any pre-registered
 * acceptance criterion fails, so a regression cannot pass CI quietly.
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { builtInScenarios } from "../app/tracking-eval-scenarios.ts";
import {
  checkAcceptance,
  evaluateScenario,
  renderMarkdownReport,
  type EvalScenario,
} from "../app/tracking-eval.ts";

const args = process.argv.slice(2);
const modelLabel = args[args.indexOf("--model") + 1] ?? "mediapipe-heavy";
const outputDir = join(process.cwd(), "docs", "tracking-eval");
const scenarioDir = join(process.cwd(), "data", "eval-scenarios");

function loadExternalScenarios(): EvalScenario[] {
  let entries: string[];
  try {
    entries = readdirSync(scenarioDir).filter((name) => name.endsWith(".json"));
  } catch {
    return [];
  }
  return entries.map((name) => {
    const scenario = JSON.parse(readFileSync(join(scenarioDir, name), "utf8")) as EvalScenario;
    if (!scenario.source) {
      throw new Error(`${name} has no "source" field. Every scenario must declare its provenance.`);
    }
    return scenario;
  });
}

const scenarios = [...builtInScenarios(), ...loadExternalScenarios()];
const results = scenarios.map((scenario) => {
  const result = evaluateScenario(scenario);
  return { result, acceptance: checkAcceptance(result) };
});

mkdirSync(outputDir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const jsonPath = join(outputDir, `${modelLabel}.json`);
const markdownPath = join(outputDir, `${modelLabel}.md`);

writeFileSync(
  jsonPath,
  `${JSON.stringify({ model: modelLabel, generatedAt: stamp, results }, null, 2)}\n`,
);
writeFileSync(markdownPath, renderMarkdownReport(modelLabel, results));

const failures = results.flatMap(({ result, acceptance }) =>
  acceptance.filter((criterion) => !criterion.passed).map((criterion) => `${result.scenario}: ${criterion.label} = ${criterion.actual} (target ${criterion.target})`),
);

console.log(`model:     ${modelLabel}`);
console.log(`scenarios: ${scenarios.length}`);
console.log(`json:      ${jsonPath}`);
console.log(`report:    ${markdownPath}`);
if (failures.length) {
  console.error(`\n${failures.length} acceptance failure(s):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exitCode = 1;
} else {
  console.log("\nAll pre-registered acceptance criteria passed.");
}
