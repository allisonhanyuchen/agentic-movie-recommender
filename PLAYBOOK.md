# Agent Building Playbook

A strategy-to-implementation reference for building LLM agents. Written as a PM–engineer handoff map: what each phase should produce, what questions to ask, and what good looks like.

---

## Orientation

### What this playbook is for

A reusable framework for any project where the core deliverable is an LLM-powered agent — something that takes input, makes decisions with a model in the loop, and returns a structured or semi-structured output. The phases below apply whether the agent is a recommendation system, a research assistant, a document processor, a customer-support bot, or an autonomous task-runner. The details change; the build order doesn't.

### Three common agent shapes

Agent projects usually fall into one of three buckets, and the bucket shapes everything downstream.

**Task-oriented (narrow contract).** One function, a defined input and output schema, evaluated by a judge (automated, human, or LLM). Examples: classification, recommendation, structured extraction. Highest leverage from deterministic scaffolding; LLM used sparingly and for clearly-bounded jobs.

```text
project/
├── agent.py           # public API: get_X(input) -> structured output
├── retrieval.py       # candidate generation, ranking
├── prompts.py         # templates + few-shot exemplars
├── validators.py      # schema + constraint checks
├── fallback.py        # deterministic fallback paths
├── baseline.py        # simple comparator for A/B
├── evaluate.py        # three-layer evaluator
├── cases/eval_cases.json
└── artifacts/         # timestamped eval reports
```

Logic flow: `input → filter → retrieve → rank → route (skip LLM?) → [LLM arbitrate | template] → validate → return (fallback on failure)`

**Conversational.** Multi-turn, stateful, often open-ended. Examples: support agents, research assistants, tutors. Contract is fuzzier; evaluation leans on rubric-based quality scoring over dialogue traces rather than pass/fail checks.

```text
project/
├── agent.py           # main loop: handle_turn(conversation, new_msg) -> response
├── state/
│   ├── session.py     # per-session memory, user profile
│   └── context.py     # context window management, truncation
├── tools/             # tool registry (search, calendar, etc.)
├── prompts.py         # system prompt, turn prompt, tool-use prompt
├── retrieval.py       # RAG over knowledge base if used
├── safety.py          # content filters, refusal logic
└── evaluate/
    ├── dialogue_cases.json     # multi-turn scenarios
    └── rubric_judge.py         # conversation-level scoring
```

Logic flow: `new_turn → load session → intent detection → [tool call | retrieval | direct] → compose response → safety filter → update state → return`

**Autonomous / long-running.** Plans, executes, uses tools, may run for minutes or hours. Examples: code agents, scheduling agents, multi-step research. Evaluation focuses on task completion rate, cost per task, safety of actions taken. Requires strong observability and rollback discipline.

```text
project/
├── agent.py           # orchestrator: run_task(task) -> result
├── planner.py         # task decomposition, step planning, replanning
├── executor.py        # step execution, tool invocation, retries
├── memory/
│   ├── working.py     # current-task scratchpad
│   └── episodic.py    # cross-task history, learned patterns
├── tools/             # action catalog with schemas
├── guardrails/        # approval gates, cost caps, dangerous-action blocks
├── observability/     # per-step traces, metrics, structured logs
└── evaluate/
    ├── task_suite.json         # end-to-end task benchmarks
    └── trace_judge.py          # process + outcome scoring
```

Logic flow: `task → plan → (loop: select step → invoke tool → observe → update memory → replan?) → finalize → report + trace`

This playbook is framed for task-oriented agents because the discipline it teaches is the foundation for the others. Every conversational and autonomous agent has task-oriented sub-components inside it.

### The core mental model

Treat the LLM as an expensive, capable, somewhat-unreliable generalist. Its value is concentrated in a narrow set of tasks: synthesis, judgment, natural-language generation, pattern matching on fuzzy inputs. Everything else — structure, validation, filtering, routing, state management — belongs in deterministic code.

The scaffolding around the LLM is what makes the system good. The LLM is a replaceable component within it.

---

## Phase 0 — Strategy

**Goal of this phase:** decide what you're building, why, and what "done" means. No code yet.

### What to produce

- A one-sentence statement of what the agent does and for whom.
- Success criteria: the specific metric(s) that tell you it's working, and the thresholds that define "good enough to ship."
- The non-negotiables: rules that must hold no matter what (latency ceiling, cost budget, safety boundaries, output format).
- The competitive baseline: what you're trying to beat. Could be a human, a simpler system, a prior version, or a rule-based heuristic.
- An honest statement of what this agent *won't* do, to prevent scope creep mid-build.

### Strategy-level decisions that compound downstream

**What's the judge?** Who or what decides whether an output is good? Automated checks are cheapest but narrow. Human raters are accurate but slow and expensive. LLM-as-judge is a middle ground but needs careful design to avoid bias. The judge choice determines how fast you can iterate.

**What's the failure mode you care most about?** An agent can fail in many ways — wrong answer, offensive output, slow response, hallucinated facts, ignored constraints. Rank them. The top item shapes the guardrail design.

**How often does the agent run?** Once per user query (cost sensitivity matters), in batch overnight (latency is forgiving), continuously (observability matters). This decides where you can afford to spend tokens.

### PM questions to ask at Phase 0

- What's the simplest version of this that still delivers value?
- What's the worst acceptable outcome, and what mechanism prevents it?
- If we can only optimize one metric, which one?
- Who is the judge of quality, and is that judge scalable?
- What's the budget per request — tokens, latency, dollars?

---

## Phase 1 — Planning

**Goal of this phase:** translate strategy into a concrete technical spec. Still no agent code, but the evaluator and the case set start taking shape.

### What to produce

**The contract.** The agent's public interface written on one line: input types, output schema, guarantees. For a task-oriented agent this is a function signature; for a conversational agent it's a message schema; for an autonomous agent it's a task description format and a result format. Everything built later is scaffolding around this contract.

**The evaluation plan.** Define how you will know the agent is working before you build it. This includes:

- Hard-fail checks that are binary (valid format, within latency budget, within cost budget, no prohibited content).
- Quality rubric (the axes you'll score, e.g. relevance, accuracy, coherence, style — 3 to 5 axes is typical, more becomes noisy).
- Comparison design (vs. baseline, vs. prior version, vs. human).

**The case set.** A representative collection of inputs that covers:

- Easy cases (sanity check — the agent shouldn't fail these).
- Common cases (reflects the real distribution of user requests).
- Hard cases (known failure modes; edge cases; adversarial inputs).
- Degenerate cases (empty input, malformed input, boundary conditions).

Aim for a case set large enough to give stable metrics (typically 20–50 for iteration, larger for final reporting) but small enough to run cheaply.

**Budget allocation.** Given the latency ceiling, how much goes to retrieval, to LLM calls, to validation, to fallback? Write it down. This constrains architecture in useful ways.

### PM questions to ask at Phase 1

- What's the contract? Write it on one line.
- Who or what is the judge, and what are its biases?
- What does the case set not cover? What's the blind spot?
- What's the per-request budget — and which component gets the largest share?
- What does "below target on metric X" mean we do?

---

## Phase 2 — Architecture & Design

**Goal of this phase:** a design that a competent engineer could implement without needing to re-invent decisions. The design answers how the agent routes work between deterministic code and LLM calls.

### Pipeline shape

Most retrieval-augmented or decision-making agents share this structure:

```text
input
  → deterministic filters (exclude/require, apply hard constraints)
  → retrieval or candidate generation (local search, vector store, API)
  → scoring / content filter (rank candidates by fit + quality signals)
  → [optional] external enrichment, intersected back to the safe set
  → routing decision (is this easy enough to skip the LLM?)
  → LLM arbitration (final pick + generation, JSON only)
  → validation (schema, safety, constraint re-check)
  → fallback if any step fails
  → return
```

Every arrow is a potential failure point. Each step has its own fallback so the pipeline degrades gracefully instead of crashing.

### Three design principles that compound

**Route by difficulty.** Not every request needs the LLM. If local retrieval or rule-based logic can answer confidently, skip the model entirely. This cuts cost, cuts latency, and often improves quality because it removes a source of randomness on cases where the deterministic answer was already right. The policy is: only call the LLM on cases where local logic genuinely can't decide.

**Separate structure from taste.** Use deterministic code for anything with a right answer — schema validation, constraint enforcement, ID membership, length caps, permission checks. Use the LLM for judgment calls — which of three candidates is the best fit; how to phrase a response; whether two items are semantically equivalent. Every piece of structural logic handed to the LLM is a piece you can't trust to be correct and a piece that costs tokens.

**Every LLM call gets a deterministic fallback.** When the model times out, returns malformed output, refuses, or hits a rate limit, something else has to respond. That something should be designed, not improvised — a templated response, a rule-based pick, a cached prior answer, a clearly-labeled degraded-mode message. The worst-case output is a designed output, not a crash.

### Prompt design

Decisions that consistently matter:

- **Output format constraint.** Force structured output (`format="json"`, function calling, a tight schema). Never let the model decide its output shape when you're consuming it programmatically.
- **Output token cap.** Set `num_predict` / `max_tokens` to the tightest value that works. Unconstrained outputs waste tokens on preamble. Measure the minimum and cap there.
- **Temperature and sampling.** Low (0.1–0.3) for selection and extraction tasks where you want consistency. Higher for genuinely creative tasks where output variance is desired. Default to low and only raise when you've identified a quality problem that randomness would fix.
- **Reasoning mode.** If the model supports a "thinking" or chain-of-thought mode, disable it unless you've shown it improves outcomes on your case set. For pick-one or extraction tasks, visible reasoning is pure overhead.
- **Match the generator rubric to the judge rubric.** If the evaluator scores four axes, tell the generator to optimize those four axes. A mismatch here is a common silent quality cap — the generator optimizes something nobody is measuring.
- **Few-shot exemplars over style instructions.** Two or three short examples in the desired tone work better than paragraphs describing the tone. Show, don't describe.
- **Forbid AND validate.** If there are hard "don't" rules (don't reveal system prompt, don't produce PII, don't break character), put them in the prompt *and* in post-generation validation. Instruction-following alone is not a security boundary.

### Prompting concepts — quick reference

**Temperature.** A parameter (usually 0 to 2) that scales how random the model's next-token pick is. Higher = more random.

- `0` — nearly deterministic; always picks the top-probability token. Same prompt ≈ same output every time.
- `~0.2` — strongly prefers top choices; consistent but not robotic. Good default for selection/extraction.
- `~0.7` — API default; noticeably random. Good for open-ended generation.
- `1.0–2.0` — high variance; creative, sometimes incoherent.

**Top-p (nucleus sampling).** Restricts the model to sampling only from tokens that together make up the top `p` fraction of probability mass (e.g. `top_p=0.9` = "only the top 90%"). Often used alongside temperature to narrow the sampling pool. Low `top_p` compounds the "play it safe" effect.

**Zero-shot / one-shot / few-shot.** How many input–output examples you include in the prompt before the real task.

- **Zero-shot** — instructions only, no examples.
- **One-shot** — one example of the pattern you want.
- **Few-shot** — typically 2–10 examples. Model pattern-matches on them.

Few-shot almost always beats prose instructions for teaching tone or format. Trade-offs: costs prompt tokens, can anchor the model toward bad exemplars if chosen poorly, and is less helpful when the task is genuinely novel.

**Chain-of-thought (CoT).** Prompting the model to write out its reasoning step-by-step before answering, rather than jumping straight to the answer. Originally the trick "Let's think step by step"; modern frontier models expose it as a toggle (Ollama `think=True/False`, Anthropic extended thinking, OpenAI o-series). Helps on multi-step reasoning (math, logic, code debugging); adds cost and latency with no gain on simple selection/extraction. Rule of thumb: if a smart human could do this task without scratch paper, the model doesn't need CoT either.

### The safety / guardrail matrix

For every non-negotiable rule, define both a prompt-level instruction and a code-level enforcement:

| Layer | Role |
|---|---|
| Prompt | Tries to do the right thing |
| Code | Guarantees the wrong thing doesn't ship |

Instruction-following alone is not a security boundary. Every hard rule needs both layers.

### PM questions to ask at Phase 2

- What is the LLM actually doing in this design? Can any of its jobs be handled deterministically?
- What happens when the LLM times out, returns garbage, or hits a rate limit?
- Is the generator optimizing the same axes the judge is scoring?
- For each non-negotiable rule, where is it enforced — prompt, code, or both?
- Which step in the pipeline is load-bearing on a single external service?

---

## Phase 3 — Implementation

**Goal of this phase:** get to a working, measurable end-to-end system as fast as possible. Optimize later.

### The build order

1. **Build the dumbest version that satisfies the contract.** Skip retrieval, skip routing, skip optimization — just wire input to a single LLM call to output. This becomes your competitive baseline; you'll compare every fancier version against it. If your eventual system can't reliably beat the dumb version, something is wrong with your design, not your implementation.
2. **Build the evaluator before iterating on the agent.** Hard-fail checks, scoring rubric, case set, A/B harness. Every code change after this has a number attached to it. Without the evaluator, you're iterating on vibes.
3. **Instrument cost and latency from day one.** Tokens per call, calls per request, latency per stage. Log it all. Late-added instrumentation misses the early signals that suggest architectural changes.
4. **Build the real agent, one layer at a time.** Retrieval first. Measure. Filtering second. Measure. Routing third. Measure. Prompt tuning last. Each layer change either moves a metric or it doesn't — and if it doesn't, you know it's not the bottleneck.
5. **Design the fallback path with the same care as the happy path.** The fallback is what ships when the LLM fails. Make it coherent, correct, and clearly-labeled as a fallback so downstream consumers can tell.

### Change discipline

Keep a running log of changes and their metric impact. "Added year-range filter → novelty 4.5 → 4.87" is the kind of note that compounds across weeks. It tells you which levers actually move which metrics, and it prevents you from re-doing work you already tried.

When a change touches more than one layer, split it. A single commit that changes retrieval, filtering, *and* the prompt makes it impossible to attribute the resulting metric shift.

### PM questions to ask at Phase 3

- Does the dumb baseline satisfy the contract? If not, why not?
- Are we measuring tokens and latency per request yet?
- When was the last time a change moved a metric — what was it?
- What's the current bottleneck layer, and how do we know?

---

## Phase 4 — QA & Measurement

**Goal of this phase:** produce numbers you'd defend in front of a skeptical reviewer.

### The three-layer evaluator

1. **Hard disqualification checks.** Binary gates. Output schema valid, within latency budget, within cost budget, no prohibited content, all required fields present. An output failing any one doesn't count as a success, regardless of its quality on other dimensions.
2. **Quality scoring.** Rubric-based scores on 3–5 axes, typically 1–5 each. Use these to trend across iterations and identify weak spots. Noise is real — don't over-interpret 0.1 differences; rely on directional movement across multiple runs.
3. **Comparison.** A/B vs. baseline, prior version, or alternative design. Pairwise judge comparison is often more stable than absolute scoring because it cancels out some of the judge's scaling biases. Randomize position to prevent A/B bias; verify the randomization works before trusting results.

### Other evaluation frameworks worth knowing

The three-layer evaluator above is a strong default. In production work you'll encounter (and should know how to read) other approaches. Most real systems combine several.

- **Reference-based metrics.** Used when there's a ground-truth correct answer. Classification → accuracy, precision, recall, F1. Text generation with reference outputs → BLEU, ROUGE, exact-match. Cheap, fast, deterministic. Only useful when "correct" is well-defined.
- **Task completion / end-to-end success.** For agentic systems, the headline metric is often just: did the agent actually achieve the task? Named benchmarks in this family: SWE-bench (code fixes), WebArena (web navigation), ToolBench (tool use), GAIA (general assistant tasks). Binary success per task; success rate across a suite.
- **Trace-level evaluation.** For multi-step agents, scoring only the final outcome misses information. Trace-level eval scores the process: did it take unnecessary steps, retry appropriately, handle errors cleanly, stay within budget? Essential for autonomous agents where two runs can reach the same outcome with very different cost profiles.
- **Red-teaming / adversarial evaluation.** A curated set of adversarial inputs designed to probe specific failure modes — jailbreaks, prompt injection, hallucinations on out-of-distribution inputs, over-refusal of valid requests. Not about average quality; about worst-case safety and robustness. Usually maintained separately from the main case set and updated as new failure modes are found.
- **Human evaluation.** Still the gold standard for subjective quality. Expensive and slow, so usually reserved for: calibrating LLM-judge reliability, spot-checking top candidate systems before launch, resolving close A/B calls. Common patterns: pairwise preference with multiple raters, rubric scoring with inter-rater agreement, or satisfaction surveys on real user sessions.
- **Online / production evaluation.** Once the agent ships, real user signals — click-through, session length, thumbs up/down, retention, task completion reported by users — become available. These are the ground truth for any UX question. A/B tests with statistical significance thresholds replace offline judge comparisons. Offline eval predicts online results imperfectly; treat offline wins as hypotheses to confirm online.
- **RAG-specific frameworks.** If your agent uses retrieval, specialized metrics matter: faithfulness (is the answer grounded in the retrieved context?), answer relevance (does it address the question?), context precision/recall (did retrieval return the right documents?). RAGAS is a popular open-source toolkit for this.
- **LLM-judge harnesses and observability platforms.** Worth knowing by name when talking to engineers: G-Eval (LLM-as-judge with CoT reasoning templates), Promptfoo and DeepEval (open-source eval harnesses), LangSmith, Braintrust, and Langfuse (observability + eval platforms with tracing, dataset management, and judge pipelines). You don't have to pick one at Phase 1, but knowing the category saves you from reinventing a framework.
- **Public leaderboards.** MMLU, HELM, Chatbot Arena — give you broad-landscape context for model choice. Not a substitute for task-specific eval; useful when choosing which base model to build on.

A rough hierarchy for picking: start with hard-fail checks + a small rubric + an A/B vs. baseline. Add reference metrics if there's ground truth. Add trace-level scoring if the agent takes multiple steps. Add red-teaming before shipping. Add human eval for final calibration and online eval once live.

### Traps to watch for

- **Broken comparator.** If your baseline is losing (or winning) unusually often, read what it actually returned. A silently-failing baseline that emits error stubs will make your agent look like a genius. Spot-check per-case comparator output before trusting any A/B number.
- **Same-model judge bias.** Having a model judge outputs from the same model family introduces systematic preference. Use it as a fast sanity check during iteration; use a different model (or human rater) for numbers that ship.
- **Position bias.** Many judges prefer option A or option B systematically regardless of content. Seeded randomization plus a dry-run that scores identical-vs-identical pairs catches this.
- **Case set blind spots.** If your eval set doesn't include the hardest cases users will actually send, your numbers look falsely good. Periodically add real failure reports to the case set.
- **Run-to-run drift.** The same code with the same case set can produce measurably different scores across runs because of LLM sampling variance. Run multiple times. Report ranges or means, not single-run numbers for close calls.
- **Cherry-picked baselines.** The baseline you compare against defines what "winning" means. Winning against a weak baseline means little; winning against a strong one means a lot. Be explicit about which baseline you're using and why.

### Cost and latency measurement

Independently of quality, track:

- P50 and P95 latency per request (not just mean — tails matter for UX).
- Mean tokens per request (prompt + output).
- Cost per request in dollars.
- Fraction of requests that hit fallback paths.

A regression in any of these can be as serious as a quality regression.

**Percentile latency.** Sort all request timings from fastest to slowest. P50 (median) is the timing at the middle — what a "typical" user experiences. P95 is the timing 95% of requests are faster than — what 1 in 20 users waits. P99 is 1 in 100. Mean latency hides shape: a few slow outliers barely move the mean but create visible UX pain, so production SLOs are almost always written as P95 or P99, not mean.

### PM questions to ask at Phase 4

- What specific run produced these numbers? How many times was it run?
- Have you spot-checked the comparator's actual outputs on a handful of cases?
- Is the judge different from the generator model?
- Which metric is below target, and what's the plan to improve it?
- What's P95 latency — not mean?
- What fraction of requests hit a fallback path?

---

## Phase 5 — Reporting & Handoff

**Goal of this phase:** communicate the system's state honestly enough that a reviewer, a next-phase team, or a future version of you can make good decisions from your report.

### Results structure that works

- State the target thresholds *before* the measured numbers.
- State the measured numbers with the run date and the exact model versions used.
- Mark each row pass or below target honestly.
- Follow immediately with a "Methodological Caveats" section.

### The methodological caveats paragraph

The most important thing in the document. It's tempting to skip because it feels like self-sabotage. It's the opposite — it's what makes the numbers credible. A good caveats section calls out:

- What the comparison point can and can't tell you.
- Which metrics missed target and what the best theory for why is.
- Any prior-run bugs you caught and how.
- Known blind spots in the case set.
- Anything a skeptical reviewer would notice if you didn't mention it first.

A reviewer who finds a methodological problem you already named trusts the rest of your work more. A reviewer who finds one you hid trusts you less on everything.

### Handoff artifacts

For the next person (or next-you), document:

- The contract (function signature, schema, guarantees).
- The architecture diagram at the level of the pipeline-shape block above.
- The evaluation commands and where the artifacts land.
- The case set and its known blind spots.
- The running changelog of what moved which metric.
- The open problems and what the next most promising direction is.

### PM questions to ask at Phase 5

- What does the caveats paragraph actually say?
- What's the single thing a skeptical reviewer would notice first, and did we address it?
- If this project stopped today, what's the handoff document for the person who picks it up?
- What's the next most important improvement, and why that one?

---

## Transferable heuristics (the short list)

These apply to nearly every agent project, independent of domain.

- **Route by difficulty.** Cheap local path for easy requests; LLM for the judgment calls.
- **Separate structure from taste.** Deterministic code for rules; LLM for aesthetics.
- **Every LLM call gets a deterministic fallback.** Design it with the same care as the happy path.
- **Cap output tokens aggressively.** Use the tightest limit that works.
- **Disable chain-of-thought when it isn't improving outcomes.**
- **LLM as tiebreaker, not pipeline stage.** Retrieve first, arbitrate second.
- **Build the evaluator before the agent.** Every change should have a number attached.
- **Build the dumbest version first.** If fancy doesn't beat dumb, the design is wrong.
- **Prompt and code both enforce every non-negotiable rule.**
- **Spot-check your comparator.** 100% win rates usually mean the baseline is broken.
- **External judge for numbers that ship; same-model judge for iteration.**
- **Instrument cost and latency from day one.**
- **Change one layer at a time.** Attribution is impossible otherwise.
- **Report the below-target numbers too**, with a caveats paragraph.

---

## Mindset

The biggest mindset shift for a first-time agent builder is flipping the question from "how much LLM can I afford per request?" to "how little LLM can I get away with?"

Every token you don't spend is faster, cheaper, and often more accurate — deterministic code doesn't have a temperature parameter and doesn't hallucinate. The job of the system designer is not to wire the LLM into every decision; it's to identify the narrow set of decisions where the LLM is genuinely the best tool and hand it only those decisions.

The best agents don't treat the LLM as a magic answer box. They treat it as an expensive, capable, somewhat-unreliable generalist whose value is concentrated in synthesis, judgment, and natural-language generation. The scaffolding around it — the retrieval, the filters, the validation, the fallbacks, the tests — is what makes the system good. When it works well, a user never realizes the LLM was involved at all; they just see a system that feels reliably smart.

---

## Appendix: common implementation pitfalls

These aren't strategy-level; they're the small surprises that waste hours when you hit them. Worth scanning once so you recognize them faster.

### Environment and runtime

- Language/runtime version mismatches on the deployment machine (e.g. a default interpreter older than the one you developed on). State the required version explicitly.
- Working directory vs. script location mismatch breaks local imports. Scripts that work from the package directory fail when run from `/tmp` or a parent directory.
- Dotfiles hidden by default in some OS file browsers. Easy to think a config file is missing when it isn't.

### Testing and artifacts

- Stale output files looking identical to fresh ones. Check file mtimes after re-runs.
- Zips silently excluding files that were present in the source directory. Round-trip by unzipping and testing in a fresh environment before shipping.
- AST-based import scanners false-positive on local modules (they can't tell a sibling `.py` from a missing PyPI package).

### Configuration

- Template placeholder values pasted literally (e.g. `<your_key>` ending up in a config file because someone copied it verbatim). Write placeholders that obviously break if pasted, or provide automated substitution.
- Credentials accidentally committed. Always gitignore credential files and use an `.example` variant for the schema.

### Measurement

- Eval runs that cache results silently — a "new run" that's actually reading a cached response. Verify cache invalidation.
- Metric improvements from changes that also happened to change the random seed. Separate seed effects from logic effects.
- Single-run metrics reported as if they were stable. Run multiple times for close calls.
