# guided-visual-check

Reference-guided visual inspection with a vision model, arranged so that **the
model reports evidence and the code decides what happens as a result**.

You give it a reference image that defines what correct looks like, a photo
taken from the same vantage point, and a list of checks. With a reference from
the same place, the model is asked what differs from a known-good picture
instead of whether a photo looks right. The result has one finding per check
(what the model saw, what the rule asked for, how confident it is, where to
look), and a policy in code turns each finding into an action: report it, or
send it to a human.

```
reference image ──┐
subject image ────┼──► model ──► findings (evidence only)
measured inputs ──┘                  │
       ▲                             ▼
       │                   Evaluator._resolve()    ← rules in code
 computed outside                    │
 the model                           ▼
                             Policy.resolve()      ← the decision, in code
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
              report to subject            human review queue
```

---

## Why it is built this way

This is extracted from an inspection system running against a real site
network. The four design decisions below are the reason the repository exists.

### 1. The model observes, the code decides

The model reports what it sees and how confident it is. What happens as a
result is decided in code, for three reasons. A threshold in code runs the same
way on every call, while a prompt instruction is a request the model may not
follow. Tuning a number in a dataclass is a reviewable diff, while rewording a
prompt can change everything else that prompt does. And when a finding is
disputed, there is a rule in code to point at.

So `Finding` holds evidence only: status, observation, expected state,
confidence and locator. `Policy.resolve()` in [`gvc/policy.py`](gvc/policy.py)
maps each finding to an action.

Before the policy runs, `Evaluator._resolve()` in
[`gvc/evaluator.py`](gvc/evaluator.py) applies the rules that must hold
whatever the model returned:

- **One finding per check.** The result has exactly one finding per check in
  the checkpoint, in checkpoint order. A finding for an id that is not in the
  checkpoint is dropped. A check with no finding, or with more than one,
  becomes `not_assessable` and goes to human review; for a repeated check,
  every reported status and observation is kept for the reviewer.
- **Missing measurements.** A check whose measured input is missing comes back
  `not_assessable` (decision 3).
- **Unusable images.** If the model calls the image unusable, the result has no
  findings (decision 4).

### 2. A false positive costs more than a false negative

This is a domain assumption rather than a general rule. In inspection, a wrong
accusation costs more trust than a missed failure, because the person who
receives it stops believing the tool.

So a `fail` below the confidence threshold **never reaches the subject of the
inspection**: it goes to human review, and so does a pass below the threshold.
[`test_policy.py`](tests/test_policy.py) covers both. Domains where a miss is
the more expensive error need the opposite asymmetry, and `policy.py` is the
place to encode it.

### 3. Measured quantities are told to the model, never asked of it

Object orientation is a weak spot for vision-language models. On the DORI
benchmark, built to test orientation on its own, the best evaluated model
reaches **64.2% on coarse orientation judgments and 42.9% on granular ones**
([arXiv:2505.21649](https://arxiv.org/abs/2505.21649)). The paper measures
orientation; applying the same rule to angles, counts and distances is a
design choice of this repository.

So a quantity with a closed form comes from a measurer outside the model, such
as a pose estimator, an inclinometer, an EXIF field or a database lookup, and
reaches the model as a fact, with an instruction not to re-derive it from the
image.

When a measurement is missing, the check that depends on it comes back
`not_assessable`. The prompt asks the model for that, and `_enforce_measured`
in [`gvc/evaluator.py`](gvc/evaluator.py) applies it to the result whatever the
model answered, so the outcome does not depend on the model declining to guess.

### 4. Judge the image before judging the subject

If the model reports the image as unusable (blurred, badly framed, too dark,
obstructed), the result contains no findings, even when the model returned
some. The CLI prints the reason and asks for a new photo, so a bad photo is
handled as a capture problem instead of a failure of the thing inspected.

---

## Try it

Requires Python 3.11+. The dry run needs no API key:

```bash
pip install -e ".[dev]"
python -m gvc.cli dry-run \
  --checkpoint examples/solar-array-row-3.yaml \
  --image examples/images/subject.jpg
```

```
{
  "checkpoint": "solar-array-row-3",
  "model": "claude-sonnet-5",
  "checks": 4,
  "vision_tokens": 1610,
  "estimated_input_tokens": 2651,
  "estimated_output_tokens": 480,
  "estimated_cost_usd": 0.010102,
  "measured_inputs": {},
  "missing_measurements": [
    "tilt_angle_deg"
  ]
}

Note: no measurer registered for tilt_angle_deg.
Checks depending on it will be reported not_assessable.
```

`dry-run` does everything except the API call: it prepares both images,
collects the measured inputs and estimates the cost, so a malformed checkpoint
fails before anything is spent. Vision tokens use the image formula below. The
text (system prompt, output schema and message) is estimated at four
characters per token and the output at 120 tokens per check, so the figures
are approximate; `evaluate` reports the real usage. The output estimate leaves
out thinking tokens, which are billed as output on models that think by
default.

The two sample images in `examples/images` are synthetic;
`python examples/make_sample_images.py` regenerates them.

To evaluate for real:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
python -m gvc.cli evaluate \
  --checkpoint examples/solar-array-row-3.yaml \
  --image examples/images/subject.jpg
```

If the model declines the request, or the response reaches the output limit
before the JSON is complete, `evaluate` raises `EvaluationError` instead of
parsing partial output.

The sample checkpoint describes a row of solar modules. Its `module-tilt` check
depends on the measured input `tilt_angle_deg`, so with no measurer registered
it comes back `not_assessable` and the other three checks are evaluated
normally. A measurer receives the image path and returns a value, or `None`
when it has no reading; the value reaches the model through `str()`:

```python
from gvc import Checkpoint, Evaluator, MeasuredInputs

measured = MeasuredInputs()
measured.register("tilt_angle_deg", read_inclinometer)  # your function

checkpoint = Checkpoint.load("examples/solar-array-row-3.yaml")
result = Evaluator(measured_inputs=measured).evaluate(
    checkpoint, "examples/images/subject.jpg"
)
```

---

## Cost, and the block order that controls it

Images are billed at `ceil(w/28) * ceil(h/28)` vision tokens. The long edge is
downscaled to 1500 px by default (`long_edge` on `Evaluator` changes it), so a
4:3 photo comes to 2,214 vision tokens at 1500x1125. The 960x640 sample images
come to 805 each.

The request puts what repeats across evaluations of a checkpoint first: the
reference image, then the checkpoint's rules. The measured inputs and the
subject image come last. The reference image and the rules each carry a cache
breakpoint, so a later evaluation of the same checkpoint can read that prefix
from cache. `evaluate` makes one call per image, and for that case the
breakpoint after the rules would be enough. The one on the reference image is
for a caller that splits the checks for one image across several calls: the
rules text then differs from call to call, and the prefix that ends at the
reference image is the part those calls share.

`evaluate` computes the cost of each call from the `usage` the API returns,
with cache writes at 1.25x the input rate and cache reads at 0.10x. Pricing
cached tokens at the full input rate would overstate what cached runs cost.

---

## What this does not do

- **It does not measure its own accuracy.** Accuracy depends on the domain, the
  reference images and the checks. Measure agreement with a human inspector on
  your own photos before relying on it.
- **It does not choose the model for you.** `GVC_MODEL` sets it (default
  `claude-sonnet-5`). The model needs a row in `PRICES` in
  [`gvc/evaluator.py`](gvc/evaluator.py) so that costs use its own prices; any
  other id is rejected before a call is made.
- **It does not ship a measurer.** Measurement is domain-specific;
  `MeasuredInputs` is where one plugs in.
- **It has no persistence, queue or UI.** One call evaluates one image against
  one checkpoint and returns one result.

---

## Tests

```bash
python -m pytest -q
```

CI runs the same command on Python 3.11
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). The tests do not
call the API: `evaluate` runs against a stub client. Good
places to start are [`test_policy.py`](tests/test_policy.py), for what happens
to a low-confidence failure, [`test_measured.py`](tests/test_measured.py), for
a confident answer about a measurement the model never received, and
[`test_evaluator.py`](tests/test_evaluator.py), for invented, missing and
repeated findings, unusable images, refusals and truncated responses.

## License

MIT. See [LICENSE](LICENSE).
