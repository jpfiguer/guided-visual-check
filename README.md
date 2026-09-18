# guided-visual-check

Reference-guided visual inspection with a vision model, arranged so that **the
model reports evidence and the code decides what happens as a result**.

Give it a reference image that defines what correct looks like, a photo taken
from the same vantage point, and a list of checks. It returns one finding per
check — what it saw, what the rule asked for, how confident it is, where to look
— and then a policy in readable code turns that into an action: report it, or
send it to a human.

```
reference image ─┐
                 ├─► model ─► findings (status, confidence, locator)
subject image  ──┤                     │
measured inputs ─┘                     ▼
   ▲                            Policy.resolve()   ← the decision, in code
   │                                   │
deterministic,          ┌──────────────┴──────────────┐
not the model           ▼                             ▼
                 report to subject              human review queue
```

---

## Why it is built this way

This is extracted from an inspection system running against a real site network.
The four decisions below are the ones that survived contact with it, and they
are the reason the repository exists — the API calls are the easy part.

### 1. The model observes, the code decides

A vision model is good at saying what it sees and roughly how sure it is. It is
the wrong place to decide the consequence.

A prompt instruction is a request; a threshold in code runs every time,
identically. Thresholds also get tuned, and tuning a number in a dataclass is a
reviewable diff, while tuning a sentence inside a prompt silently changes
everything else that prompt does. And when a finding is disputed — it will be —
you need to point at the rule that produced it.

So `Finding` carries status and confidence and nothing else. `Policy.resolve()`
in [`gvc/policy.py`](gvc/policy.py) is the entire decision layer, and it is
twenty lines you can read in one sitting.

### 2. A false positive costs more than a false negative

Not a universal truth — a domain assumption, written down where it can be
argued with. In inspection, one wrong accusation destroys trust faster than ten
quiet catches build it, because the person receiving it stops believing the
tool.

So a `fail` the model is not confident about **never reaches the subject of the
inspection**. It goes to a human first. That single behaviour has its own test,
and if it ever flips the system has started accusing people on a hunch.

Domains where a miss is the expensive error want the opposite asymmetry, and
should say so in `policy.py` rather than anywhere else.

### 3. Measured quantities are told to the model, never asked of it

This is the one that is easiest to skip and most expensive to get wrong.

Vision-language models are weak at metric questions — orientation, angles, exact
counts, distances. Published evaluations put general VLMs near the floor on
"which way is this object facing" while humans sit near the ceiling. The failure
is quiet: the model does not refuse, it returns a confident number.

So anything with a closed form is computed by something that computes — a pose
estimator, an inclinometer, an EXIF field, a database lookup — and handed over
as a fact, with an explicit instruction not to re-derive it from the image.

And when the measurement does not arrive, the check that depends on it comes
back `not_assessable`. That is enforced twice on purpose: the prompt asks for
it, and [`_enforce_measured`](gvc/evaluator.py) makes it true whether or not the
model complied. A gap in the data has to look like a gap.

### 4. Judge the image before judging the subject

If the model says the image is unusable, **no findings are emitted at all** —
enforced in code, not left to the prompt. A blurred photo is a capture problem.
Systems that skip this start blaming people for bad lighting, and the people
being blamed stop taking photos.

---

## Try it

Runs with no API key and no assets:

```bash
pip install -e ".[dev]"
python examples/make_sample_images.py
python -m gvc.cli dry-run \
  --checkpoint examples/solar-array-row-3.yaml \
  --image examples/images/subject.jpg
```

```
Note: no measurer registered for tilt_angle_deg.
Checks depending on it will be reported not_assessable.
{
  "checkpoint": "solar-array-row-3",
  "checks": 4,
  "vision_tokens": 1610,
  "estimated_cost_usd": 0.008666,
  "missing_measurements": ["tilt_angle_deg"]
}
```

`dry-run` exists for its own sake. Being able to answer *what will this cost*
before spending anything changes how willing people are to try things, and it
catches a malformed checkpoint without burning a call.

To evaluate for real:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
python -m gvc.cli evaluate \
  --checkpoint examples/solar-array-row-3.yaml \
  --image examples/images/subject.jpg
```

The sample domain is a solar array, chosen because it is nothing like the system
this came from. `module-tilt` depends on `tilt_angle_deg` — a number nobody
should ask a vision model for — so out of the box that check returns
`not_assessable` and the other three are evaluated normally. Register a measurer
and it starts working:

```python
from gvc import Evaluator, MeasuredInputs

measured = MeasuredInputs()
measured.register("tilt_angle_deg", lambda image: read_inclinometer(image))

Evaluator(measured_inputs=measured).evaluate(checkpoint, image)
```

---

## Cost, and the block order that controls it

Images are billed at `ceil(w/28) * ceil(h/28)` vision tokens, so the long edge is
downscaled to 1500 px by default: around 2,200 tokens per image, keeping the
detail an inspection needs.

The request is ordered stable-first — reference image, then rules, then the
subject image last — with cache breakpoints at the end of the stable part. The
breakpoint sits **behind the reference image**, not only after the rules, and
that detail only pays off at scale: once one image is evaluated across several
calls (one per group of checks, to keep each call small), the rules differ
between calls. A cache breaking only after them would never hit, and the
reference image — the expensive block — would be paid for in full every time.

`gvc/evaluator.py` reports real cost per call from `usage`, including the two
cache rates (writing ≈1.25x input, reading ≈0.10x). Ignoring that split makes
cached runs look more expensive than they are and hides the thing you are
optimising.

---

## What this does not do

- **It does not tell you it is accurate.** Accuracy is a property of your
  domain, your reference images and your checks, and the only number worth
  quoting is agreement with a competent human on your own photos. Synthetic
  scenes are a floor, not a ceiling. Measure it before you trust it.
- **It does not pick a model for you.** `GVC_MODEL` is an environment variable
  because the choice should be revisited as labelled data arrives, and a cheaper
  model that ties on your data is the right model.
- **It does not ship a measurer.** Deliberately: the measurement is the
  domain-specific part, and pretending otherwise would encourage exactly the
  guessing this design exists to prevent.
- **It has no persistence, queue or UI.** One image, one checkpoint, one result.
  The production system it came from splits capture from evaluation into
  separate processes for good reasons, but that is an architecture, not a
  pattern, and it does not belong in a reference implementation.

---

## Tests

```bash
python -m pytest -q
```

The tests are the specification of the guarantees, not coverage decoration. The
ones worth reading first are
[`test_policy.py`](tests/test_policy.py) — what happens to an unconfident
failure — and [`test_measured.py`](tests/test_measured.py) — what happens when a
model answers confidently about a measurement it was never given.

## Licence

MIT — see [LICENSE](LICENSE).
