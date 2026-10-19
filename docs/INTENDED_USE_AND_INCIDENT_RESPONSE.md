# Intended Use

AsterWhite is narrative/creative-writing generation infrastructure:
long-form fiction, character simulation, and interactive story tools.

**Not intended for**: factual question-answering, code generation,
medical/legal/financial advice, or any safety-critical decision. Nothing
in this repository trains or validates for those uses, and
`asterwhite.safety.PolicyFilter` is a basic heuristic filter, not a
red-teamed safety classifier (see its module docstring) -- don't rely on
it as the only safeguard in a deployment that faces real users.

Every deployment should fill in `docs/MODEL_CARD_TEMPLATE.md` for the
specific checkpoint being served, before real users reach it. The bundled
`pretrained/asterwhite-small` already has its own card
(`pretrained/asterwhite-small/MODEL_CARD.md`) describing exactly what it
was trained on and what it can't do -- it is a small demonstration model
trained on public-domain fiction, with no safety tuning, so it is not
suitable as-is for a public-facing product.

# Incident Response (skeleton -- fill in per deployment)

1. **Detect**: what alerting actually watches `/metrics` and the
   structured logs (`asterwhite.observability`) in this deployment?
   `<Prometheus alert rules, log-based alerts, ...>`
2. **Contain**: fastest safe mitigation.
   - Kill switch: set `ASTERWHITE_SAFE_MODE=1` in the serving environment
     and restart (`asterwhite.safety.safe_mode`) to block tool execution
     and external writes without a full redeploy.
   - Rollback: `registry.manifest.ModelRelease.rollback_target` should
     name a known-good prior release; redeploy pointing
     `ASTERWHITE_SERVE_CHECKPOINT_DIR` at it.
3. **Assess**: who is notified, what counts as "resolved," what evidence
   gets preserved (relevant `run_record.json` / `ModelRelease` files,
   scrubbed logs).
4. **Post-incident**: write an ADR (`docs/ADR_TEMPLATE.md`) for any
   architecture/process change that comes out of this.

This skeleton is a starting structure, not a substitute for your actual
on-call process, legal/compliance sign-off, or communication plan.
