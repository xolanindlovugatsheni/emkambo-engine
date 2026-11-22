# Working together

## Ground rules

1. **No real data in the repository.** ZMX trades, bank loan files and anything a partner gave us stay outside git. The `.gitignore` blocks `data/`, `runs/` and `config.yaml`, but check before every commit.
2. **Every published number traces to a run.** A figure in a note, deck or email must come from a `run.json` with its seed and input hashes. Keep that file with the document.
3. **Tests pass before merging.** GitHub runs the full suite on every push to `main`. A red result blocks the change.
4. **Reasoning on paper.** Every methodology choice gets a short justification and the alternative that was rejected, in the pull request or the methodology note.
5. **Partner data stays within Emkambo.** It is never shared and never used for coursework without written consent.

## How a change gets in

1. Create a branch from `main`: `git checkout -b short-description`.
2. Make the change, with a test that would have failed before it.
3. Run `pytest -q` locally.
4. Push the branch and open a pull request describing what changed and why.
5. The other partner reviews. Methodology changes need the financial engineering lead's sign-off; engine changes need the founder's.

## Weekly rhythm

One-hour review each week: what was delivered, what is blocked, what decisions are needed. Decisions are written down the same day.

## Open questions we are working on

- What liquidation lag do lenders actually experience?
- How should ZiG-settled trades be converted: official or market rate?
- Do lenders treat delivery to the GMB as their exit for maize, and on what terms?
- Does a two-regime model absorb policy shocks, or is a jump term needed?

← [Home](Home.md)
