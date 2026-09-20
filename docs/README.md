# TinyInfer documentation

TinyInfer separates documentation by purpose so that descriptions of the
current implementation do not get mixed with learning material or historical
design rationale.

## Documentation map

| Area | Purpose | Start here |
| --- | --- | --- |
| [Architecture](architecture/README.md) | Describes how the current system is structured and behaves. | [System architecture](architecture/architecture.md) |
| [Tutorials](tutorials/README.md) | Builds runtime concepts step by step with code, tests, and experiments. | [Tutorial plan](tutorials/README.md) |
| [Design decisions](decisions/README.md) | Records why important architectural choices were made and their tradeoffs. | [Decision log](decisions/README.md) |

The phased development status remains in the
[roadmap](architecture/roadmap.md). It lives with the architecture documents
for now because it describes the implemented boundaries of each subsystem as
well as future milestones.

## Writing rule

Each document should answer one primary question:

- Architecture: **How does TinyInfer work now?**
- Tutorial: **How can a reader build and understand this mechanism?**
- Design decision: **Why did TinyInfer choose this design over alternatives?**

When behavior changes, update the architecture document. When a decision is
reconsidered, preserve the original decision record and mark its status rather
than rewriting its history.
