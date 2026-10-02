# Kea × SwanLake distributed-integration discovery TODO

## Gate: complete discovery before implementation

**Do not begin distributed-integration implementation until the discovery and
reuse sweep below is complete.** The first deliverable is a written mapping of
which SwanLake components Kea can reuse directly, adapt behind an existing
Kea interface, use only as a behavioral reference, or deliberately exclude.
That decision should drive the integration design rather than copying code
preemptively.

## Discovery and comparison

- [ ] Map both repositories' training pipelines end-to-end: sampling,
  labeling, training-data construction/propagation, model training,
  recursive rounds, and inference.
- [ ] Perform a broad reuse sweep of SwanLake's distributed components:
  RPC/protocol definitions and serialization; coordinator/worker lifecycle;
  shard discovery/configuration; request dispatch; result collection, retry,
  and error handling; model broadcast; federated weight aggregation;
  distributed query hooks; and timing/metrics instrumentation.
- [ ] Classify every SwanLake component as **directly reusable**,
  **adaptable**, **reference-only**, or **not needed** by Kea, and record the
  reason, dependencies, and owning interface in Kea.
- [ ] Inventory Kea's existing abstraction boundaries:
  `ITrainingExecutionBackend`, `IShardWorker`, sampler, labeler,
  `ITrainingDataBuilder`, `RunConfig`, protocol request/response types, and
  the training runner.
- [ ] Identify which Kea abstractions are already transport-neutral and which
  embed single-machine assumptions that must be refactored before remote
  execution.
- [ ] Compare data ownership in both systems: text, IDs, embeddings, cluster
  assignments, direct labels, pseudo-labels, training examples, and models.
  In particular, determine what must remain worker-local and what must cross
  the network.
- [ ] Verify that Kea's borrowed DuckDB embedding views never cross a process
  boundary; RPC payloads must own or serialize any data they transfer.

## Scope and design decisions

- [ ] Define acceptance criteria for the required modes: LC1, LC2, LC3, LC4,
  and their recursive variants. Decide explicitly whether GC1/GC2 are in
  scope.
- [ ] Trace SwanLake propagation in both one-shot and recursive modes,
  including majority voting, tie behavior, uncovered clusters, and the direct
  label history sent to stateless workers.
- [ ] Map propagation onto Kea's `ITrainingDataBuilder`: pseudo-labels expand
  training data but are not written into DuckDB and do not count against the
  ground-truth labeling budget.
- [ ] Design a remote backend behind Kea's existing interfaces for:
  initial-label acquisition, recursive uncertain-label acquisition, optional
  local model training, model broadcast, and per-round timing/error reporting.
- [ ] Define coordinator and worker lifecycle, shard configuration, run IDs,
  cancellation/error semantics, and serialization formats before selecting or
  adapting SwanLake transport code.
- [ ] List required Kea refactors: removal of coordinator-local assumptions,
  explicit shard cache/partition lifetime, stable direct-label history,
  central aggregation, and worker-local federated training.

## Incremental implementation, after the discovery gate

- [ ] Build a minimal vertical slice with two real Kea worker processes:
  random clean-label sampling, central training, model broadcast, and output
  equivalence with Kea's in-process multi-shard backend.
- [ ] Implement and validate one-shot cluster modes in order: LC1, LC2, LC3,
  then LC4.
- [ ] Add recursive rounds only after the corresponding one-shot mode works.
  Confirm that workers reconstruct the intended propagated training set and
  exclude only directly ground-truth-labeled IDs from future selection.
- [ ] Adapt distributed observability to report worker sampling, fetching,
  clustering, labeling, training, serialization/transfer, coordinator
  aggregation, and full round timing.

## Validation

- [ ] Add a single-shard equivalence test between the remote backend and the
  current Kea execution path.
- [ ] Test exact global-budget splitting, label counts, propagation coverage,
  majority/tie behavior, and per-mode data movement.
- [ ] Test that central and federated modes transmit the intended artifacts:
  examples for central training versus weights for federated training.
- [ ] Run a small deterministic multi-shard quality and timing matrix, then
  compare it with the relevant SwanLake evaluation patterns before scaling up.
