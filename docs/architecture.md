# Kea Architecture

This is the single source of truth for Kea's architecture. Solid arrows are
implemented; the dashed distributed backend is future work.

```mermaid
classDiagram
    direction LR

    class RunConfig {
        +execution_mode
        +label_mode
        +training_placement
        +dataset
        +rounds
        +label_budget
        +initial_label_fraction
        +seed
    }

    class ProxyTrainingRunner {
        +Run(config, sampler, labeler) ProxyModel
    }

    class ISampler {
        <<interface>>
        +SelectInitial(context, budget, seed)
    }

    class ILabeler {
        <<interface>>
        +Label(candidates)
    }

    class ITrainingExecutionBackend {
        <<interface>>
        +AcquireInitialLabels(request)
        +AcquireUncertainLabels(request)
        +TrainLocalModels(request)
        +BroadcastModel(model)
    }

    class SingleMachineBackend {
        +direct calls to one local worker
    }

    class DuckDbShardWorker {
        <<internal>>
        +local sampling and labeling
        +local labeled-ID state
    }

    class DistributedBackend {
        <<planned>>
        +remote worker dispatch
    }

    class LogisticRegressionTrainer {
        +Train(examples) ProxyModel
    }

    class ProxyModel {
        +weights
        +intercept
        +PredictProbability(embedding)
    }

    class DuckDB {
        <<external>>
    }

    ProxyTrainingRunner --> RunConfig : reads
    ProxyTrainingRunner --> ISampler : initial sampling
    ProxyTrainingRunner --> ILabeler : labels through worker
    ProxyTrainingRunner --> ITrainingExecutionBackend : dispatches rounds
    ProxyTrainingRunner --> LogisticRegressionTrainer : central training
    ProxyTrainingRunner --> ProxyModel : returns and scores

    SingleMachineBackend ..|> ITrainingExecutionBackend
    SingleMachineBackend --> DuckDbShardWorker : direct call
    DuckDbShardWorker --> DuckDB : reads and tracks IDs

    DistributedBackend ..|> ITrainingExecutionBackend
    DistributedBackend ..> DuckDbShardWorker : future remote dispatch
    LogisticRegressionTrainer --> ProxyModel : produces
```

`ProxyTrainingRunner` owns the round schedule and central training. In
single-machine mode it creates an internal `DuckDbShardWorker`, wraps it in
`SingleMachineBackend`, and routes initial and uncertainty-label acquisition
through `ITrainingExecutionBackend`. The missing piece is a real distributed
backend that sends the same requests to multiple remote shard workers.
