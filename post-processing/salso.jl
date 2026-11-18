using RCall

function perform_salso(
    group_iter::AbstractMatrix{<:Integer},
    cols;
    max_clusters::Int,
    n_runs::Int = 16,
    seed::Int = 123,
    n_cores::Int = 0          # salso: 0 = use all CPU cores (the n_runs are spread over them)
)
    isempty(cols) && error("No samples remain after burn-in and thinning.")

    # Julia currently stores: items × iterations.
    # R SALSO expects: sampled iterations × items.
    draws = Int.(permutedims(group_iter[:, cols]))

    @rput draws max_clusters n_runs seed n_cores

    R"""
    set.seed(seed)

    salso_fit <- salso::salso(
        draws,
        loss = salso::VI(),
        maxNClusters = max_clusters,
        nRuns = n_runs,
        nCores = n_cores
    )

    salso_labels <- as.integer(salso_fit)

    salso_loss <- salso::partition.loss(
        draws,
        salso_fit,
        loss = salso::VI()
    )
    """

    raw_labels = Int.(rcopy(R"salso_labels"))

    # Ensure labels are consecutive integers starting at 1.
    ids = sort(unique(raw_labels))
    remap = Dict(label => i for (i, label) in enumerate(ids))
    labels = [remap[label] for label in raw_labels]

    return labels, length(ids), rcopy(R"salso_loss")
end

# posterior expected VI loss of a fixed partition `labels` under the draws in group_iter (items × samples)
function partition_expected_loss(group_iter::AbstractMatrix{<:Integer}, labels::AbstractVector{<:Integer})
    draws = Int.(permutedims(group_iter))
    estimate = Int.(labels)
    @rput draws estimate
    return rcopy(R"salso::partition.loss(draws, estimate, loss = salso::VI())")
end