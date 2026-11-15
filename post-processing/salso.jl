using RCall

function perform_salso(
    group_iter::AbstractMatrix{<:Integer},
    cols;
    max_clusters::Int,
    n_runs::Int = 16,
    seed::Int = 123
)
    isempty(cols) && error("No samples remain after burn-in and thinning.")

    # Julia currently stores: items × iterations.
    # R SALSO expects: sampled iterations × items.
    draws = Int.(permutedims(group_iter[:, cols]))

    @rput draws max_clusters n_runs seed

    R"""
    set.seed(seed)

    salso_fit <- salso::salso(
        draws,
        loss = salso::VI(),
        maxNClusters = max_clusters,
        nRuns = n_runs,
        nCores = 1
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