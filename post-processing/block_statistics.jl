using Statistics
using DataFrames

function block_statistics(
    y::AbstractMatrix,
    spot_labels::AbstractVector,
    gene_labels::AbstractVector
)
    # Data rows must represent spots; columns must represent genes.
    size(y, 1) == length(spot_labels) ||
        error("Spot labels do not match the number of data rows.")

    size(y, 2) == length(gene_labels) ||
        error("Gene labels do not match the number of data columns.")

    results = DataFrame(
        spot_cluster = Int[],
        gene_cluster = Int[],
        n_spots = Int[],
        n_genes = Int[],
        n_values = Int[],
        mean_expression = Union{Missing, Float64}[],
        variance_expression = Union{Missing, Float64}[]
    )

    for r in sort(unique(spot_labels))
        spots = findall(==(r), spot_labels)

        for k in sort(unique(gene_labels))
            genes = findall(==(k), gene_labels)

            # Collect all expression values in this block.
            values = vec(y[spots, genes])
            n = length(values)

            # Sample variance uses the denominator n - 1.
            block_mean = n > 0 ? mean(values) : missing
            block_variance = n > 1 ? var(values; corrected=true) : missing

            push!(results, (
                Int(r),
                Int(k),
                length(spots),
                length(genes),
                n,
                block_mean,
                block_variance
            ))
        end
    end

    return results
end