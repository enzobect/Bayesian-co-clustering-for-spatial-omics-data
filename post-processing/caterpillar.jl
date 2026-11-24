include("../Pott's Model simulation/Potts_MCMC_sampler.jl")   # MCMC_State, so JLD2 can load the history
include("utils.jl")
using CairoMakie
using LaTeXStrings

# arbitrary choice of the clusters for the plots
const gene_cluster_id = 5
const spot_cluster_id = 5
const wide_slice_bounds = [100,10_000]
const zoom_slice_bounds = [1000,1500]

function main()
    file_paths = choose_history_files()
    history = load_history(file_paths[1])

    isempty(history) && error("The selected history is empty")

    μ_size = size(history[1].μ)
    σ_size = size(history[1].σ2)
    if μ_size != σ_size
        error("Inconsistent μ/σ2 dimensions in the history: $(μ_size) vs $(σ_size)")
    end
    if μ_size == (1, 1)
        error("The selected history contains summarized states with μ and σ2 set to 1×1 placeholders. " *
              "Set sampler.save_mu_sigma2 = true and rerun the sampler, or regenerate the history with μ and σ2 preserved.")
    end
    spot_clusters, gene_clusters = μ_size
    if spot_cluster_id > spot_clusters
        error("spot_cluster_id=$spot_cluster_id is outside μ/σ2 dimension 1:$spot_clusters")
    end
    if gene_cluster_id > gene_clusters
        error("gene_cluster_id=$gene_cluster_id is outside μ/σ2 dimension 1:$gene_clusters")
    end


    μ_vals = zeros(length(history))
    σ_vals = zeros(length(history))
    for i in 1:length(history)
        μ_vals[i] = history[i].μ[spot_cluster_id,gene_cluster_id]
        σ_vals[i] = history[i].σ2[spot_cluster_id,gene_cluster_id]
    end

    wide_start = clamp(wide_slice_bounds[1], 1, length(history))
    wide_stop = clamp(wide_slice_bounds[2], wide_start, length(history))
    zoom_start = clamp(zoom_slice_bounds[1], 1, length(history))
    zoom_stop = clamp(zoom_slice_bounds[2], zoom_start, length(history))
    wide_slice = range(wide_start, wide_stop)
    zoomed_in_slice = range(zoom_start, zoom_stop)

    fig = Figure()
    ax1 = Axis(fig[1,1], title=L"\text{Caterpillar wide view}(\mu)")
    ax2 = Axis(fig[1,2], title=L"\text{Caterpillar stretched view}(\mu)")
    ax3 = Axis(fig[2,1], title=L"\text{Caterpillar wide view}(\sigma^2)")
    ax4 = Axis(fig[2,2], title=L"\text{Caterpillar stretched view}(\sigma^2)")

    lines!(ax1,wide_slice,μ_vals[wide_slice])
    lines!(ax2,zoomed_in_slice,μ_vals[zoomed_in_slice])

    lines!(ax3,wide_slice,σ_vals[wide_slice])
    lines!(ax4,zoomed_in_slice,σ_vals[zoomed_in_slice])

    mkpath(FIGURES_DIR)
    save(joinpath(FIGURES_DIR,"caterpilars.png"), fig)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
