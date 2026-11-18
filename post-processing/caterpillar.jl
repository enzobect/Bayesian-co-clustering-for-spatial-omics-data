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


    μ_vals = zeros(length(history))
    σ_vals = zeros(length(history))
    for i in 1:length(history)
        μ_vals[i] = history[i].μ[spot_cluster_id,gene_cluster_id]
        σ_vals[i] = history[i].σ2[spot_cluster_id,gene_cluster_id]
    end

    wide_slice = range(wide_slice_bounds[1],wide_slice_bounds[2])
    zoomed_in_slice = range(zoom_slice_bounds[1],zoom_slice_bounds[2])

    fig = Figure()
    ax1 = Axis(fig[1,1], title=L"\text{Caterpillar wide view}(\mu)")
    ax2 = Axis(fig[1,2], title=L"\text{Caterpillar stretched view}(\mu)")
    ax3 = Axis(fig[2,1], title=L"\text{Caterpillar wide view}(\sigma^2)")
    ax4 = Axis(fig[2,2], title=L"\text{Caterpillar stretched view}(\sigma^2)")

    lines!(ax1,wide_slice,μ_vals[wide_slice])
    lines!(ax2,zoomed_in_slice,μ_vals[zoomed_in_slice])

    lines!(ax3,wide_slice,σ_vals[wide_slice])
    lines!(ax4,zoomed_in_slice,σ_vals[zoomed_in_slice])

    save(joinpath(FIGURES_DIR,"caterpilars.png"), fig)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
