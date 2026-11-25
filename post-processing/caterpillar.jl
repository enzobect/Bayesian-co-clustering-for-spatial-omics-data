include("utils.jl")
using CairoMakie
using LaTeXStrings




# choose a spot and a gene to match the cluster indicies of
# different runs based ont these.
const gene_id = 5
const spot_id = 5


const wide_slice_bounds = [100,10_000]
const zoom_slice_bounds = [1000,1500]
const opacity = .5
const figure_file_name = "traces.png"
function main()

    # arrange the plots
    fig = Figure()
    ax1 = Axis(fig[1,1], title=L"\text{Caterpillar wide view}(\mu)")
    ax2 = Axis(fig[1,2], title=L"\text{Caterpillar stretched view}(\mu)")
    ax3 = Axis(fig[2,1], title=L"\text{Caterpillar wide view}(\sigma^2)")
    ax4 = Axis(fig[2,2], title=L"\text{Caterpillar stretched view}(\sigma^2)")

    # choose the histories of interest
    file_paths = choose_history_files()

    run_to_color = Makie.wong_colors()[1:length(file_paths)]


    for (path,color) in zip(file_paths,run_to_color)
        history = load_history(path)

        # match the block based on the choice of spot and gene
        spot_cluster_id = last(history).w[spot_id]
        gene_cluster_id = last(history).z[gene_id]

        # extract the traces
        μ_vals = zeros(length(history))
        σ_vals = zeros(length(history))
        for i in 1:length(history)
            μ_vals[i] = history[i].μ[spot_cluster_id,gene_cluster_id]
            σ_vals[i] = history[i].σ2[spot_cluster_id,gene_cluster_id]
        end

        wide_slice = range(wide_slice_bounds[1],wide_slice_bounds[2])
        zoomed_in_slice = range(zoom_slice_bounds[1],zoom_slice_bounds[2])

        # superimpose the traces with reduced opacity
        # # μ
        lines!(ax1,wide_slice,μ_vals[wide_slice],alpha=opacity,color=color)
        lines!(ax2,zoomed_in_slice,μ_vals[zoomed_in_slice],alpha=opacity,color=color)
        # # σ
        lines!(ax3,wide_slice,σ_vals[wide_slice],alpha=opacity,color=color)
        lines!(ax4,zoomed_in_slice,σ_vals[zoomed_in_slice],alpha=opacity,color=color)

    end

    # legend on the RHS of the plots
    run_elements = [LineElement(color = c, linewidth = 2) for c in run_to_color]
    Legend(fig[1:2,3], run_elements, [string(i) for i in 1:length(file_paths)], "Chain Index")

    save(joinpath(FIGURES_DIR,figure_file_name), fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
