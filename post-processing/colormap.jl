using Printf
using NearestNeighbors    # lattice spacing / orientation for the honeycomb
using Statistics
using CSV
using DataFrames
include("utils.jl")
include("clustering.jl")
include("salso.jl")

#TODO: move to a more appropriate module/config file
const LINKAGE        = :average       # :average (UPGMA) is the usual choice for 1 - P

const BASE_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                     "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
const HEX_SCALE      = 1.0            # <1 shrinks hexagons slightly to leave gaps
const FLIP_Y         = true           # true: y increases upwards, like a normal scatter plot

const BURN_IN_FRAC   = 0.5            # discard the first 50 % of iterations
const THIN           = 10              # keep every THIN-th iteration after burn-in


function hsl_hex(h, s, l)
    c = (1 - abs(2l - 1)) * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = l - c / 2
    r, g, b = h < 60 ? (c, x, 0) : h < 120 ? (x, c, 0) : h < 180 ? (0, c, x) :
              h < 240 ? (0, x, c) : h < 300 ? (x, 0, c) : (c, 0, x)
    return @sprintf("#%02X%02X%02X", round(Int, 255 * (r + m)), round(Int, 255 * (g + m)), round(Int, 255 * (b + m)))
end

function cluster_colors(k)
    cols = copy(BASE_COLORS)
    for m in 11:k
        push!(cols, hsl_hex(mod((m - 1) * 137.508, 360), 0.65, isodd(m) ? 0.45 : 0.62))
    end
    return cols[1:k]
end

# ---------------------------------------------------------------------------
# 4a. honeycomb
#     Hexagon size and orientation are estimated from the data: the nearest-neighbour distance d of
#     a hexagonal lattice gives circumradius d/sqrt(3); neighbour directions are at theta0 + m*60deg
#     and the hexagon vertices point at theta0 + 30deg + m*60deg.
# ---------------------------------------------------------------------------
function hex_geometry(coords::Matrix{Float64})
    tree = KDTree(coords')
    idxs, dists = knn(tree, coords', 2, true)
    d = median([dists[i][2] for i in axes(coords, 1)])
    z = 0.0 + 0.0im
    for i in axes(coords, 1)
        j = idxs[i][2]
        z += exp(6im * atan(coords[j, 2] - coords[i, 2], coords[j, 1] - coords[i, 1]))
    end
    theta0 = angle(z) / 6
    return d / sqrt(3), theta0, d
end

function write_honeycomb_svg(path, coords, labels, colors; title = "")
    R, theta0, _ = hex_geometry(coords)
    R *= HEX_SCALE
    vang = [theta0 + pi / 6 + m * pi / 3 for m in 0:5]
    vx = R .* cos.(vang)
    vy = R .* sin.(vang)

    xmin, xmax = extrema(coords[:, 1]); ymin, ymax = extrema(coords[:, 2])
    pad = 1.2 * R
    plot_w = 900.0
    s = plot_w / (xmax - xmin + 2pad)
    plot_h = (ymax - ymin + 2pad) * s
    top = 40.0; legend_w = 170.0
    W = plot_w + legend_w + 20; H = max(plot_h, 30.0 + 22 * length(colors)) + top + 20

    tx(x) = (x - xmin + pad) * s
    ty(y) = FLIP_Y ? top + (ymax - y + pad) * s : top + (y - ymin + pad) * s

    k = length(colors)
    sizes = [count(==(c), labels) for c in 1:k]
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(round(Int, W))" height="$(round(Int, H))" viewBox="0 0 $(round(Int, W)) $(round(Int, H))" font-family="sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="10" y="26" font-size="18" font-weight="bold">$(title)</text>""")
    for c in 1:k
        println(io, """<g fill="$(colors[c])" stroke="$(colors[c])" stroke-width="0.5" stroke-linejoin="round">""")
        for i in findall(==(c), labels)
            cx, cy = tx(coords[i, 1]), ty(coords[i, 2])
            pts = join((@sprintf("%.1f,%.1f", cx + vx[m] * s, cy + (FLIP_Y ? -1 : 1) * vy[m] * s) for m in 1:6), " ")
            println(io, """<polygon points="$pts"/>""")
        end
        println(io, "</g>")
    end
    lx = plot_w + 20
    for c in 1:k
        y = top + 4 + 22 * (c - 1)
        println(io, """<rect x="$lx" y="$y" width="16" height="16" fill="$(colors[c])"/>""")
        println(io, """<text x="$(lx + 24)" y="$(y + 13)" font-size="13">cluster $c (n=$(sizes[c]))</text>""")
    end
    println(io, "</svg>")
    write(path, String(take!(io)))
end

function write_posterior_report(
    path::AbstractString,
    history_paths,
    n_spots::Int,
    n_iterations::Int,
    n_samples::Int,
    spot_k::Int,
    spot_loss,
    gene_k::Int,
    gene_loss,
    info
)
    open(path, "w") do io
        for history_path in history_paths
            println(io, "history_file = $(abspath(history_path))")
        end
        println(io, "beta = $(info.beta)")
        println(io, "seed = $(info.seed)")
        println(io, "replicate = $(info.replicate)")
        println(io, "spots = $n_spots")
        println(io, "iterations = $n_iterations")
        println(io, "posterior_samples = $n_samples")
        println(io, "spot_clusters = $spot_k")
        println(io, "spot_posterior_expected_VI_loss = $spot_loss")
        println(io, "gene_clusters = $gene_k")
        println(io, "gene_posterior_expected_VI_loss = $gene_loss")
    end
end

# history_paths: one file, or several chains with the same β that are merged
function process_history(history_paths, coords)
    println("\nProcessing $(join(basename.(history_paths), ", "))")
    infos = load_run_info.(history_paths)
    info = (beta = infos[1].beta,
            seed = join([i.seed for i in infos], ", "),
            replicate = join([i.replicate for i in infos], ", "))
    println("   β = $(info.beta), seed = $(info.seed), replicate = $(info.replicate)")

    # burn-in and thinning in each chain, then put the chains one after the other
    spot_samples = Matrix{Int}[]
    gene_samples = Matrix{Int}[]
    max_iters = 0
    for history_path in history_paths
        history = load_history(history_path)
        chain_iter::Matrix{Int} = states_to_cluster_mat(history)
        chain_gene_iter = hcat([state.z for state in history]...)

        # choose a subset
        max_iters = size(chain_iter, 2)
        burn = floor(Int, BURN_IN_FRAC * max_iters)
        chain_cols = (burn + 1):THIN:max_iters
        @printf("   %d iterations; keeping %d samples (burn-in %d, thin %d)\n", max_iters, length(chain_cols), burn, THIN)

        push!(spot_samples, chain_iter[:, chain_cols])
        push!(gene_samples, chain_gene_iter[:, chain_cols])
    end
    group_iter = hcat(spot_samples...)
    gene_iter = hcat(gene_samples...)

    N, n_samples = size(group_iter)
    @assert size(coords, 1) == N "coordinates ($(size(coords, 1))) and group_iter ($N) disagree on the number of spots"
    cols = 1:n_samples

    # distance as (1 - posterior prob. of i and j being in the same cluster)
    # D, ns = coclustering_distance(group_iter, cols)


    # TODO: use strategy to be able to switch the clustering algorithm
    # labels,k = preform_heirarchial_clustering(D,LINKAGE)

    # Run SALSO on the selected MCMC samples to find a final grouping
    labels, k, expected_loss = perform_salso(
    group_iter,
    cols;
    max_clusters = 20
    )

    # Display the number of clusters and the loss
    println("SALSO selected $k clusters")
    println("Posterior expected VI loss: $expected_loss")

    # Find the final grouping of genes using the same selected iterations.
    gene_labels, gene_k, gene_loss = perform_salso(
        gene_iter,
        cols;
        max_clusters = 20
    )

    println("SALSO selected $gene_k gene clusters")
    println("Gene posterior expected VI loss: $gene_loss")

    colors = cluster_colors(k)
    mkpath(FIGURES_DIR)
    stem = splitext(basename(history_paths[1]))[1]
    if length(history_paths) > 1
        stem = replace(stem, r"_chain.*" => "") * "_merged"
    end
    write_honeycomb_svg(
        joinpath(FIGURES_DIR, "$stem.svg"),
        coords,
        labels,
        colors;
        title = "Pott's Model's spot clusters (β = $(info.beta), chain $(info.replicate), k = $k)"
    )

    mkpath(DIAGNOSTICS_DIR)
    write_posterior_report(
        joinpath(DIAGNOSTICS_DIR, "$(stem)_posterior_loss.txt"),
        history_paths,
        N,
        max_iters,
        n_samples,
        k,
        expected_loss,
        gene_k,
        gene_loss,
        info
    )
    append_summary_row(stem, info, k, expected_loss, gene_k, gene_loss)
    println("Saved figure: $(abspath(joinpath(FIGURES_DIR, "$stem.svg")))")
    println("Saved posterior loss: $(abspath(joinpath(DIAGNOSTICS_DIR, "$(stem)_posterior_loss.txt")))")
    println("location: $(abspath(FIGURES_DIR))")
end

# one row per processed chain: compare β values / chains at a glance
function append_summary_row(stem, info, k, loss, gene_k, gene_loss)
    path = joinpath(DIAGNOSTICS_DIR, "beta_summary.csv")
    row = DataFrame(file = stem, beta = info.beta, replicate = info.replicate,
                    seed = info.seed, spot_clusters = k, spot_VI_loss = loss,
                    gene_clusters = gene_k, gene_VI_loss = gene_loss)
    CSV.write(path, row; append = isfile(path))
    println("Appended to summary: $(abspath(path))")
end

function main()
    coords_df = CSV.read(COORDS_FILE, DataFrames.DataFrame)
    coords = Float64.(Matrix(coords_df[:, 2:3]))

    for history_paths in choose_history_groups()
        process_history(history_paths, coords)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
