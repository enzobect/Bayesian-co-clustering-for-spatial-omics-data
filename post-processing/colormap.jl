using Printf
using NearestNeighbors    # lattice spacing / orientation for the honeycomb
using Statistics
using CSV
using DataFrames
using Dates
include("utils.jl")
include("clustering.jl")
include("salso.jl")

#TODO: move to a more appropriate module/config file
const LINKAGE        = :average       # :average (UPGMA) is the usual choice for 1 - P
const RESULTS_DIR   = "../../potts_history"

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


function main()

    #load the chain
    history = choose_history_and_load()
    group_iter::Matrix{Int} = states_to_cluster_mat(history)

    # load the coordinates
    coords_df = CSV.read(COORDS_FILE, DataFrames.DataFrame)
    coords = Float64.(Matrix(coords_df[:, 2:3]))

    N, max_iters = size(group_iter)
    @assert size(coords, 1) == N "coordinates ($(size(coords, 1))) and group_iter ($N) disagree on the number of spots"

    # choose a subset
    burn = floor(Int, BURN_IN_FRAC * max_iters)
    cols = (burn + 1):THIN:max_iters
    @printf("   %d spots, %d iterations; keeping %d samples (burn-in %d, thin %d)\n", N, max_iters, length(cols), burn, THIN)

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

    # Collect gene cluster labels from every saved MCMC iteration.
    # Rows represent genes; columns represent iterations.
    gene_iter = hcat([state.z for state in history]...)

    # Find the final grouping of genes using the same selected iterations.
    gene_labels, gene_k, gene_loss = perform_salso(
        gene_iter,
        cols;
        max_clusters = 20
    )

    println("SALSO selected $gene_k gene clusters")
    println("Gene posterior expected VI loss: $gene_loss")

    colors = cluster_colors(k)

    timestamp =Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    mkpath(FIGURES_DIR)
    write_honeycomb_svg(joinpath(FIGURES_DIR,"bison_honeycomb-$timestamp.svg"), coords, labels, colors; title = "Pott's Model's spot clusters (k = $k)")

    out = DataFrames.DataFrame(id = coords_df[:, 1], x = coords[:, 1], y = coords[:, 2], cluster = labels)
    # CSV.write("bison_cluster_labels.csv", out)
    println("Saved figure")
    println("location: $(abspath(FIGURES_DIR))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
