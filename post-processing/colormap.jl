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
const RESULTS_DIR   = "../../potts_history_visium100"   # = path_to_history_files in config_visium100.json

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

# report = list of key => value lines
function write_report(path::AbstractString, lines)
    open(path, "w") do io
        for (key, value) in lines
            println(io, "$key = $value")
        end
    end
end

# SALSO point estimates of spots and genes from items × samples label matrices
function salso_estimates(W, Z)
    labels, k, loss = perform_salso(W, axes(W, 2); max_clusters = 20)
    println("   SALSO: $k spot clusters, posterior expected VI loss $loss")
    gene_labels, gene_k, gene_loss = perform_salso(Z, axes(Z, 2); max_clusters = 20)
    println("   SALSO: $gene_k gene clusters, posterior expected VI loss $gene_loss")
    return (labels = labels, k = k, loss = loss, gene_labels = gene_labels, gene_k = gene_k, gene_loss = gene_loss)
end

function save_estimate(stem, title, coords, est, report_lines, summary_row)
    mkpath(FIGURES_DIR); mkpath(DIAGNOSTICS_DIR)
    fig = joinpath(FIGURES_DIR, "$stem.svg")
    write_honeycomb_svg(fig, coords, est.labels, cluster_colors(est.k); title = title)
    rep = joinpath(DIAGNOSTICS_DIR, "$(stem)_posterior_loss.txt")
    write_report(rep, report_lines)
    append_summary_row(summary_row)
    println("   saved figure: $(abspath(fig))")
    println("   saved report: $(abspath(rep))")
end

# ---------------------------------------------------------------------------
# one chain
# ---------------------------------------------------------------------------
function process_chain(chain, coords)
    info = chain.info
    println("\nChain $(basename(chain.path))  (β = $(info.beta), replicate $(info.replicate), seed $(info.seed))")
    est = salso_estimates(chain.W, chain.Z)
    stem = splitext(basename(chain.path))[1]
    save_estimate(stem, "Pott's Model's spot clusters (β = $(info.beta), chain $(info.replicate), k = $(est.k))",
                  coords, est,
                  Pair{String,Any}["history_file" => abspath(chain.path), "beta" => info.beta, "seed" => info.seed,
                   "replicate" => info.replicate, "spots" => size(chain.W, 1), "iterations" => chain.n_iter,
                   "posterior_samples" => size(chain.W, 2),
                   "spot_clusters" => est.k, "spot_posterior_expected_VI_loss" => est.loss,
                   "gene_clusters" => est.gene_k, "gene_posterior_expected_VI_loss" => est.gene_loss],
                  (file = basename(chain.path), beta = info.beta, replicate = string(info.replicate),
                   seed = string(info.seed), spot_clusters = est.k, spot_VI_loss = est.loss,
                   gene_clusters = est.gene_k, gene_VI_loss = est.gene_loss))
    return est
end

# ---------------------------------------------------------------------------
# chains with the same β pooled into one posterior sample
#   Valid only if every chain targets the same posterior (same data, β, R, K, priors) and has
#   reached the same mode. SALSO only uses partitions, so a different label numbering across
#   chains (label switching) does not matter. Chains with different β are never mixed:
#   they sample different posteriors.
# ---------------------------------------------------------------------------
function process_pooled(β, chains, coords; single_estimates = Dict())
    n = length(chains)
    println("\nPooling $n chains with β = $β: ", join([basename(c.path) for c in chains], ", "))
    @assert length(unique(size(c.W, 1) for c in chains)) == 1 &&
            length(unique(size(c.Z, 1) for c in chains)) == 1 "chains with β = $β have different numbers of spots/genes: not the same data"

    W = reduce(hcat, [c.W for c in chains])
    Z = reduce(hcat, [c.Z for c in chains])
    est = salso_estimates(W, Z)

    # convergence check: the pooled estimate should fit every chain about equally well,
    # and (mode 3) each chain's own estimate should be close to the pooled one
    lines = Pair{String,Any}[]
    for c in chains
        tag = "chain$(c.info.replicate)"
        l_spot = partition_expected_loss(c.W, est.labels)
        l_gene = partition_expected_loss(c.Z, est.gene_labels)
        push!(lines, "$(tag)_file" => basename(c.path))
        push!(lines, "$(tag)_spot_expected_VI_loss_of_pooled_estimate" => l_spot)
        push!(lines, "$(tag)_gene_expected_VI_loss_of_pooled_estimate" => l_gene)
        @printf("   %s: expected VI loss of the pooled estimate  spots %.4f  genes %.4f", tag, l_spot, l_gene)
        if haskey(single_estimates, c.path)
            s = single_estimates[c.path]
            d_spot = vi_distance(s.labels, est.labels)
            d_gene = vi_distance(s.gene_labels, est.gene_labels)
            push!(lines, "$(tag)_spot_VI_own_estimate_vs_pooled" => d_spot)
            push!(lines, "$(tag)_gene_VI_own_estimate_vs_pooled" => d_gene)
            @printf("   | VI(own estimate, pooled)  spots %.4f  genes %.4f", d_spot, d_gene)
        end
        println()
    end
    println("   (much larger values for some chains = chains in different modes: the pooled result mixes them)")

    stem = "pooled_beta$(beta_tag(β))_$(n)chains"
    files = [basename(c.path) for c in chains]
    save_estimate(stem, "Pott's Model's spot clusters (β = $β, $n chains pooled, k = $(est.k))", coords, est,
                  vcat(Pair{String,Any}["beta" => β, "chains" => n, "spots" => size(W, 1),
                        "posterior_samples" => size(W, 2),
                        "samples_per_chain" => join([size(c.W, 2) for c in chains], ","),
                        "spot_clusters" => est.k, "spot_posterior_expected_VI_loss" => est.loss,
                        "gene_clusters" => est.gene_k, "gene_posterior_expected_VI_loss" => est.gene_loss], lines),
                  (file = "pooled: " * join(files, " + "), beta = β, replicate = "pooled($n)",
                   seed = join([string(c.info.seed) for c in chains], ";"),
                   spot_clusters = est.k, spot_VI_loss = est.loss,
                   gene_clusters = est.gene_k, gene_VI_loss = est.gene_loss))
    return est
end

beta_tag(β) = replace(@sprintf("%g", β), "." => "p")   # 2.5 -> "2p5", as in the chain file names

# one row per processed chain / pooled group: compare β values / chains at a glance
function append_summary_row(row::NamedTuple)
    path = joinpath(DIAGNOSTICS_DIR, "beta_summary.csv")
    CSV.write(path, DataFrame([row]); append = isfile(path))
    println("   appended to summary: $(abspath(path))")
end

# ---------------------------------------------------------------------------
# trace plots of μ and σ² (after burn-in)
# ---------------------------------------------------------------------------
function process_traces(chain)
    if chain.mu === nothing
        println("   no μ / σ² in this file (old format): no trace plots")
        return
    end
    # one caterpillar figure per block (r, k) that is non-empty after burn-in, in a folder per chain
    dir = joinpath(TRACES_DIR, splitext(basename(chain.path))[1])
    mkpath(dir)
    blocks = findall(chain.occupied)
    for I in blocks
        r, k = Tuple(I)
        write_caterpillar_svg(joinpath(dir, @sprintf("r%02d_k%02d.svg", r, k)),
                              view(chain.mu, r, k, :), view(chain.sigma2, r, k, :), chain.burn;
                              title = "β = $(chain.info.beta), chain $(chain.info.replicate): spot cluster r = $r, gene cluster k = $k")
    end
    println("   saved $(length(blocks)) caterpillar plots (one per block r, k): $(abspath(dir))")
    n_skip = length(chain.occupied) - length(blocks)
    n_skip > 0 && println("   $n_skip of $(length(chain.occupied)) blocks were empty in some iteration ",
                          "after burn-in (prior draws): not drawn")
end

# ---------------------------------------------------------------------------
function choose_mode(n_files)
    n_files == 1 && return 1
    println("How should the chains be processed?")
    println("1. Each chain separately")
    println("2. Pool chains with the same β into one result")
    println("3. Both (also compares each chain's estimate with the pooled one)")
    choice = try
        parse(Int, strip(readline()))
    catch
        0
    end
    1 <= choice <= 3 || throw(ArgumentError("Please enter 1, 2 or 3."))
    return choice
end

function main()
    coords_df = CSV.read(COORDS_FILE, DataFrames.DataFrame)
    coords = Float64.(Matrix(coords_df[:, 2:3]))

    # optional argument: folder with the chain files (default RESULTS_DIR), e.g. for the smoke test
    files = choose_history_files(isempty(ARGS) ? RESULTS_DIR : ARGS[1])
    mode = choose_mode(length(files))

    chains = Any[]
    for f in files
        println("\nLoading $(basename(f))")
        c = load_chain(f, BURN_IN_FRAC, THIN)
        @assert size(c.W, 1) == size(coords, 1) "coordinates ($(size(coords, 1))) and $(basename(f)) ($(size(c.W, 1))) disagree on the number of spots"
        @printf("   β = %s, %d iterations; keeping %d samples (burn-in %d, thin %d)\n",
                c.info.beta, c.n_iter, length(c.kept), c.burn, THIN)
        process_traces(c)
        push!(chains, c)
    end

    single = Dict()
    if mode in (1, 3)
        for c in chains
            single[c.path] = process_chain(c, coords)
        end
    end
    if mode in (2, 3)
        for (β, group) in group_chains_by_beta(chains)
            if length(group) < 2
                println("\nβ = $β: only one chain, nothing to pool")
                mode == 2 && (single[group[1].path] = process_chain(group[1], coords))
                continue
            end
            process_pooled(β, group, coords; single_estimates = single)
        end
    end
    println("\nfigures: $(abspath(FIGURES_DIR))\ntraces:  $(abspath(TRACES_DIR))\nreports: $(abspath(DIAGNOSTICS_DIR))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
