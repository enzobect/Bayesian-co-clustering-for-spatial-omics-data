# Figures of the toy examples: true vs estimated clusters.
#
# Input: the histories saved by "Pott's Model simulation/toy_simulation.jl" with
# "save_history": true (as in toy_config.json):
#     julia -t 4 "Pott's Model simulation/toy_simulation.jl"
#
# Usage (from any folder):
#     julia post-processing/toy_examples_figures.jl [path/to/results_dir]
# (default ../../toy_final_results, relative to this folder = output_dir of toy_config.json)
#
# Requires R with the salso package (install.packages("salso")), called through RCall.
#
# For every toy example and parameter setting it pools the saved draws of all chains
# (they are already post burn-in and thinned), takes the SALSO point estimate (VI loss)
# of the spot and of the gene partition, and writes to <results_dir>/figures/:
#   <example>__<params>.svg   left: true spot clusters, right: estimated spot clusters,
#                             below: true vs estimated cluster of each gene
#   toy_figures_summary.csv   number of clusters, ARI against the truth, VI losses
# and, for the presentation, when several β were run with the same other parameters:
#   <example>__beta_sweep.svg truth + estimated spot map for every β, and the gene
#                             clusters for every β (one strip per β)
#   <example>__ari_trace.svg  ARI of every post burn-in draw, one line per chain,
#                             coloured by β (convergence / mixing check)
#   ari_vs_beta.svg           ARI of the SALSO estimate vs β, one line per example
# Estimated clusters are relabelled to best match the true ones, so the same colour
# means "the same cluster" in all panels; extra clusters get new colours.
# The figures are SVG: PowerPoint / Keynote insert them directly (Insert > Pictures).
include("colormap.jl")   # hex_geometry, cluster_colors, perform_salso, Potts types (MCMC_State)

const TOY_EXAMPLES_DIR = "../toy_examples"
const TOY_COORDS_FILE = joinpath(TOY_EXAMPLES_DIR, "bin_centers.csv")
const DEFAULT_TOY_RESULTS = "../../toy_final_results"
const MAX_CLUSTERS = 20

"""Adjusted Rand index (1 = identical partitions, ≈0 = random)."""
function adjusted_rand_index(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    ia = Dict(v => i for (i, v) in enumerate(unique(a)))
    ib = Dict(v => i for (i, v) in enumerate(unique(b)))
    C = zeros(Int, length(ia), length(ib))
    for i in eachindex(a, b)
        C[ia[a[i]], ib[b[i]]] += 1
    end
    c2(x) = x * (x - 1) / 2
    s_ij = sum(c2, C)
    s_a = sum(c2, sum(C, dims=2))
    s_b = sum(c2, sum(C, dims=1))
    expected = s_a * s_b / c2(length(a))
    max_index = (s_a + s_b) / 2
    return max_index == expected ? 1.0 : (s_ij - expected) / (max_index - expected)
end

"""Relabel `est` (labels 1:k) so that its clusters take the label of the true cluster
they overlap most (greedy, one-to-one); unmatched clusters get labels after the true ones."""
function match_labels(est::Vector{Int}, truth::Vector{Int})
    ke, kt = maximum(est), maximum(truth)
    C = zeros(Int, ke, kt)
    for i in eachindex(est, truth)
        C[est[i], truth[i]] += 1
    end
    newlab = zeros(Int, ke)
    used = falses(kt)
    for idx in sortperm(vec(C); rev=true)
        e, t = Tuple(CartesianIndices(C)[idx])
        if newlab[e] == 0 && !used[t] && C[idx] > 0
            newlab[e] = t
            used[t] = true
        end
    end
    next = kt
    for e in 1:ke
        newlab[e] == 0 && (newlab[e] = (next += 1))
    end
    return newlab[est]
end

# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------
function honeycomb_panel!(io, coords, labels, colors, x0, y0, width, title)
    R, theta0, _ = hex_geometry(coords)
    vang = [theta0 + pi / 6 + m * pi / 3 for m in 0:5]
    xmin, xmax = extrema(coords[:, 1]); ymin, ymax = extrema(coords[:, 2])
    pad = 1.2 * R
    s = width / (xmax - xmin + 2pad)
    height = (ymax - ymin + 2pad) * s
    tx(x) = x0 + (x - xmin + pad) * s
    ty(y) = y0 + 24 + (ymax - y + pad) * s
    vx = R .* cos.(vang) .* s
    vy = -R .* sin.(vang) .* s
    println(io, """<text x="$(x0 + width / 2)" y="$(y0 + 16)" font-size="15" font-weight="bold" text-anchor="middle">$title</text>""")
    for c in sort(unique(labels))
        println(io, """<g fill="$(colors[c])" stroke="$(colors[c])" stroke-width="0.4" stroke-linejoin="round">""")
        for i in findall(==(c), labels)
            cx, cy = tx(coords[i, 1]), ty(coords[i, 2])
            pts = join((@sprintf("%.1f,%.1f", cx + vx[m], cy + vy[m]) for m in 1:6), " ")
            println(io, """<polygon points="$pts"/>""")
        end
        println(io, "</g>")
    end
    return height + 24
end

function write_toy_svg(path, coords, w_true, w_est, z_true, z_est, title, subtitle)
    k = max(maximum(w_true), maximum(w_est), maximum(z_true), maximum(z_est))
    colors = cluster_colors(k)
    panel_w, gap, left, top = 520.0, 40.0, 20.0, 64.0
    io_body = IOBuffer()
    h1 = honeycomb_panel!(io_body, coords, w_true, colors, left, top, panel_w, "True spot clusters")
    honeycomb_panel!(io_body, coords, w_est, colors, left + panel_w + gap, top, panel_w,
                     "Estimated spot clusters (k = $(length(unique(w_est))))")

    # gene strips: genes ordered by true cluster
    G = length(z_true)
    order = sortperm(z_true; alg=MergeSort)
    gy = top + h1 + 30
    label_w = 90.0
    cell = (2panel_w + gap - label_w) / G
    println(io_body, """<text x="$left" y="$(gy - 10)" font-size="15" font-weight="bold">Gene clusters (k = $(length(unique(z_est))) estimated), genes sorted by true cluster</text>""")
    for (row, (name, z)) in enumerate((("true", z_true), ("estimated", z_est)))
        y = gy + (row - 1) * 30
        println(io_body, """<text x="$left" y="$(y + 17)" font-size="13">$name</text>""")
        for (p, g) in enumerate(order)
            println(io_body, """<rect x="$(@sprintf("%.1f", left + label_w + (p - 1) * cell))" y="$y" width="$(@sprintf("%.1f", cell - 1))" height="24" fill="$(colors[z[g]])"/>""")
        end
    end

    W = left + 2panel_w + gap + 20
    H = gy + 70
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(round(Int, W))" height="$(round(Int, H))" viewBox="0 0 $(round(Int, W)) $(round(Int, H))" font-family="sans-serif">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="26" font-size="18" font-weight="bold">$title</text>""")
        println(io, """<text x="$left" y="48" font-size="13" fill="#444">$subtitle</text>""")
        write(io, take!(io_body))
        println(io, "</svg>")
    end
end

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
fmt_params(p) = join(["$k = $(p[k])" for k in (:R, :K, :beta, :mu_0, :lambda, :alpha_0, :beta_0) if haskey(p, k)], ", ")
params_key(p::Dict) = Dict(Symbol(k) => v for (k, v) in p)

function process_setting(example, tag, files, coords, out_dir)
    w_draws = Vector{Int}[]
    z_draws = Vector{Int}[]
    traces = []   # per chain: (iterations, ARI spots, ARI genes)
    local w_true, z_true, params
    for f in files
        data = load(f)
        w_true, z_true, params = data["w_true"], data["z_true"], params_key(data["params"])
        for s in data["history"]
            push!(w_draws, Int.(s.w))
            push!(z_draws, Int.(s.z))
        end
        if haskey(data, "ari_w")   # older history files have no ARI trace
            it = data["burn_in"] .+ data["thin"] .* (1:length(data["ari_w"]))
            push!(traces, (it, data["ari_w"], data["ari_z"]))
        end
    end
    W = reduce(hcat, w_draws)
    Z = reduce(hcat, z_draws)
    n = size(W, 2)
    @assert size(coords, 1) == size(W, 1) "bin_centers.csv has $(size(coords, 1)) spots, the chains $(size(W, 1))"
    println("$example | $tag: $(length(files)) chain(s), $n draws")

    w_est, kw, loss_w = perform_salso(W, 1:n; max_clusters=MAX_CLUSTERS)
    z_est, kz, loss_z = perform_salso(Z, 1:n; max_clusters=MAX_CLUSTERS)
    ari_w = adjusted_rand_index(w_est, w_true)
    ari_z = adjusted_rand_index(z_est, z_true)
    w_est = match_labels(w_est, w_true)
    z_est = match_labels(z_est, z_true)
    @printf("   spots: k = %d, ARI = %.3f | genes: k = %d, ARI = %.3f\n", kw, ari_w, kz, ari_z)

    svg = joinpath(out_dir, "$(example)__$(tag).svg")
    write_toy_svg(svg, coords, w_true, w_est, z_true, z_est,
                  replace(example, "toy_example_" => "", "_" => " "),
                  @sprintf("%s  |  ARI spots %.3f, ARI genes %.3f  |  %d chains, %d draws, SALSO (VI)",
                           fmt_params(params), ari_w, ari_z, length(files), n))
    println("   saved $(abspath(svg))")
    row = (; example, params=tag, beta=Float64(params[:beta]), n_chains=length(files), n_draws=n,
           spot_clusters=kw, ari_spots=ari_w, spot_VI_loss=loss_w,
           gene_clusters=kz, ari_genes=ari_z, gene_VI_loss=loss_z)
    # everything the presentation figures need
    est = (; row..., others=delete!(copy(params), :beta), w_true, z_true, w_est, z_est, traces)
    return row, est
end

# --------------------------------------------------------------------------
# Presentation figures
# --------------------------------------------------------------------------
# chart colours (validated categorical palette, fixed order) and an ordinal blue ramp for β
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd1", "#7a7a75"]
const BETA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
const INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
beta_colors(n) = n <= length(BETA_RAMP) ? BETA_RAMP[round.(Int, range(1, length(BETA_RAMP), length=n))] :
                 error("more than $(length(BETA_RAMP)) β values: extend BETA_RAMP")
fmt_beta(b) = @sprintf("%g", b)
xml(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

svg_open(io, W, H) = println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(round(Int, W))" height="$(round(Int, H))" viewBox="0 0 $(round(Int, W)) $(round(Int, H))" font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="white"/>""")

"""Truth + one estimated spot map per β (grid of panels), then one gene strip per β."""
function write_beta_sweep_svg(path, coords, sweep, title)
    w_true, z_true = sweep[1].w_true, sweep[1].z_true
    k = maximum(max(maximum(s.w_est), maximum(s.z_est)) for s in sweep)
    colors = cluster_colors(max(k, maximum(w_true), maximum(z_true)))
    ncol, panel_w, gap, left, top = 3, 360.0, 24.0, 20.0, 70.0
    body = IOBuffer()
    panels = [("Truth", w_true); [("β = $(fmt_beta(s.beta))  ·  ARI $(@sprintf("%.2f", s.ari_spots))", s.w_est) for s in sweep]]
    h = 0.0
    for (p, (t, w)) in enumerate(panels)
        r, c = divrem(p - 1, ncol)
        h = honeycomb_panel!(body, coords, w, colors, left + c * (panel_w + gap), top + r * (h + 16), panel_w, t)
    end
    nrow_panels = cld(length(panels), ncol)
    W = left + ncol * panel_w + (ncol - 1) * gap + 20

    G = length(z_true)
    order = sortperm(z_true; alg=MergeSort)
    gy = top + nrow_panels * (h + 16) + 30
    label_w = 150.0
    cell = (W - left - 20 - label_w) / G
    println(body, """<text x="$left" y="$(gy - 12)" font-size="15" font-weight="bold" fill="$INK">Gene clusters (genes sorted by true cluster)</text>""")
    strips = [("Truth", z_true); [("β = $(fmt_beta(s.beta))  ·  ARI $(@sprintf("%.2f", s.ari_genes))", s.z_est) for s in sweep]]
    for (row, (name, z)) in enumerate(strips)
        y = gy + (row - 1) * 26
        println(body, """<text x="$left" y="$(y + 15)" font-size="13" fill="$INK">$name</text>""")
        for (p, g) in enumerate(order)
            println(body, """<rect x="$(@sprintf("%.1f", left + label_w + (p - 1) * cell))" y="$y" width="$(@sprintf("%.1f", max(cell - 1, 0.5)))" height="20" fill="$(colors[z[g]])"/>""")
        end
    end
    H = gy + length(strips) * 26 + 16
    open(path, "w") do io
        svg_open(io, W, H)
        println(io, """<text x="$left" y="28" font-size="20" font-weight="bold" fill="$INK">$(xml(title))</text>""")
        println(io, """<text x="$left" y="50" font-size="13" fill="$INK2">$(xml(fmt_params(sweep[1].others)))  ·  SALSO point estimate (VI loss), colours matched to the truth</text>""")
        write(io, take!(body))
        println(io, "</svg>")
    end
end

"""Small multiples of line charts sharing one y-axis (ARI).
`panels` = [(subtitle, [(name, xs, ys, color), ...]), ...]; `legend` = [(name, color), ...]."""
function write_line_chart_svg(path, title, subtitle, xlabel, panels, legend; xticks=nothing, markers=false)
    pw, ph, gap, left, top = 420.0, 260.0, 60.0, 64.0, 110.0
    W = left + length(panels) * pw + (length(panels) - 1) * gap + 30
    H = top + ph + 64
    allx = [x for (_, ss) in panels for s in ss for x in s[2]]
    ally = [y for (_, ss) in panels for s in ss for y in s[3]]
    xmin, xmax = extrema(allx)
    xmax == xmin && (xmax = xmin + 1)
    ymin = min(0.0, floor(minimum(ally) * 10) / 10)
    yticks = ymin:(ymin < 0 ? 0.25 : 0.2):1.0
    io = IOBuffer()
    svg_open(io, W, H)
    println(io, """<text x="20" y="28" font-size="20" font-weight="bold" fill="$INK">$(xml(title))</text>""")
    println(io, """<text x="20" y="50" font-size="13" fill="$INK2">$(xml(subtitle))</text>""")
    # legend: one row above the charts
    lx = 20.0
    for (name, col) in legend
        println(io, """<rect x="$lx" y="66" width="14" height="4" rx="2" fill="$col"/><text x="$(lx + 20)" y="72" font-size="13" fill="$INK">$(xml(name))</text>""")
        lx += 34 + 7.2 * length(name)
    end
    for (p, (sub, series)) in enumerate(panels)
        x0 = left + (p - 1) * (pw + gap)
        sx(x) = x0 + (x - xmin) / (xmax - xmin) * pw
        sy(y) = top + (1 - (y - ymin) / (1 - ymin)) * ph
        println(io, """<text x="$x0" y="$(top - 14)" font-size="15" font-weight="bold" fill="$INK">$(xml(sub))</text>""")
        for t in yticks
            println(io, """<line x1="$x0" x2="$(x0 + pw)" y1="$(sy(t))" y2="$(sy(t))" stroke="$(t == 0 ? "#b5b4ae" : GRID)" stroke-width="1"/>""")
            p == 1 && println(io, """<text x="$(x0 - 8)" y="$(sy(t) + 4)" font-size="12" fill="$INK2" text-anchor="end">$(@sprintf("%g", round(t, digits=2)))</text>""")
        end
        xt = something(xticks, xmax - xmin >= 8 ? unique(round.(Int, range(xmin, xmax, length=5))) :
                                                  range(xmin, xmax, length=5))
        for t in xt
            println(io, """<text x="$(sx(t))" y="$(top + ph + 20)" font-size="12" fill="$INK2" text-anchor="middle">$(@sprintf("%g", round(t, digits=2)))</text>""")
        end
        println(io, """<text x="$(x0 + pw / 2)" y="$(top + ph + 44)" font-size="13" fill="$INK2" text-anchor="middle">$(xml(xlabel))</text>""")
        p == 1 && println(io, """<text transform="translate(18 $(top + ph / 2)) rotate(-90)" font-size="13" fill="$INK2" text-anchor="middle">ARI vs truth</text>""")
        for (_, xs, ys, col) in series
            pts = join((@sprintf("%.1f,%.1f", sx(x), sy(y)) for (x, y) in zip(xs, ys)), " ")
            println(io, """<polyline points="$pts" fill="none" stroke="$col" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>""")
            if markers
                for (x, y) in zip(xs, ys)
                    println(io, """<circle cx="$(sx(x))" cy="$(sy(y))" r="4.5" fill="$col" stroke="white" stroke-width="2"/>""")
                end
            end
        end
    end
    println(io, "</svg>")
    write(path, take!(io))
end

short_name(ex) = replace(ex, "toy_example_" => "", "_spot_clust_" => " · spots ", "_gene_clust_" => " · genes ")

function presentation_figures(ests, coords, out_dir)
    # a "sweep" = same example and same parameters except β, with ≥ 2 values of β
    sweeps = Dict{Tuple{String,String},Vector{Any}}()
    for e in ests
        push!(get!(sweeps, (e.example, fmt_params(e.others)), []), e)
    end
    filter!(kv -> length(kv[2]) >= 2, sweeps)
    isempty(sweeps) && return println("Only one β per example: no β-sweep figures.")

    for ((example, _), sw) in sweeps
        sort!(sw, by = s -> s.beta)
        tag = length(sweeps) == length(unique(first.(keys(sweeps)))) ? "" : "_" * replace(fmt_params(sw[1].others), r"[^A-Za-z0-9]+" => "")
        f = joinpath(out_dir, "$(example)__beta_sweep$tag.svg")
        write_beta_sweep_svg(f, coords, sw, short_name(example))
        println("saved $(abspath(f))")

        any(isempty(s.traces) for s in sw) && continue
        cols = beta_colors(length(sw))
        panels = [(lab, [("β = $(fmt_beta(s.beta))", tr[1], tr[i], cols[j]) for (j, s) in enumerate(sw) for tr in s.traces])
                  for (lab, i) in (("Spot clusters", 2), ("Gene clusters", 3))]
        f = joinpath(out_dir, "$(example)__ari_trace$tag.svg")
        write_line_chart_svg(f, "$(short_name(example)): ARI along the chains",
            "Post burn-in draws, one line per chain ($(length(sw[1].traces)) per β)", "iteration",
            panels, [("β = $(fmt_beta(s.beta))", cols[j]) for (j, s) in enumerate(sw)])
        println("saved $(abspath(f))")
    end

    # ARI vs β, one line per example (only sweeps with the most common β grid)
    sws = sort(collect(values(sweeps)), by = s -> s[1].example)
    length(sws) > length(SERIES_COLORS) && (sws = sws[1:length(SERIES_COLORS)])
    betas = sort(unique(s.beta for sw in sws for s in sw))
    panels = [(lab, [(short_name(sw[1].example), [s.beta for s in sw], [getfield(s, fld) for s in sw], SERIES_COLORS[j])
                     for (j, sw) in enumerate(sws)])
              for (lab, fld) in (("Spot clusters", :ari_spots), ("Gene clusters", :ari_genes))]
    f = joinpath(out_dir, "ari_vs_beta.svg")
    write_line_chart_svg(f, "Recovery of the true clusters vs Potts β",
        "ARI of the SALSO point estimate against the truth  ·  $(fmt_params(sws[1][1].others))", "β",
        panels, [(short_name(sw[1].example), SERIES_COLORS[j]) for (j, sw) in enumerate(sws)];
        xticks=betas, markers=true)
    println("saved $(abspath(f))")
end

function toy_figures_main()
    # resolve the results path from where the user launched, then work from this folder
    # so that TOY_EXAMPLES_DIR and the default results path are always valid
    results = isempty(ARGS) ? joinpath(@__DIR__, DEFAULT_TOY_RESULTS) : abspath(ARGS[1])
    cd(@__DIR__)
    hist_dir = joinpath(results, "histories")
    isdir(hist_dir) || error("No histories in $(abspath(hist_dir)). Run \"Pott's Model simulation/toy_simulation.jl\" with \"save_history\": true first.")
    out_dir = joinpath(results, "figures")
    mkpath(out_dir)
    coords = Float64.(Matrix(DataFrame(CSV.File(TOY_COORDS_FILE))[:, 2:3]))

    rows = []
    ests = []
    for example in sort(filter(d -> isdir(joinpath(hist_dir, d)), readdir(hist_dir)))
        files = filter(f -> endswith(f, ".jld2"), readdir(joinpath(hist_dir, example); join=true))
        # group the replicates of the same parameters: "<tag>_rep<r>.jld2"
        groups = Dict{String,Vector{String}}()
        for f in files
            push!(get!(groups, replace(basename(f), r"_rep\d+\.jld2$" => ""), String[]), f)
        end
        for tag in sort(collect(keys(groups)))
            row, est = process_setting(example, tag, sort(groups[tag]), coords, out_dir)
            push!(rows, row)
            push!(ests, est)
        end
    end
    summary = DataFrame(rows)
    CSV.write(joinpath(out_dir, "toy_figures_summary.csv"), summary)
    show(stdout, summary; allcols=true, summary=false, eltypes=false)
    println()
    presentation_figures(ests, coords, out_dir)
    println("\nFigures in $(abspath(out_dir))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    toy_figures_main()
end
