include("../Pott's Model simulation/Potts_MCMC_sampler.jl")
using Clustering   # hclust, cutree
const MIN_CLUSTER_FRAC = 0.005        # branches smaller than this fraction of the spots (min. 10 spots)
const K_SEARCH       = 2:15           # candidate k for the automatic (largest-gap) choice
const N_CLUSTERS     = nothing        # set e.g. 6 to override the automatic choice


function substantial_heights(hc, min_size)
    N = length(hc.order)
    nsize = zeros(Int, N - 1)
    sub = Float64[]
    leafsize(c) = c < 0 ? 1 : nsize[c]
    for st in 1:(N - 1)
        sa, sb = leafsize(hc.merges[st, 1]), leafsize(hc.merges[st, 2])
        nsize[st] = sa + sb
        if sa >= min_size && sb >= min_size
            push!(sub, hc.heights[st])
        end
    end
    return push!(sort(sub; rev = true), 0.0)     # baseline 0 = "always co-clustered"
end

function cluster_gaps(sub, ks)
    return [(m, sub[m - 1] - sub[m]) for m in ks if 2 <= m <= length(sub)]
end

choose_m(gaps) = gaps[argmax([g for (_, g) in gaps])][1]
cut_height(sub, m) = (sub[m - 1] + sub[m]) / 2

# clusters smaller than min_size -> the large cluster with the smallest mean distance
function absorb_small_clusters!(labels::Vector{Int}, D::Matrix{Float64}, min_size::Int)
    ids = sort(unique(labels))
    sizes = Dict(c => count(==(c), labels) for c in ids)
    large = [c for c in ids if sizes[c] >= min_size]
    isempty(large) && return 0
    lidx = Dict(c => findall(==(c), labels) for c in large)
    moved = 0
    for c in ids
        sizes[c] >= min_size && continue
        members = findall(==(c), labels)
        best = large[argmin([mean(@view D[members, lidx[l]]) for l in large])]
        labels[members] .= best
        moved += length(members)
    end
    return moved
end

function write_dendrogram_svg(path, hc, labels, colors, k, hcut, gaps)
    N = length(hc.order)
    pos = zeros(Float64, N)
    for (p, leaf) in enumerate(hc.order)
        pos[leaf] = p
    end
    hts = hc.heights
    hmax = maximum(hts)
    dw, dh = 900.0, 420.0
    left, top = 60.0, 50.0
    px(p) = left + (p - 0.5) / N * dw
    py(h) = top + dh - h / hmax * dh

    nx = zeros(N - 1); rep = zeros(Int, N - 1)
    node(c) = c < 0 ? (pos[-c], 0.0, -c) : (nx[c], hts[c], rep[c])
    paths = Dict{String,IOBuffer}()
    for st in 1:(N - 1)
        xa, ha, ra = node(hc.merges[st, 1])
        xb, hb, _ = node(hc.merges[st, 2])
        h = hts[st]
        nx[st] = (xa + xb) / 2
        rep[st] = ra
        col = h <= hcut ? colors[labels[ra]] : "#555555"
        buf = get!(paths, col) do
            IOBuffer()
        end
        @printf(buf, "M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f ", px(xa), py(ha), px(xa), py(h), px(xb), py(h), px(xb), py(hb))
    end

    gw = 300.0; gx = left + dw + 70
    gmax = maximum(g for (_, g) in gaps)
    W = gx + gw + 30; H = top + dh + 60
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(round(Int, W))" height="$(round(Int, H))" viewBox="0 0 $(round(Int, W)) $(round(Int, H))" font-family="sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="10" y="26" font-size="18" font-weight="bold">Dendrogram of 1 - co-clustering probability: $k clusters</text>""")
    # axis
    println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + dh)" stroke="black"/>""")
    for t in 0:0.25:1
        t * 1.0 <= hmax * 1.0001 || continue
        y = py(t)
        println(io, """<line x1="$(left - 4)" y1="$y" x2="$left" y2="$y" stroke="black"/><text x="$(left - 8)" y="$(y + 4)" font-size="11" text-anchor="end">$(t)</text>""")
    end
    println(io, """<text transform="translate(16,$(top + dh / 2)) rotate(-90)" font-size="12" text-anchor="middle">1 - P(same cluster)</text>""")
    for (col, buf) in paths
        println(io, """<path d="$(String(take!(buf)))" fill="none" stroke="$col" stroke-width="0.8"/>""")
    end
    println(io, """<line x1="$left" y1="$(py(hcut))" x2="$(left + dw)" y2="$(py(hcut))" stroke="red" stroke-dasharray="6,4"/>""")
    println(io, """<text x="$(left + dw)" y="$(py(hcut) - 5)" font-size="12" fill="red" text-anchor="end">cut at $(round(hcut, digits = 3))</text>""")
    println(io, """<text x="$(left + dw / 2)" y="$(top + dh + 22)" font-size="12" text-anchor="middle">spots ($N leaves)</text>""")
    # gap panel
    println(io, """<text x="$gx" y="$(top - 12)" font-size="13" font-weight="bold">Gap between substantial merge heights, by k</text>""")
    bw = gw / length(gaps)
    for (m, (kk, g)) in enumerate(gaps)
        bh = g / gmax * dh
        col = kk == k ? "#E15759" : "#9aa5b1"
        println(io, """<rect x="$(gx + (m - 1) * bw + 1)" y="$(top + dh - bh)" width="$(bw - 2)" height="$bh" fill="$col"/>""")
        println(io, """<text x="$(gx + (m - 0.5) * bw)" y="$(top + dh + 16)" font-size="10" text-anchor="middle">$kk</text>""")
    end
    println(io, """<text x="$(gx + gw / 2)" y="$(top + dh + 36)" font-size="12" text-anchor="middle">number of clusters k</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
end

function preform_heirarchial_clustering(distances::Matrix{Float64}, linkage::Symbol; save_dendogram=false)
    hc = hclust(distances; linkage = linkage)
    N = size(distances)[1]
    @assert issorted(hc.heights) "merge heights are not monotone; use :average, :complete or :ward"

    min_size = max(10, round(Int, MIN_CLUSTER_FRAC * N))
    sub = substantial_heights(hc, min_size)
    gaps = cluster_gaps(sub, K_SEARCH)
    isempty(gaps) && error("no substantial merges found; lower MIN_CLUSTER_FRAC or set N_CLUSTERS")
    auto_k = choose_m(gaps)
    k = N_CLUSTERS === nothing ? auto_k : N_CLUSTERS
    @assert 2 <= k <= length(sub) "N_CLUSTERS must be between 2 and $(length(sub)) for this dendrogram"

    hcut = cut_height(sub, k)
    raw = cutree(hc; h = hcut)
    labels = copy(raw)
    moved = absorb_small_clusters!(labels, distances, min_size)
    moved > 0 && println("   $moved outlier spots assigned to their nearest cluster")
    ids = sort(unique(labels))
    order = sortperm([count(==(c), labels) for c in ids]; rev = true)   # renumber: 1 = largest
    remap = Dict(ids[o] => r for (r, o) in enumerate(order))
    labels = [remap[c] for c in labels]
    k = length(ids)
    # TODO: figure out how to choose the colors
    # if save_dendogram
    #     write_dendrogram_svg(joinpath(FIGURES_DIR,"bison_dendrogram.svg"), hc, labels, colors, k, hcut, gaps)
    # end
    return (labels,k)
end
