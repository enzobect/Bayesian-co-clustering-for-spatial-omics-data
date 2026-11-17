# File to generate the toy examples for the simulation study of the Pott's Model
# We create one simple square hexagonal topologies
# We also look at different cluster attributions for spots and we give different
# sizes of gene clusters. We always have K = 3, R = 3, 30 genes and
# approximately 3000 spots. We test for two different distributions for the sampling of the data: normal and Cauchy, and we sample the parameters of the distributions from a normal-inverse-gamma prior.
# Things we could have done but we did not:
# 1. we could have created a more complex topology with holes, but we did not do it because the distribution of the number of neighbours is already quite close to the one of the human tissue dataset.
# 2. we could have introduced noise in the sampled means and variances within each gene cluster.

library(sf)
library(spdep)
library(tidyverse)
library(ggplot2)
library(stats)

# Fixed seed so the toy examples are reproducible
set.seed(2026)

# Creating the original "square" hexagonal grid
sfc <- st_sfc(st_polygon(list(rbind(c(0, 0), c(1, 0), c(1, 1), c(0, 0)))))
# Increasing `cellsize` decreases the number of cells
hex_grid <- st_make_grid(sfc, cellsize = .01, square = FALSE)
# Check properties
hex_grid

# Visualise grid
plot(hex_grid, col = "grey")

# Creating dataset
hex_sf <- st_sf(id = seq_along(hex_grid), geometry = hex_grid)
#Adding centroids
hex_sf <- cbind(hex_sf, st_coordinates(st_centroid(hex_sf)))

# Creating neighbours' list
neighbours_list <- poly2nb(hex_grid)
neighbours_list[[1]]

# Visualise distribution of number of neighbours
prop.table(table(factor(card(neighbours_list), levels = 0:6)))

# Comparing with the human tissue dataset
nl_human_tissue <- read.csv("Data/Visium_HD_human_tissue/adjecency.csv", header = FALSE)
nb_human_tissue <- nl_human_tissue |>
  mutate_all(function(x) as.integer(x != 0)) |>
  rowSums()
freq_compare <- rbind(
  as.data.frame(prop.table(table(factor(
    nb_human_tissue,
    levels = 0:6
  )))) |>
    mutate(from = "human tissue"),
  as.data.frame(prop.table(table(factor(
    card(neighbours_list),
    levels = 0:6
  )))) |>
    mutate(from = "toy example")
) |>
  rename(number_of_neighbours = "Var1")

ggplot(freq_compare, aes(x = number_of_neighbours, y = Freq, fill = from)) +
  geom_col(position = "dodge")

# Let's incrementally remove some bins to form holes with probability
# proportional to exp(- 4 * mean number of neighbours of your neighbours)
# n_holes <- 100
# hex_grid_emmental <- hex_grid
# for (i in 1:n_holes) {
#   print(paste(i, "out of", n_holes))
#   neighbours_list_emmental <- poly2nb(hex_grid_emmental)
#   hex_grid_emmental <- hex_grid_emmental[-sample.int(
#     length(hex_grid_emmental),
#     1,
#     prob = sapply(
#       seq_along(hex_grid_emmental),
#       function(x) {
#         if (neighbours_list_emmental[[x]][1] == 0) {
#           0
#         } else {
#           exp(- 4 * mean(card(neighbours_list_emmental)[neighbours_list_emmental[[x]]]))
#         }
#       }
#     )
#   ), ]
# }

# # Creating neighbours' list
# neighbours_list_emmental <- poly2nb(hex_grid_emmental)

# # Comparing distributions
# plot(hex_grid_emmental, col = "grey")
# rbind(
#   freq_compare,
#   as.data.frame(prop.table(table(factor(
#     card(neighbours_list_emmental),
#     levels = 0:6
#   )))) |>
#     mutate(from = "toy example emmental") |>
#     rename(number_of_neighbours = "Var1")
# ) |>
#   ggplot(aes(x = number_of_neighbours, y = Freq, fill = from)) +
#   geom_col(position = "dodge")
# The distribution is closer to the one of the human tissue data set

# # Creating dataset
# hex_sf_emmental <- st_sf(
#   id = seq_along(hex_grid_emmental),
#   geometry = hex_grid_emmental
# )
# #Adding centroids
# hex_sf_emmental <- cbind(
#   hex_sf_emmental,
#   st_coordinates(st_centroid(hex_sf_emmental))
# )

# Creating functions for cluster attribution of spots
spot_clust_attrib <- function(x, y, type) {
  f1 <- function(x, y) {
    case_when(
      (x - y / 5)^2 + 2 * (x / 5 + y - 1 / 2)^2 <= 1 / 4 ~ 1,
      (x + y / 2 - 3 / 2)^2 + 2 * (x / 2 - y + 1 / 3)^2 <= 3 / 4 ~ 2,
      TRUE ~ 3
    )
  }
  f2 <- function(x, y) {
    z <- (x - y + 1)^2 + 5 * (x + y - 1)^2
    case_when(
      z <= 0.96^2 ~ 1,
      z <= 1^2 ~ 2,
      z <= 1.5^2 ~ 3,
      z <= 1.54^2 ~ 2,
      TRUE ~ 1
    )
  }
  case_when(
    type == 1 ~ f1(x, y),
    type == 2 ~ f2(x, y),
    TRUE ~ f1(x, y)
  )
}

# Computing spot cluster attribution
hex_sf$cluster_type1 <- spot_clust_attrib(
  hex_sf$X,
  hex_sf$Y,
  1
)
hex_sf$cluster_type2 <- spot_clust_attrib(
  hex_sf$X,
  hex_sf$Y,
  2
)
# hex_sf_emmental$cluster_type1 <- spot_clust_attrib(
#   hex_sf_emmental$X,
#   hex_sf_emmental$Y,
#   1
# )
# hex_sf_emmental$cluster_type2 <- spot_clust_attrib(
#   hex_sf_emmental$X,
#   hex_sf_emmental$Y,
#   2
# )

# Visualisation with the simple grid
par(mfrow = c(1, 2))
plot(
  hex_grid,
  col = hex_sf$cluster_type1,
  border = "grey",
  main = "Type one spot clusters"
)
plot(
  hex_grid,
  col = hex_sf$cluster_type2,
  border = "grey",
  main = "Type two spot clusters"
)
par(mfrow = c(1, 1))

# Gene clusters for the 30 genes
# Type 1 attribution (evenly distributed):
# - k = 1: {1,...,10}
# - k = 2: {11,...,20}
# - k = 3: {21,...,30}
# Type 2 attribution (3rd cluster is smaller):
# - k = 1: {1,...,12}
# - k = 2: {13,...,24}
# - k = 3: {25,...,30}
K <- 3
R <- 3
gene_clust_type_1 <- c(rep(1, 10), rep(2, 10), rep(3, 10))
gene_clust_type_2 <- c(rep(1, 12), rep(2, 12), rep(3, 6))

# Creating function for sampling the normal inverse gamma prior
rnormgamma <- function(n, mu0, lambda, alpha, beta) {
  if (length(n) > 1)
    n <- length(n)
  sigma_sq <- 1 / rgamma(n, alpha, beta)
  mu <- rnorm(n, mu0, sqrt(sigma_sq / lambda))
  tibble(mu = mu, sigma_sq = sigma_sq) |>
  mutate(
    k = (row_number() + R - 1) %/% R,
    r = (row_number() - 1) %% R + 1
  )
}

# 2 different distributions
# nominal one: normal distribution
# heavy tailed one: Cauchy distribution
plot_dist <- function(t, qfun, dfun, alpha = 0.05, res = 1000) {
  q <- qfun(1 - alpha / 2)
  t <- t |> mutate(
    mini = mu - q * sqrt(sigma_sq),
    maxi = mu + q * sqrt(sigma_sq)
  )
  lower_bound <- min(t$mini)
  upper_bound <- max(t$maxi)
  x <- seq(lower_bound, upper_bound, length.out = res)
  t |>
    mutate(curve = paste(
      "k =", k, "and r =", r,
      "\nmu =",
      round(mu, 3),
      "and sigma_sq =",
      round(sigma_sq, 3)
    )) |>
    crossing(x = x) |>
    mutate(y = dfun(x, mu, sqrt(sigma_sq))) |>
    ggplot(aes(x, y, color = curve)) +
    geom_line()
}

plot_dist(rnormgamma(9, 0, 1, 3, 1), qnorm, dnorm)

plot_dist(rnormgamma(9, 0, 1, 3, 1), qcauchy, dcauchy, alpha = 0.5)

# Creating toy exammples with all possible combinations (namely 8)
toy_examples <- tibble(
  dist = c("normal", "cauchy"),
  spot_clust = c(1, 2),
  gene_clust = c(1, 2)
) |>
  expand(dist, spot_clust, gene_clust) |>
  rowwise() |>
  mutate(parameters = list(rnormgamma(9, 0, 1, 3, 1)))

create_mean_array <- function(parameters) {
  parameters |>
    select(-sigma_sq) |>
    pivot_wider(names_from = r, values_from = mu) |>
    select(-k) |>
    as.matrix()
}

create_sd_array <- function(parameters) {
  parameters |>
    select(-mu) |>
    pivot_wider(names_from = r, values_from = sigma_sq) |>
    select(-k) |>
    mutate_all(sqrt) |>
    as.matrix()
}

sample_matrix <- function(dist, spot_clust, gene_clust, parameters) {
  if (gene_clust == 1) {
    gene_clustering <- gene_clust_type_1
  } else {
    gene_clustering <- gene_clust_type_2
  }
  if (dist == "normal") {
    dist_fun <- rnorm
  } else {
    dist_fun <- rcauchy
  }
  if (spot_clust == 1) {
    spot_clustering <- hex_sf$cluster_type1
  } else {
    spot_clustering <- hex_sf$cluster_type2
  }
  mean_array <- create_mean_array(parameters)
  sd_array <- create_sd_array(parameters)
  n_spots <- length(spot_clustering)
  n_genes <- length(gene_clustering)
  sample_mat <- matrix(0, nrow = n_spots, ncol = n_genes)
  for (i in 1:K){
    indices_spots <- which(spot_clustering == i)
    for (j in 1:R){
      indices_genes <- which(gene_clustering == j)
      sample_mat[indices_spots, indices_genes] <- dist_fun(length(indices_spots) * length(indices_genes), mean_array[i, j], sd_array[i, j])
    }
  }
  return(sample_mat)
}

toy_examples <- toy_examples |>
  rowwise() |>
  mutate(
    samples = list(sample_matrix(dist, spot_clust, gene_clust, parameters))
  )

# Visualisation of the toy examples
tei <- 7 # toy example index to visualise
gi <- 22 # gene index to visualise
ggplot() +
  geom_sf(data = hex_grid, aes(fill = toy_examples[tei,]$samples[[1]][,gi])) +
  scale_fill_viridis_c() +
  ggtitle(paste("Toy example", tei, "with", toy_examples[tei,]$dist, "sample source, spots of type", toy_examples[tei,]$spot_clust, "and\n genes of type", toy_examples[tei,]$gene_clust, "for gene", gi)) +
  labs(fill=paste("Gene", gi)) +
  theme_minimal()

plot_dist(toy_examples[tei,]$parameters[[1]], qfun = qnorm, dfun = dnorm)

# Saving the toy examples

# Saving adjecency list of list as matrix with 6 rows where we fill the empty cells with 0
target_length <- 6
padded_list <- lapply(neighbours_list, function(vec) {
  if (length(vec) < target_length) {
    c(vec, rep(0, target_length - length(vec)))
  } else {
    vec
  }
})
adjacency_matrix <- do.call("rbind",padded_list)
write.table(adjacency_matrix, "toy_examples/adjacency.csv", row.names = FALSE, col.names = FALSE, sep = ",")

# Saving bin centers
# Create header
write.csv(as.matrix(as.data.frame(hex_sf) |> select(id, X, Y)), "toy_examples/bin_centers.csv", row.names = FALSE)

# Saving toy examples
register_toy <- function(toy_example) {
  directory <- paste0("toy_examples/toy_example_", toy_example$dist, "_spot_clust_", toy_example$spot_clust, "_gene_clust_", toy_example$gene_clust)
  dir.create(directory, showWarnings = FALSE)
  write.csv(toy_example$samples[[1]], paste0(directory, "/samples.csv"))
  if (toy_example$gene_clust == 1) {
    write.csv(gene_clust_type_1, paste0(directory, "/gene_clustering.csv"))
  } else {
    write.csv(gene_clust_type_2, paste0(directory, "/gene_clustering.csv"))
  }
  if (toy_example$spot_clust == 1) {
    write.csv(hex_sf$cluster_type1, paste0(directory, "/spot_clustering.csv"))
  } else {
    write.csv(hex_sf$cluster_type2, paste0(directory, "/spot_clustering.csv"))
  }
  # toy_example is a 1-row tibble: list columns must be unwrapped with [[1]]
  write.csv(create_mean_array(toy_example$parameters[[1]]), paste0(directory, "/mean_array.csv"))
  write.csv(create_sd_array(toy_example$parameters[[1]]), paste0(directory, "/sd_array.csv"))
}

dir.create("toy_examples", showWarnings = FALSE)
for (i in seq_len(nrow(toy_examples))) {
  register_toy(toy_examples[i, ])}