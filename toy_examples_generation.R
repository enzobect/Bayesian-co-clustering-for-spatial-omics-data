# File to generate the toy examples for the simulation study of the Pott's Model
# We create to hexagonal topologies, one simple square and one where we removed
# some hexagons to better mimic the distribution of the number of neighbours
# We also look at different cluster attributions for spots and we give different
# sizes of gene clusters. We always have K = 3, R = 3, 30 genes and
# approximately 3000 spots.

library(sf)
library(spdep)
library(tidyverse)
library(ggplot2)

# Creating the original "square" hexagonal grid
sfc <- st_sfc(st_polygon(list(rbind(c(0, 0), c(1, 0), c(1, 1), c(0, 0)))))
# Increasing `cellsize` decreases the number of cells
hex_grid <- st_make_grid(sfc, cellsize = .02, square = FALSE)
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
barplot(prop.table(table(factor(card(neighbours_list), levels = 0:6))))

# Comparing with the mouse brain dataset
nl_mouse_brain <- read.csv("../mouse_brain_data/adjecency.csv")
nb_mouse_brain <- nl_mouse_brain |>
  mutate_all(function(x) as.integer(x != 0)) |>
  rowSums()
freq_compare <- rbind(
  as.data.frame(prop.table(table(factor(
    nb_mouse_brain,
    levels = 0:6
  )))) |>
    mutate(from = "mouse brain"),
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
n_holes <- 100
hex_grid_papyrus <- hex_grid
for (i in 1:n_holes) {
  print(i)
  neighbours_list_papyrus <- poly2nb(hex_grid_papyrus)
  hex_grid_papyrus <- hex_grid_papyrus[-sample.int(
    length(hex_grid_papyrus),
    1,
    prob = sapply(
      seq_along(hex_grid_papyrus),
      function(x) {
        if (neighbours_list_papyrus[[x]][1] == 0) {
          0
        } else {
          exp(- 4 * mean(card(neighbours_list_papyrus)[neighbours_list_papyrus[[x]]]))
        }
      }
    )
  ), ]
}

# Creating neighbours' list
neighbours_list_papyrus <- poly2nb(hex_grid_papyrus)

# Comparing distributions
plot(hex_grid_papyrus, col = "grey")
rbind(
  freq_compare,
  as.data.frame(prop.table(table(factor(
    card(neighbours_list_papyrus),
    levels = 0:6
  )))) |>
    mutate(from = "toy example papyrus") |>
    rename(number_of_neighbours = "Var1")
) |>
  ggplot(aes(x = number_of_neighbours, y = Freq, fill = from)) +
  geom_col(position = "dodge")
# The distribution is closer to the one of the mouse data set

# Creating dataset
hex_sf_papyrus <- st_sf(
  id = seq_along(hex_grid_papyrus),
  geometry = hex_grid_papyrus
)
#Adding centroids
hex_sf_papyrus <- cbind(
  hex_sf_papyrus,
  st_coordinates(st_centroid(hex_sf_papyrus))
)

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
hex_sf_papyrus$cluster_type1 <- spot_clust_attrib(
  hex_sf_papyrus$X,
  hex_sf_papyrus$Y,
  1
)
hex_sf_papyrus$cluster_type2 <- spot_clust_attrib(
  hex_sf_papyrus$X,
  hex_sf_papyrus$Y,
  2
)

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

# Creating toy exammples with all possible combinations (namely 8)
toy_examples <- tibble(
  grid = c("simple", "papyrus"),
  spot_clust = c(1, 2),
  gene_clust = c(1, 2)
) |>
  expand(grid, spot_clust, gene_clust)

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

plot_norm_dist <- function(t, alpha = 0.05, res = 1000) {
  q <- qnorm(1 - alpha / 2)
  t <- t |> mutate(
    mini = mu - q * sigma_sq,
    maxi = mu + q * sigma_sq
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
    mutate(y = dnorm(x, mu, sigma_sq)) |>
    ggplot(aes(x, y, color = curve)) +
    geom_line()
}

plot_norm_dist(rnormgamma(9, 0, 1, 3, 1))
