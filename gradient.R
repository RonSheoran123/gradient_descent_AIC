# Install necessary packages if not already installed
if (!require(quantmod)) install.packages("quantmod")
if (!require(copula)) install.packages("copula")
if (!require(PerformanceAnalytics)) install.packages("PerformanceAnalytics")

# Load libraries
library(quantmod)
library(copula)
library(PerformanceAnalytics)

start_time <- proc.time()


# Fetch stock data for Google and Microsoft
getSymbols(c("GOOGL", "MSFT"), src = "yahoo", from = "2015-01-01", to = Sys.Date())

# Combine the closing prices into a data frame
data <- merge(Cl(GOOGL), Cl(MSFT))
colnames(data) <- c("GOOGL", "MSFT")

# Calculate log returns
log_returns <- diff(log(data))[-1, ]  # Remove NA values from the diff
log_returns <- as.data.frame(log_returns)  # Convert to data frame for compatibility

# Transform returns to uniform margins
u <- pobs(log_returns)

# Function to calculate the log-likelihood of the mixed copula model
calc_log_likelihood <- function(u, weights, params) {
  # Densities for each copula
  densities <- list(
    gaussian = dCopula(u, normalCopula(param = params$gaussian)),
    gumbel = dCopula(u, gumbelCopula(param = params$gumbel)),
    clayton = dCopula(u, claytonCopula(param = params$clayton))
  )
  
  # Mixed density based on weights
  mixed_density <- weights[1] * densities$gaussian + 
    weights[2] * densities$gumbel + 
    weights[3] * densities$clayton
  
  # Log-likelihood
  log_likelihood <- sum(log(mixed_density))
  return(log_likelihood)
}

# Function to calculate AIC
calc_aic <- function(u, weights, params) {
  log_likelihood <- calc_log_likelihood(u, weights, params)
  num_params <- 3  # Three weights to estimate
  aic <- -2 * log_likelihood + 2 * num_params
  return(aic)
}

# Function to compute gradients for AIC
compute_gradients <- function(u, weights, params) {
  grad <- numeric(length(weights))
  
  # Small perturbation for numerical gradient estimation
  epsilon <- 1e-6
  
  for (i in 1:length(weights)) {
    perturbed_weights <- weights
    perturbed_weights[i] <- perturbed_weights[i] + epsilon
    aic_plus <- calc_aic(u, perturbed_weights, params)
    
    perturbed_weights[i] <- perturbed_weights[i] - 2 * epsilon
    aic_minus <- calc_aic(u, perturbed_weights, params)
    
    # Numerical gradient
    grad[i] <- (aic_plus - aic_minus) / (2 * epsilon)
  }
  
  return(grad)
}

# Gradient Descent Function to Optimize Weights
gradient_descent_aic <- function(u, initial_weights, initial_params, alpha = 0.01, tol = 1e-6, max_iter = 100) {
  weights <- initial_weights
  
  for (iter in 1:max_iter) {
    # Calculate AIC and gradients
    aic <- calc_aic(u, weights, initial_params)
    gradients <- compute_gradients(u, weights, initial_params)
    
    # Update weights
    weights <- weights - alpha * gradients
    
    # Ensure weights sum to 1 (simple normalization)
    weights <- weights / sum(weights)
    
    # Check for convergence
    if (max(abs(gradients)) < tol) {
      cat("Convergence reached at iteration:", iter, "\n")
      break
    }
  }
  
  return(weights)
}

# Initialize parameters for the copulas (fit initially)
initial_params <- list(
  gaussian = fitCopula(normalCopula(), u, method = "ml")@estimate,
  gumbel = fitCopula(gumbelCopula(), u, method = "ml")@estimate,
  clayton = fitCopula(claytonCopula(), u, method = "ml")@estimate
)

# Initialize weights
initial_weights <- c(0.01, 0.01, 0.98)

# Run gradient descent to optimize weights
optimized_weights <- gradient_descent_aic(u, initial_weights, initial_params)

# Calculate final log-likelihood and AIC with optimized weights
final_log_likelihood <- calc_log_likelihood(u, optimized_weights, initial_params)
final_aic <- calc_aic(u, optimized_weights, initial_params)

end_time <- proc.time()

time_taken <- end_time - start_time


# Print final optimized weights, log-likelihood, and AIC
cat("Optimized Weights:\n")
print(optimized_weights)
cat("\nFinal Log-Likelihood:\n")
print(final_log_likelihood)
cat("\nFinal AIC:\n")
print(final_aic)

print("Time taken:")
print(time_taken)
