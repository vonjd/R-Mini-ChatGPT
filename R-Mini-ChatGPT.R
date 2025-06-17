######################################################################################################
## See accompanying blog post:                                                                      ##
## https://blog.ephorie.de/building-your-own-mini-chatgpt-with-r-from-markov-chains-to-transformers ##
######################################################################################################

# install torch package with GPU support: 
# https://torch.mlverse.org/docs/articles/installation
library(torch)

# Word-level tokenizer
create_tokenizer <- function(text) {
  # Convert to lowercase and split into words
  text <- tolower(text)
  words <- unlist(strsplit(text, "\\s+"))
  words <- words[words != ""]  # Remove empty strings
  
  # Get unique words and sort for consistency
  unique_words <- unique(words)
  unique_words <- sort(unique_words)
  
  # Add special tokens
  vocab <- c("<start>", "<end>", unique_words)
  
  word_to_idx <- setNames(seq_along(vocab), vocab)
  idx_to_word <- setNames(vocab, seq_along(vocab))
  
  list(
    word_to_idx = word_to_idx,
    idx_to_word = idx_to_word,
    vocab_size = length(vocab)
  )
}

# Encode text to indices
encode_text <- function(text, tokenizer) {
  text <- tolower(text)
  words <- unlist(strsplit(text, "\\s+"))
  words <- words[words != ""]
  
  # Add start token and map words to indices
  words <- c("<start>", words, "<end>")
  indices <- tokenizer$word_to_idx[words]
  
  # Handle unknown words (shouldn't happen with our setup, but just in case)
  indices[is.na(indices)] <- tokenizer$word_to_idx["<end>"]
  
  as.integer(indices)
}

# Decode indices to text
decode_text <- function(indices, tokenizer) {
  words <- tokenizer$idx_to_word[as.character(indices)]
  # Remove start/end tokens and paste with spaces
  words <- words[!words %in% c("<start>", "<end>")]
  paste(words, collapse = " ")
}

# Create positional encoding
create_positional_encoding <- function(max_len, d_model, device = NULL) {
  # Auto-detect device if not provided
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  
  pe <- torch_zeros(max_len, d_model, device = device)
  position <- torch_arange(0, max_len - 1, device = device)$unsqueeze(2)
  
  div_term <- torch_exp(torch_arange(0, d_model - 1, 2, device = device) * 
                          -(log(10000.0) / d_model))
  
  pe[, seq(1, d_model, 2)] <- torch_sin(position * div_term)
  if (d_model %% 2 == 0) {
    pe[, seq(2, d_model, 2)] <- torch_cos(position * div_term)
  } else {
    pe[, seq(2, d_model - 1, 2)] <- torch_cos(position * div_term[1:(length(div_term)-1)])
  }
  
  pe
}

# Transformer decoder layer
transformer_layer <- nn_module(
  initialize = function(d_model, n_heads, d_ff = NULL) {
    if (is.null(d_ff)) d_ff <- d_model * 4
    
    self$d_model <- d_model
    self$n_heads <- n_heads
    self$d_k <- d_model %/% n_heads
    
    # Multi-head attention components
    self$w_q <- nn_linear(d_model, d_model, bias = FALSE)
    self$w_k <- nn_linear(d_model, d_model, bias = FALSE)
    self$w_v <- nn_linear(d_model, d_model, bias = FALSE)
    self$w_o <- nn_linear(d_model, d_model)
    
    # Feed-forward network
    self$ff <- nn_sequential(
      nn_linear(d_model, d_ff),
      nn_relu(),
      nn_linear(d_ff, d_model)
    )
    
    # Layer normalization
    self$ln1 <- nn_layer_norm(d_model)
    self$ln2 <- nn_layer_norm(d_model)
    
    # Dropout
    self$dropout <- nn_dropout(0.1)
  },
  
  forward = function(x, mask = NULL) {
    batch_size <- x$size(1)
    seq_len <- x$size(2)
    
    # Multi-head self-attention
    q <- self$w_q(x)$view(c(batch_size, seq_len, self$n_heads, self$d_k))$transpose(2, 3)
    k <- self$w_k(x)$view(c(batch_size, seq_len, self$n_heads, self$d_k))$transpose(2, 3)
    v <- self$w_v(x)$view(c(batch_size, seq_len, self$n_heads, self$d_k))$transpose(2, 3)
    
    # Scaled dot-product attention
    scores <- torch_matmul(q, k$transpose(-2, -1)) / sqrt(self$d_k)
    
    if (!is.null(mask)) {
      scores <- scores + mask$unsqueeze(1)$unsqueeze(1)
    }
    
    attn_weights <- nnf_softmax(scores, dim = -1)
    attn_weights <- self$dropout(attn_weights)
    
    attn_output <- torch_matmul(attn_weights, v)
    attn_output <- attn_output$transpose(2, 3)$contiguous()$view(c(batch_size, seq_len, self$d_model))
    attn_output <- self$w_o(attn_output)
    
    # First residual connection and layer norm
    x <- self$ln1(x + self$dropout(attn_output))
    
    # Feed-forward network
    ff_output <- self$ff(x)
    
    # Second residual connection and layer norm
    x <- self$ln2(x + self$dropout(ff_output))
    
    x
  }
)

# Main LLM model
toy_llm <- nn_module(
  initialize = function(vocab_size, d_model = 128, n_heads = 4, n_layers = 3, max_len = 256, device = NULL) {
    # Auto-detect device if not provided
    if (is.null(device)) {
      device <- if (cuda_is_available()) "cuda" else "cpu"
    }
    
    self$d_model <- d_model
    self$max_len <- max_len
    self$device <- device
    
    # Embeddings
    self$token_embedding <- nn_embedding(vocab_size, d_model)
    self$pos_encoding <- create_positional_encoding(max_len, d_model, "cpu")  # Always create on CPU initially
    
    # Transformer layers (using individual modules for simplicity)
    self$transformer_layer_1 <- transformer_layer(d_model, n_heads)
    if (n_layers >= 2) self$transformer_layer_2 <- transformer_layer(d_model, n_heads)
    if (n_layers >= 3) self$transformer_layer_3 <- transformer_layer(d_model, n_heads)
    self$n_layers <- n_layers
    
    # Output
    self$ln_f <- nn_layer_norm(d_model)
    self$lm_head <- nn_linear(d_model, vocab_size)
    
    self$dropout <- nn_dropout(0.1)
  },
  
  forward = function(x) {
    seq_len <- x$size(2)
    
    # Create causal mask on the correct device
    mask <- torch_triu(torch_ones(seq_len, seq_len, device = x$device), diagonal = 1)
    mask <- mask$masked_fill(mask == 1, -Inf)
    
    # Token embeddings + positional encoding
    x <- self$token_embedding(x) * sqrt(self$d_model)
    pos_enc <- self$pos_encoding[1:seq_len, ]$to(device = x$device)
    x <- x + pos_enc$unsqueeze(1)
    x <- self$dropout(x)
    
    # Pass through transformer layers
    x <- self$transformer_layer_1(x, mask)
    if (self$n_layers >= 2) x <- self$transformer_layer_2(x, mask)
    if (self$n_layers >= 3) x <- self$transformer_layer_3(x, mask)
    
    # Final layer norm and projection
    x <- self$ln_f(x)
    logits <- self$lm_head(x)
    
    logits
  }
)

# Training function
train_model <- function(model, data, tokenizer, epochs = 500, seq_len = 24, batch_size = 4, lr = 0.001, device = NULL) {
  # Auto-detect device if not provided
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  optimizer <- optim_adam(model$parameters, lr = lr)
  
  # Prepare training data
  indices <- encode_text(data, tokenizer)
  
  cat(sprintf("Training on %d tokens...\n", length(indices)))
  
  for (epoch in 1:epochs) {
    total_loss <- 0
    n_batches <- 0
    
    # Create random batches
    max_start <- length(indices) - seq_len - 1
    if (max_start < 1) {
      cat("Error: Text too short for sequence length\n")
      return()
    }
    
    # Calculate number of possible batches
    n_possible_batches <- max_start %/% (seq_len %/% 2)  # Allow overlap
    n_batches_per_epoch <- min(n_possible_batches, 20)  # Limit batches per epoch
    
    for (batch_idx in 1:n_batches_per_epoch) {
      # Get batch
      batch_data <- list()
      batch_targets <- list()
      
      for (b in 1:batch_size) {
        start_idx <- sample(1:max_start, 1)
        
        batch_data[[b]] <- indices[start_idx:(start_idx + seq_len - 1)]
        batch_targets[[b]] <- indices[(start_idx + 1):(start_idx + seq_len)]
      }
      
      # Convert to tensors and move to device
      input_tensor <- torch_tensor(do.call(rbind, batch_data), dtype = torch_long())$to(device = device)
      target_tensor <- torch_tensor(do.call(rbind, batch_targets), dtype = torch_long())$to(device = device)
      
      # Forward pass
      optimizer$zero_grad()
      logits <- model(input_tensor)
      
      # Calculate loss
      loss <- nnf_cross_entropy(logits$view(c(-1, tokenizer$vocab_size)), 
                                target_tensor$view(-1))
      
      # Backward pass
      loss$backward()
      optimizer$step()
      
      total_loss <- total_loss + loss$item()
      n_batches <- n_batches + 1
    }
    
    if (epoch %% 50 == 0) {
      avg_loss <- total_loss / n_batches
      cat(sprintf("Epoch %d/%d, Loss: %.4f\n", epoch, epochs, avg_loss))
    }
  }
  
  cat("Training completed!\n")
}

# Text generation function
generate_text <- function(model, tokenizer, prompt = "", max_new_tokens = 20, temperature = 1.0, device = NULL) {
  # Auto-detect device if not provided
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  
  model$eval()
  
  if (prompt == "") {
    # Start with start token
    current_words <- c("<start>")
  } else {
    prompt <- tolower(prompt)
    prompt_words <- unlist(strsplit(prompt, "\\s+"))
    prompt_words <- prompt_words[prompt_words != ""]
    current_words <- c("<start>", prompt_words)
  }
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      # Encode current words
      indices <- tokenizer$word_to_idx[current_words]
      
      # Handle any unknown words
      indices[is.na(indices)] <- tokenizer$word_to_idx["<end>"]
      
      # Limit context window
      if (length(indices) > 32) {
        indices <- tail(indices, 32)
      }
      
      # Convert to tensor and move to device
      input_tensor <- torch_tensor(matrix(indices, nrow = 1), dtype = torch_long())$to(device = device)
      
      # Get predictions
      logits <- model(input_tensor)
      
      # Get logits for the last position
      last_logits <- logits[1, -1, ] / temperature
      
      # Sample next token
      probs <- nnf_softmax(last_logits, dim = 1)
      next_token <- torch_multinomial(probs, 1)$item()
      
      # Decode and append
      next_word <- tokenizer$idx_to_word[as.character(next_token)]
      
      # Stop if we hit end token
      if (next_word == "<end>") break
      
      current_words <- c(current_words, next_word)
    }
  })
  
  # Remove start token and join with spaces
  output_words <- current_words[current_words != "<start>"]
  paste(output_words, collapse = " ")
}

# Check for GPU availability
if (cuda_is_available()) {
  device <- "cuda"
  cat("Using GPU (CUDA)\n")
} else {
  device <- "cpu"
  cat("Using CPU\n")
}

# Example usage with much larger training corpus
cat("Loading training text from URL...\n")
txt <- readLines(url("http://paulo-jorente.de/text/alice_oz.txt"), warn = FALSE)
training_text <- paste(txt, collapse = " ")

cat(sprintf("Raw text length: %d characters\n", nchar(training_text)))
cat(sprintf("Raw first 100 chars: %s...\n", substr(training_text, 1, 100)))

# Clean up the text more carefully - preserve spaces!
training_text <- gsub("[^a-zA-Z0-9 .,!?;:-]", "", training_text)  # Explicitly include space character
training_text <- gsub("\\s+", " ", training_text)  # Normalize multiple spaces to single space
training_text <- gsub("^\\s+|\\s+$", "", training_text)  # Trim leading/trailing spaces
training_text <- tolower(training_text)

cat(sprintf("Cleaned text length: %d characters\n", nchar(training_text)))
cat(sprintf("First 200 characters: %s...\n\n", substr(training_text, 1, 200)))

# Create tokenizer
cat("Creating tokenizer...\n")
start_time <- Sys.time()
tokenizer <- create_tokenizer(training_text)
tokenizer_time <- Sys.time() - start_time

cat("Vocabulary size:", tokenizer$vocab_size, "\n")
cat("Tokenizer creation time:", round(as.numeric(tokenizer_time), 2), "seconds\n")
cat("Sample words:", paste(head(names(tokenizer$word_to_idx), 15), collapse = ", "), "\n\n")

# Create model (larger for the bigger vocabulary and more complex text)
cat("Creating model...\n")
model <- toy_llm(vocab_size = tokenizer$vocab_size, d_model = 256, n_heads = 8, n_layers = 4)
model <- model$to(device = device)  # Move model to GPU if available

# Count model parameters
total_params <- sum(sapply(model$parameters, function(p) p$numel()))
cat(sprintf("Model created with %s parameters\n", format(total_params, big.mark = ",")))

# Train model with timing
cat("\nStarting training...\n")
cat("Training for 500 epochs with larger corpus - this will take longer but produce much better results!\n")
cat("Target: Loss should drop below 2.5 for good quality text generation.\n\n")
training_start_time <- Sys.time()

# Use more epochs for the larger corpus to get better quality
train_model(model, training_text, tokenizer, epochs = 1000, seq_len = 32, batch_size = 4)

training_end_time <- Sys.time()
training_duration <- training_end_time - training_start_time
cat(sprintf("Training completed in %.2f minutes\n\n", as.numeric(training_duration, units = "mins")))

# Generate text with various prompts
cat("Generating text samples...\n")
generation_start_time <- Sys.time()

prompts <- c("alice", "the queen", "down the", "once upon", "the wizard")

for (i in 1:length(prompts)) {
  generated <- generate_text(model, tokenizer, prompt = prompts[i], max_new_tokens = 20, temperature = 0.8)
  cat(sprintf("Prompt '%s': %s\n", prompts[i], generated))
}

cat("\nMore creative examples (higher temperature):\n")
for (i in 1:3) {
  generated <- generate_text(model, tokenizer, prompt = "alice saw", max_new_tokens = 15, temperature = 1.2)
  cat(sprintf("Sample %d: %s\n", i, generated))
}

generation_time <- Sys.time() - generation_start_time
cat(sprintf("\nGeneration completed in %.2f seconds\n", as.numeric(generation_time)))

# Final summary
cat(paste0("\n", paste(rep("=", 50), collapse = ""), "\n"))
cat("TRAINING SUMMARY\n")
cat(paste0(paste(rep("=", 50), collapse = ""), "\n"))
cat(sprintf("Text length: %s characters\n", format(nchar(training_text), big.mark = ",")))
cat(sprintf("Vocabulary size: %s words\n", format(tokenizer$vocab_size, big.mark = ",")))
cat(sprintf("Model parameters: %s\n", format(total_params, big.mark = ",")))
cat(sprintf("Training time: %.2f minutes\n", as.numeric(training_duration, units = "mins")))
cat(sprintf("Device used: %s\n", device))
cat(paste0(paste(rep("=", 50), collapse = ""), "\n"))