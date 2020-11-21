library(tensorflow)
library(tfautograph)
library(keras)
library(tfdatasets)

Linear <- function() {
    keras_model_custom(model_fn = function(self) {
        self$w <- tf$Variable(5, name = "weight")
        self$b <- tf$Variable(10, name = "bias")
        function(inputs, mask = NULL, training = TRUE) {
            inputs*self$w + self$b
        }
    }) 
}

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES <- 2000
training_inputs <- tf$random$normal(shape = shape(NUM_EXAMPLES))
noise <- tf$random$normal(shape = shape(NUM_EXAMPLES))
training_outputs <- training_inputs * 3 + 2 + noise

# The loss function to be optimized
loss <- function(model, inputs, targets) {
    error <- model(inputs) - targets
    tf$reduce_mean(tf$square(error))
}

grad <- function(model, inputs, targets) {
    with(tf$GradientTape() %as% tape, {
        loss_value <- loss(model, inputs, targets)
    })
    tape$gradient(loss_value, list(model$w, model$b))
}

model <- Linear()
optimizer <- optimizer_sgd(lr = 0.01)

cat("Initial loss: ", as.numeric(loss(model, training_inputs, training_outputs), "\n"))


for (i in seq_len(300)) {
    grads <- grad(model, training_inputs, training_outputs)
    optimizer$apply_gradients(purrr::transpose(
        list(grads, list(model$w, model$b))
    ))
    if (i %% 20 == 0)
        cat("Loss at step ", i, ": ", as.numeric(loss(model, training_inputs, training_outputs)), "\n")
}

model$w

model$b
