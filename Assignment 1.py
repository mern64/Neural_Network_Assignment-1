# Imran Mansor 301844
# Assignment 1 :Perceptron Model to Predict Small Animal Will Approach an Object
# STINK 3014 - Neural Networks

import matplotlib.pyplot as plt

# Learning rate 
# I tried using higher learning rates like 0.8, but they caused the weights to not converge properly and making the wrong predictions
# I tried lower learning rates like 0.00001, but they made the training process very slow and still has errors after 10 epochs. If i need to use this i need to increase the epochs to more than 50 to make sure the model converges
# Finally,I choose 0.06 because after several trials, I found that this learning rate gives a good balance between convergence speed and stability
eta = 0.06

# I choose to fix it to 10 for because the model stop coverging around epoch 4, so 10 is sufficient to make sure the model converges.
epochs = 10 
# Initial weights (importance of each factor)
# x1 = Small is 0.1 because size is quite important for small animals to decide to approach an object if the object is small, then small animal will approach it
# x2 = Red is -0.2 because color is less important for small animals to decide to approach an object if the object is red, it could be a sign of danger (like a predator)
# x3 = Movement is 0.5 because movement is very important for small animals to decide to approach an object because if the movement is detected, it could be a predator
w = [0.1, -0.2, 0.5]
bias = 0.4 #Bias weight is total of the initial weights

# Display original weights
print("Original weights (initial guesses of importance):", [round(weight, 2) for weight in w])
print("Bias:", bias, "\n")

# Example dataset: [x1, x2, x3, y]
# x1 = Small
# x2 = Red
# x3 = Movement
# y: 1 = Approach, 0 = Not Approach
data = [
    [0, 1, 1, 0],  # Red & Movement = Not Approach
    [0, 1, 0, 0],  # Red only = Not Approach
    [1, 0, 0, 1],  # Small only = Approach
    [1, 1, 0, 1]   # Small & Red = Approach
]

# Activation function (step)
def activation(y):
    return 1 if y >= 0.5 else 0    #threshold at 0.5

# To record weights and errors
weight_history = []
error_history = []

# Training loop
for epoch in range(epochs):
    total_error = 0
    print(f"\n=== Epoch {epoch+1} ===")
    
    for idx, sample in enumerate(data):
        x1, x2, x3, actual = sample
        inputs = [x1, x2, x3]

        # Compute weighted sum
        y_in = sum(inputs[i] * w[i] for i in range(3)) + bias
        predicted = activation(y_in)
        
        # Compute error
        error = actual - predicted
        total_error += abs(error)
        print(f"Trip {idx+1}: Inputs={inputs}, Actual={actual}, Predicted={predicted}, Error={error}")
        
        # Update rule
        for i in range(3):
            w[i] += eta * error * inputs[i]
        bias += eta * error        
        print(f"Updated weights: {[round(weight, 2) for weight in w]}, Bias={round(bias, 2)}")
   
    # Record after each epoch
    weight_history.append(w.copy())
    error_history.append(total_error)

# --- Final results ---
print("\nFinal learned weights:", [round(weight, 2) for weight in w])
print("Final bias:", round(bias, 2))

# --- Visualization Section ---
# Convert weight history for plotting
w1_vals = [weights[0] for weights in weight_history]
w2_vals = [weights[1] for weights in weight_history]
w3_vals = [weights[2] for weights in weight_history]
epochs_range = range(1, epochs + 1)

# --- Plot weight changes ---
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, w1_vals, marker='o', label='Weight w1 (Small)')
plt.plot(epochs_range, w2_vals, marker='o', label='Weight w2 (Red)')
plt.plot(epochs_range, w3_vals, marker='o', label='Weight w3 (Movement)')
plt.title('Weight Evolution Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot error trend ---
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, error_history, marker='s', color='red')
plt.title('Total Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.grid(True)
plt.show()

# --- Prediction using trained model ---
print("\n--- Predicting whether the Small Animal will Approach or Not Approach an Object ---")
x1 = float(input("Enter Object Size : Small = 1 , Not Small =0 (0 to 1): "))
x2 = float(input("Enter Object Color : Red = 1 , Not Red =0 (0 to 1): "))
x3 = float(input("Enter Local Movement Detected : Move = 1 , Others =0 (0 to 1): "))

inputs = [x1, x2, x3]
y_in = sum(inputs[i] * w[i] for i in range(3)) + bias
predicted = activation(y_in)

print(f"\nWeighted sum (y_in): {y_in:.2f}")
if predicted == 1:
    print("Predicted Result: Small animal WILL NOT APPROACH the object!")
else:
    print("Predicted Result: Small animal WILL APPROACH the object!")
