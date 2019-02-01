import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        #print('N Features:' + str(features.shape[1]))
              
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch Shape = (n_records, n_features)
            
            SELF Variables
            ---------
            weights_input_to_hidden Shape = (InputNodes x HiddenNodes)
            weights_hidden_to_output Shape = (HiddenNodes x OutputNodes)
            activation_function (Sigmoid)

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer  #NOTE: Check if matmul necessary here
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer Note: Hidden Output Shape = (HiddenNodes x 1)
        final_outputs = final_inputs # signals from final output layer
        
        #print('Final Outputs:' + str(final_outputs))
        #print('Hidden Outputs:' + str(hidden_outputs))
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        #print('Error' + str(error))
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        #print('Hidden Error' + str(hidden_error))
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error #Note: f'(x) = 1 for output layer
        #print('Output Error Term' + str(output_error_term))
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs) #Note: f'(h) = sigmoid - (1 - sigmoid) at hidden units
        #print('Hidden Error Term' + str(hidden_error_term))
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        #print('Delta_W_I_H' + str(delta_weights_i_h))
        
        # Weight step (hidden to output)
        #print('Hidden Outputs' + str(hidden_outputs[:,None]))
        
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        #print('Delta_W_H_O' + str(delta_weights_h_o))
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here   .254
##########################################################
iterations = 5000
learning_rate = 0.575
hidden_nodes = 25
output_nodes = 1


######
# HyperParams Set 1: Training Loss: 0.1178 | Validation Loss: 0.345
#iterations = 10000
#learning_rate = 0.095
#hidden_nodes = 25
#output_nodes = 1
######

######
# HyperParams Set 2: Training Loss: 0.125 | Validation Loss: 0.240
#iterations = 10000
#learning_rate = 0.15
#hidden_nodes = 25
#output_nodes = 1
######

######
# HyperParams Set 3: Training Loss: 0.125 | Validation Loss: 0.240
#iterations = 4000
#learning_rate = 0.3
#hidden_nodes = 30
#output_nodes = 1
######

######
# HyperParams Set 4: Training Loss: 0.070 | Validation Loss: 0.155
#iterations = 4000
#learning_rate = 0.5
#hidden_nodes = 30
#output_nodes = 1
######

######
# HyperParams Set 5: Training Loss: 0.062 | Validation Loss: 0.147
#iterations = 4000
#learning_rate = 0.5
#hidden_nodes = 30
#output_nodes = 1
######

######
# HyperParams Set 6: Training Loss: 0.062 | Validation Loss: 0.134
#iterations = 5000
#learning_rate = 0.5
#hidden_nodes = 25
#output_nodes = 1
######