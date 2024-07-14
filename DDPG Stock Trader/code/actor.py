import tensorflow as tf
import numpy as np

class Actor():
    """ 
    Actor model (current policy network)
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        input = tf.keras.Input(state_size)
        self.hidden_layer_1 = tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.hidden_layer_2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.output_layer = tf.keras.layers.Dense(units=action_size, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))

        x = self.hidden_layer_1(input)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=input, outputs=x)
        
        self.grads = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=self.optimizer)
    def __call__(self, state):
        return self.model(state, training=True)
    def get_weights(self):
        return self.model.weights
    def set_weights(self, weights):
        self.model.set_weights(weights)
    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    def save_model(self, location):
        self.model.save(location, save_format="keras")
    def load_model(self, location):
        self.model = tf.keras.models.load_model(location)
