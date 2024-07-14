import tensorflow as tf
import numpy as np

class Critic():
    """ 
    Critic model (current policy network)
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        state_input = tf.keras.Input(state_size)
        action_input = tf.keras.Input(action_size)
        self.state_hidden_layer_1 = tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.state_hidden_layer_2 = tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.action_hidden_layer_1 = tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.joint_hidden_layer_1 = tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.joint_hidden_layer_2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001))
        self.output_layer = tf.keras.layers.Dense(units=1)

        x1 = self.state_hidden_layer_1(state_input)
        x1 = self.state_hidden_layer_2(x1)
        x2 = self.action_hidden_layer_1(action_input)
        x = self.concat_layer([x1, x2])
        x = self.joint_hidden_layer_1(x)
        x = self.joint_hidden_layer_2(x)
        x = self.output_layer(x)
        self.model = tf.keras.Model(inputs=[state_input, action_input], outputs=x)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.model.compile(optimizer=self.optimizer)


    def __call__(self, state, action):
        return self.model([state, action], training=True)
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