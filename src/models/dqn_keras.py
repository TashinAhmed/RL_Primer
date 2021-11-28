import tensorflow as tf


# Neural Network model for Deep Q Learning
def OurModel(input_shape, action_space):
    X_input = tf.keras.layers.Input(input_shape)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = tf.keras.layers.Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    # Hidden layer with 256 nodes
    X = tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    # Hidden layer with 64 nodes
    X = tf.keras.layers.Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    # Output Layer with # of actions: 2 nodes (left, right)
    X = tf.keras.layers.Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = tf.keras.Model(inputs = X_input, outputs = X, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    print(model.summary())
    return model

OurModel([100], 2)