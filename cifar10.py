import time
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as fnn
from flax.training.train_state import TrainState
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocessing(x, y):
    x = tf.cast(x, tf.float32) / 255.    
    return x, y

ds = tfds.load("cifar10", as_supervised=True, shuffle_files=False, download=True)
train_set = ds["train"]
train_set = train_set.shuffle(len(train_set), seed=0, reshuffle_each_iteration=True).batch(32).map(preprocessing).prefetch(1)
val_set = ds["test"]
val_set = val_set.batch(32).map(preprocessing).prefetch(1)

# model
class CNN(fnn.Module):

    @fnn.compact
    def __call__(self, x, is_training):
        x = fnn.Conv(features=32, kernel_size=(3, 3))(x)
        x = fnn.BatchNorm(use_running_average=not is_training, momentum=0.1)(x)
        x = fnn.relu(x)
        x = fnn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = fnn.Conv(features=64, kernel_size=(3, 3))(x)
        x = fnn.BatchNorm(use_running_average=not is_training, momentum=0.1)(x)
        x = fnn.relu(x)
        x = fnn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = fnn.Dense(features=256)(x)
        x = fnn.relu(x)
        x = fnn.Dense(features=10)(x)
        x = fnn.log_softmax(x)
        
        return x

model = CNN()
variables = model.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 3]), True)
params = variables["params"]
batch_stats = variables["batch_stats"]
tx = optax.adam(0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=(4,))
def step(x, y, state, batch_stats, is_training=True):
    def loss_fn(params, batch_stats):
        y_pred, mutated_vars = state.apply_fn({"params": params, "batch_stats": batch_stats}, x, is_training, mutable=["batch_stats"]) 
        new_batch_stats = mutated_vars["batch_stats"]
        loss = optax.softmax_cross_entropy(logits=y_pred, labels=y).mean()
        return loss, (y_pred, new_batch_stats)
    y = jnp.eye(10)[y]
    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (y_pred, new_batch_stats)), grads = grad_fn(state.params, batch_stats)
        state = state.apply_gradients(grads=grads)
    else:
        loss, (y_pred, new_batch_stats) = loss_fn(state.params, batch_stats)
    return loss, y_pred, state, new_batch_stats

for e in range(5):
    tic = time.time()
    train_loss, val_loss, acc = 0., 0., 0.
    for x, y in train_set.as_numpy_iterator(): 
        loss, y_pred, state, batch_stats = step(x, y, state, batch_stats, is_training=True)
        train_loss += loss
    train_loss /= len(train_set)

    for x, y in val_set.as_numpy_iterator(): 
        loss, y_pred, state, batch_stats = step(x, y, state, batch_stats, is_training=False)
        val_loss += loss
        acc += (jnp.argmax(y_pred, 1) == y).mean()
    val_loss /= len(val_set)
    acc /= len(val_set)
    elapsed = time.time() - tic
    
    print(f"train_loss: {train_loss:0.2f}, val_loss: {val_loss:0.2f}, val_acc: {acc:0.2f}, elapsed: {elapsed:0.2f}")
