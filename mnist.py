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

ds = tfds.load("mnist", as_supervised=True, shuffle_files=False, download=True)
train_set = ds["train"]
train_set = train_set.shuffle(len(train_set), seed=0, reshuffle_each_iteration=True).batch(32).map(preprocessing).prefetch(1)
val_set = ds["test"]
val_set = val_set.batch(32).map(preprocessing).prefetch(1)

# model
class MLP(fnn.Module):
    @fnn.compact
    def __call__(self, x):
        x = fnn.Dense(128)(x)
        x = fnn.relu(x)
        x = fnn.Dense(256)(x)
        x = fnn.relu(x)
        x = fnn.Dense(10)(x)
        x = fnn.log_softmax(x)
        
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 28 * 28]))['params']
tx = optax.adam(0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=(3,))
def step(x, y, state, is_train=True):
    def loss_fn(params):
        y_pred = state.apply_fn({'params': params}, x) 
        loss = optax.softmax_cross_entropy(logits=y_pred, labels=y).mean()
        return loss, y_pred
    x = x.reshape(-1, 28 * 28)
    y = jnp.eye(10)[y]
    if is_train:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, y_pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
    else:
        loss, y_pred = loss_fn(state.params)
    return loss, y_pred, state

for e in range(10):
    tic = time.time()
    train_loss, val_loss, acc = 0., 0., 0.
    for x, y in train_set.as_numpy_iterator(): 
        loss, y_pred, state = step(x, y, state, is_train=True)
        train_loss += loss
    train_loss /= len(train_set)

    for x, y in val_set.as_numpy_iterator(): 
        loss, y_pred, state = step(x, y, state, is_train=False)
        val_loss += loss
        acc += (jnp.argmax(y_pred, 1) == y).mean()
    val_loss /= len(val_set)
    acc /= len(val_set)
    elapsed = time.time() - tic
    
    print(f"train_loss: {train_loss:0.2f}, val_loss: {val_loss:0.2f}, val_acc: {acc:0.2f}, elapsed: {elapsed:0.2f}")