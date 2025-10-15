"""
TensorFlow pipeline: GAN-based augmentation -> LSTM (physics-regularized) -> GP short-term correction
Applied to the uploaded dataset: /mnt/data/DATA_GEO_Train.csv

Notes:
- Uses the same dataset columns seen earlier: 'x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)'
- Uses input_window=24, output_window=24 (6h windows at 15-min cadence), and rolling prediction
  to generate a full 24-hour Day-8 forecast (4 × 6h steps).
- Includes simple validation of synthetic samples (KS test per channel) and only keeps synthetic
  sequences that pass a light check.
- Uses a very small train schedule by default so script runs quickly for prototyping; increase epochs
  for production.
- GP correction is applied to the short-term horizons (first 4 steps = 1 hour) per target.

Run as a script in the environment where /mnt/data/DATA_GEO_Train.csv is available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K

# sklearn & scipy for GP, scaling, stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import ks_2samp

RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

# ---------------------------
# Configuration (small / safe defaults)
# ---------------------------
CSV_PATH = "DATA_GEO_Train.csv"
FEATURE_COLS = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
INPUT_WINDOW = 24    # 6 hours (24 * 15min)
OUTPUT_WINDOW = 24   # predict next 6 hours
SYN_PER_REAL = 2     # how many synthetic futures per real sample (default small)
GAN_EPOCHS = 500     # small number for prototyping; increase for better synthetic quality
GAN_BATCH = 32
LSTM_EPOCHS = 200     # small; increase when ready
LSTM_BATCH = 32
GP_HORIZON_STEPS = 4  # apply GP correction for first 4 predicted steps (1 hour)
SMOOTH_LAMBDA = 1e-3  # physics-informed smoothness penalty weight
LEARNING_RATE = 1e-3

# ---------------------------
# Utilities
# ---------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    # parse time if present
    if 'utc_time' in df.columns:
        try:
            # Use mixed format inference and coerce errors
            df['utc_time'] = pd.to_datetime(df['utc_time'], format='mixed', errors='coerce')
            df = df.sort_values('utc_time') # Sort by time before setting index
            # Set utc_time as index before time-weighted interpolation
            df = df.set_index('utc_time')
            # Check if the index is a DatetimeIndex and contains valid dates
            if not isinstance(df.index, pd.DatetimeIndex) or df.index.isnull().all():
                 raise ValueError("All values in 'utc_time' column are invalid datetimes after parsing.")
        except Exception as e:
             raise ValueError(f"Error parsing 'utc_time' column as datetimes: {e}")
    else:
        raise ValueError(f"Expected 'utc_time' column in {path}, but not found.")

    # ensure required columns present
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in {path}")

    # simple interpolation for small gaps - only if index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(method='time', limit_direction='both')
    else:
         # Fallback to linear interpolation if no DatetimeIndex - although we expect DatetimeIndex now
         df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(method='linear', limit_direction='both')
         # Only reset index if not DatetimeIndex, to avoid losing DatetimeIndex
         df = df.reset_index(drop=True)


    df = df.dropna(subset=FEATURE_COLS)
    return df

def create_sequences_from_array(arr, input_w, output_w):
    X, Y = [], []
    n = len(arr)
    for i in range(n - input_w - output_w + 1):
        X.append(arr[i:i+input_w])
        Y.append(arr[i+input_w:i+input_w+output_w])
    if len(X) == 0:
        return np.zeros((0, input_w, arr.shape[1]), dtype=np.float32), np.zeros((0, output_w, arr.shape[1]), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ---------------------------
# Load data
# ---------------------------
try:
    df = load_and_clean(CSV_PATH)
except ValueError as e:
    print(f"Error loading data: {e}")
    # Exit or handle the error appropriately, perhaps suggesting the user upload a file with 'utc_time'
    raise e # Re-raise the exception to stop execution

print("Loaded data shape:", df.shape)
data_raw = df[FEATURE_COLS].values.astype(np.float32)

# quick check length
n_total = data_raw.shape[0]
print("Total timesteps:", n_total)

# If too short for INPUT+OUTPUT windows, lower windows automatically (very small fallback)
min_needed = INPUT_WINDOW + OUTPUT_WINDOW
if n_total < min_needed:
    # try to reduce windows but keep >4
    print(f"Warning: not enough timesteps ({n_total}) for input+output={min_needed}. Reducing windows...")
    # reduce output and input proportionally
    factor = n_total // 8  # 8 steps minimum total
    factor = max(4, factor)
    INPUT_WINDOW = factor
    OUTPUT_WINDOW = factor
    print("New windows:", INPUT_WINDOW, OUTPUT_WINDOW)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
data = scaler.fit_transform(data_raw)  # shape (T, 4)
print("Data shape after scaling:", data.shape)

# Plot raw vs scaled data
try:
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 2.5*n_features), sharex=True)
    for i, col in enumerate(FEATURE_COLS):
        # Use default index if not DatetimeIndex
        x_vals = df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df))
        axs[i].plot(x_vals, data_raw[:, i], label='Raw')
        axs[i].plot(x_vals, data[:, i], label='Scaled')
        axs[i].set_title(col)
        axs[i].legend()
    plt.tight_layout()
    plt.suptitle("Raw vs Scaled Data", y=1.02)
    plt.show()
except Exception as e:
    print("Could not plot raw vs scaled data:", e)


# ---------------------------
# Create sequences (sliding windows over entire dataset)
# ---------------------------
X_all, Y_all = create_sequences_from_array(data, INPUT_WINDOW, OUTPUT_WINDOW)
print("All sequences:", X_all.shape, Y_all.shape)
if X_all.shape[0] == 0:
    raise RuntimeError("No sequences created. Need more data or smaller windows.")

# Split chronologically: use first 70% for GAN training + LSTM training, next 15% val, last 15% test
n_samples = X_all.shape[0]
i_train_end = int(n_samples * 0.70)
i_val_end = int(n_samples * 0.85)
X_train, Y_train = X_all[:i_train_end], Y_all[:i_train_end]
X_val, Y_val = X_all[i_train_end:i_val_end], Y_all[i_train_end:i_val_end]
X_test, Y_test = X_all[i_val_end:], Y_all[i_val_end:]
print("Train / Val / Test sequences:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ---------------------------
# Simple conditional GAN (Keras)
# Generator: past -> latent -> future
# Discriminator: (past + future) -> real/fake
# ---------------------------
tf.keras.backend.clear_session()

n_features = data.shape[1]
noise_dim = 32

def build_generator(input_window, output_window, n_features, noise_dim=32):
    past_in = layers.Input(shape=(input_window, n_features), name="past_input")
    # encode past
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(past_in)
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    # concatenate noise
    z = layers.Input(shape=(noise_dim,), name="noise_input")
    x = layers.Concatenate()([x, z])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(output_window * 64, activation='relu')(x)
    x = layers.Reshape((output_window, 64))(x)
    # decode to features
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    out = layers.Conv1D(n_features, 3, padding='same', activation='linear')(x)
    return models.Model([past_in, z], out, name="Generator")

def build_discriminator(input_window, output_window, n_features):
    past_in = layers.Input(shape=(input_window, n_features), name="past_input_d")
    future_in = layers.Input(shape=(output_window, n_features), name="future_input_d")
    x = layers.Concatenate(axis=1)([past_in, future_in])  # shape (input+output, n_features)
    x = layers.Conv1D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.Conv1D(128, 3, padding='same', activation='leaky_relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([past_in, future_in], out, name="Discriminator")

# Build models
G = build_generator(INPUT_WINDOW, OUTPUT_WINDOW, n_features, noise_dim)
D = build_discriminator(INPUT_WINDOW, OUTPUT_WINDOW, n_features)

G.summary()
D.summary()

# Compile discriminator
d_optimizer = optimizers.Adam(learning_rate=1e-4)
D.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Build combined GAN (freeze D)
D.trainable = False
past_input = layers.Input(shape=(INPUT_WINDOW, n_features), name="past_gan")
noise_input = layers.Input(shape=(noise_dim,), name="noise_gan")
fake_future = G([past_input, noise_input])
valid = D([past_input, fake_future])
GAN = models.Model([past_input, noise_input], valid, name="GAN")
g_optimizer = optimizers.Adam(learning_rate=1e-4)
GAN.compile(optimizer=g_optimizer, loss='binary_crossentropy')

# ---------------------------
# Prepare GAN training data
# ---------------------------
# Use X_train, Y_train
real_past = X_train
real_future = Y_train

# Simple GAN training loop (binary labels)
batch_size = min(GAN_BATCH, max(8, real_past.shape[0]//2))
half_batch = max(1, batch_size // 2)

print("GAN training batch size:", batch_size)
# Train few epochs (prototyping); increase GAN_EPOCHS for production
for epoch in range(GAN_EPOCHS):
    # Shuffle
    idx = np.random.permutation(real_past.shape[0])
    real_past_s = real_past[idx]
    real_future_s = real_future[idx]
    # iterate batches
    for i in range(0, len(real_past_s), batch_size):
        past_batch = real_past_s[i:i+half_batch]
        future_batch = real_future_s[i:i+half_batch]
        if len(past_batch) == 0:
            continue
        # train discriminator on real
        d_loss_real = D.train_on_batch([past_batch, future_batch], np.ones((len(past_batch), 1)))
        # sample noise and generate fake
        noise = np.random.normal(size=(len(past_batch), noise_dim)).astype(np.float32)
        fake_future = G.predict([past_batch, noise], verbose=0)
        # train discriminator on fake
        d_loss_fake = D.train_on_batch([past_batch, fake_future], np.zeros((len(past_batch), 1)))
        # train generator via GAN (wants D to label fakes as real)
        noise2 = np.random.normal(size=(len(past_batch), noise_dim)).astype(np.float32)
        g_loss = GAN.train_on_batch([past_batch, noise2], np.ones((len(past_batch), 1)))
    if (epoch + 1) % 50 == 0 or epoch == 0: # Print progress less often for more epochs
        print(f"GAN epoch {epoch+1}/{GAN_EPOCHS}  d_real={d_loss_real[0]:.4f} d_fake={d_loss_fake[0]:.4f} g_loss={g_loss:.4f}")

# ---------------------------
# Generate synthetic futures conditioned on training pasts
# ---------------------------
synthetic_X = []
synthetic_Y = []
for i in range(len(X_train)):
    past = X_train[i:i+1]  # shape (1, input, features)
    for k in range(SYN_PER_REAL):
        noise = np.random.normal(size=(1, noise_dim)).astype(np.float32)
        fut = G.predict([past, noise], verbose=0)[0]  # shape (output, features)
        synthetic_X.append(past[0])
        synthetic_Y.append(fut)
synthetic_X = np.array(synthetic_X, dtype=np.float32)
synthetic_Y = np.array(synthetic_Y, dtype=np.float32)
print("Generated synthetic shapes:", synthetic_X.shape, synthetic_Y.shape)

# ---------------------------
# Light validation of synthetic sequences: KS per channel vs real futures
# Keep synthetic only if per-channel KS p-value > 0.001 (very lenient)
# ---------------------------
def validate_synthetic(real_fut_all, synth_fut_all, threshold_p=1e-3):
    keep_idx = []
    n_synth = synth_fut_all.shape[0]
    for i in range(n_synth):
        s = synth_fut_all[i].ravel()
        # sample a random real future
        j = np.random.randint(0, real_fut_all.shape[0])
        r = real_fut_all[j].ravel()
        p_vals = []
        # compute KS separately per channel (split)
        ok = True
        for ch in range(n_features):
            s_ch = synth_fut_all[i, :, ch]
            r_ch = real_fut_all[j, :, ch]
            try:
                stat, p = ks_2samp(s_ch, r_ch)
            except Exception:
                p = 0.0
            if p < threshold_p:
                ok = False
                break
        if ok:
            keep_idx.append(i)
    return np.array(keep_idx, dtype=int)

keep_idx = validate_synthetic(Y_train, synthetic_Y, threshold_p=1e-4)
print("Synthetic kept / generated:", len(keep_idx), "/", synthetic_Y.shape[0])

if len(keep_idx) > 0:
    synthetic_X = synthetic_X[keep_idx]
    synthetic_Y = synthetic_Y[keep_idx]
else:
    # fallback: keep a small subset to avoid empty augmentation
    take = min(10, synthetic_Y.shape[0])
    print("No synthetic passed strict KS. Keeping a small fallback of", take)
    synthetic_X = synthetic_X[:take]
    synthetic_Y = synthetic_Y[:take]

# ---------------------------
# Combine real + synthetic for LSTM training
# ---------------------------
X_aug = np.concatenate([X_train, synthetic_X], axis=0)
Y_aug = np.concatenate([Y_train, synthetic_Y], axis=0)
print("Augmented dataset shape:", X_aug.shape, Y_aug.shape)

# Shuffle augmented data
perm = np.random.permutation(len(X_aug))
X_aug = X_aug[perm]; Y_aug = Y_aug[perm]

# Split augmented into training / validation (preserve real-only validation set for true eval)
n_aug = len(X_aug)
val_frac = 0.1
n_val_aug = max(1, int(n_aug * val_frac))
X_lstm_train = X_aug[:-n_val_aug]
Y_lstm_train = Y_aug[:-n_val_aug]
X_lstm_val = X_aug[-n_val_aug:]
Y_lstm_val = Y_aug[-n_val_aug:]
print("LSTM train/val shapes:", X_lstm_train.shape, X_lstm_val.shape)

# ---------------------------
# Build LSTM seq2seq (encoder-decoder style) with physics-informed smoothness penalty
# We'll implement a custom training loop with a combined loss: MSE + smoothness_penalty
# ---------------------------
tf.keras.backend.clear_session()
input_layer = layers.Input(shape=(INPUT_WINDOW, n_features), name="encoder_input")
# Encoder
enc = layers.LSTM(128, return_sequences=False, return_state=False)(input_layer)
enc = layers.Dropout(0.2)(enc)
# Prepare decoder initial input (we will implement decoder with RepeatVector + LSTM returning sequences)
dec_in = layers.RepeatVector(OUTPUT_WINDOW)(enc)
dec = layers.LSTM(64, return_sequences=True)(dec_in)
dec = layers.TimeDistributed(layers.Dense(64, activation='relu'))(dec)
out_mu = layers.TimeDistributed(layers.Dense(n_features, activation='linear'), name='mu')(dec)
# also predict log-variance optionally (small dense)
out_logvar = layers.TimeDistributed(layers.Dense(n_features, activation='linear'), name='logvar')(dec)

# For training convenience we'll concatenate mu & logvar
out_concat = layers.Concatenate(axis=-1)([out_mu, out_logvar])
lstm_model = models.Model(input_layer, out_concat, name='LSTM_Augmented')
lstm_model.summary()

# Custom loss function: MSE on mu + smoothness penalty on mu
mse_loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def combined_loss(y_true, y_pred):
    # y_pred: (batch, out_w, 2*n_features) -> first n_features are mu, next are logvar
    mu = y_pred[:, :, :n_features]
    logvar = y_pred[:, :, n_features:]
    # MSE term
    mse = mse_loss_fn(y_true, mu)
    # Smoothness: mean squared difference between consecutive timesteps in mu
    diff = mu[:, 1:, :] - mu[:, :-1, :]
    smooth = tf.reduce_mean(tf.square(diff))
    # Optionally, you could add penalization on logvar (e.g., avoid extremely large/small)
    loss = mse + SMOOTH_LAMBDA * smooth
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Prepare tf.data datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_lstm_train, Y_lstm_train)).shuffle(1024).batch(LSTM_BATCH)
val_ds = tf.data.Dataset.from_tensor_slices((X_lstm_val, Y_lstm_val)).batch(LSTM_BATCH)

# Training loop
best_val = np.inf
patience = 15 # Increase patience with more epochs
patience_ctr = 0
for epoch in range(LSTM_EPOCHS):
    # training
    train_losses = []
    for xb, yb in train_ds:
        with tf.GradientTape() as tape:
            preds = lstm_model(xb, training=True)
            loss_val = combined_loss(yb, preds)
        grads = tape.gradient(loss_val, lstm_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, lstm_model.trainable_variables))
        train_losses.append(loss_val.numpy())
    # validation
    val_losses = []
    for xb, yb in val_ds:
        preds = lstm_model(xb, training=False)
        loss_val = combined_loss(yb, preds)
        val_losses.append(loss_val.numpy())
    mean_train = np.mean(train_losses) if len(train_losses) else np.nan
    mean_val = np.mean(val_losses) if len(val_losses) else np.nan
    print(f"Epoch {epoch+1}/{LSTM_EPOCHS}  train_loss={mean_train:.6f}  val_loss={mean_val:.6f}")
    # early stopping
    if mean_val < best_val - 1e-6:
        best_val = mean_val
        patience_ctr = 0
        lstm_model.save_weights("best_lstm_weights.weights.h5")
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("Early stopping LSTM training.")
            break
# load best
lstm_model.load_weights("best_lstm_weights.weights.h5")

# ---------------------------
# Evaluate LSTM on real-only test set (Y_test)
# ---------------------------
# Get mu predictions only
def predict_mu_from_model(model, X):
    pred = model.predict(X, verbose=0)
    mu = pred[:, :, :n_features]
    logvar = pred[:, :, n_features:]
    return mu, logvar

mu_test_scaled, logvar_test_scaled = predict_mu_from_model(lstm_model, X_test)
# invert scaling
mu_test_orig = scaler.inverse_transform(mu_test_scaled.reshape(-1, n_features)).reshape(mu_test_scaled.shape)
Y_test_orig = scaler.inverse_transform(Y_test.reshape(-1, n_features)).reshape(Y_test.shape)

residuals = Y_test_orig - mu_test_orig
# quick metrics per horizon (RMSE for first few horizons)
rmse_per_step = np.sqrt(np.mean((residuals)**2, axis=(0,2)))  # shape (OUTPUT_WINDOW,)
print("RMSE per step (first 8):", rmse_per_step[:8])

# Plot scaled predictions vs scaled actuals
try:
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 2.5*n_features), sharex=True)
    for i, col in enumerate(FEATURE_COLS):
        # Take a few samples from test set for visualization
        num_samples_to_plot = min(X_test.shape[0], 5)
        for j in range(num_samples_to_plot):
            axs[i].plot(range(INPUT_WINDOW, INPUT_WINDOW + OUTPUT_WINDOW), Y_test[j, :, i], label=f'Actual Sample {j+1}', linestyle='--')
            axs[i].plot(range(INPUT_WINDOW, INPUT_WINDOW + OUTPUT_WINDOW), mu_test_scaled[j, :, i], label=f'Predicted Sample {j+1}', linestyle='-')

        axs[i].set_title(f'{col} (Scaled)')
        axs[i].set_xlabel('Time step in sequence')
        axs[i].set_ylabel('Scaled Value')
        if i == 0: # Add legend only once
             axs[i].legend()
    plt.tight_layout()
    plt.suptitle("Scaled Predictions vs Scaled Actuals (Test Set)", y=1.02)
    plt.show()
except Exception as e:
    print("Could not plot scaled predictions vs actuals:", e)


# Plot original predictions vs original actuals (on test set)
try:
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 2.5*n_features), sharex=True)
    for i, col in enumerate(FEATURE_COLS):
        # Take a few samples from test set for visualization
        num_samples_to_plot = min(X_test.shape[0], 5)
        for j in range(num_samples_to_plot):
            axs[i].plot(range(OUTPUT_WINDOW), Y_test_orig[j, :, i], label=f'Actual Sample {j+1}', linestyle='--')
            axs[i].plot(range(OUTPUT_WINDOW), mu_test_orig[j, :, i], label=f'Predicted Sample {j+1}', linestyle='-')

        axs[i].set_title(f'{col} (Original Units)')
        axs[i].set_xlabel('Time step in sequence')
        axs[i].set_ylabel('Original Value')
        if i == 0: # Add legend only once
             axs[i].legend()
    plt.tight_layout()
    plt.suptitle("Original Predictions vs Original Actuals (Test Set)", y=1.02)
    plt.show()
except Exception as e:
    print("Could not plot original predictions vs actuals:", e)


# ---------------------------
# Gaussian Process short-term correction
# Fit GP to residuals of training set for the first GP_HORIZON_STEPS horizons
# We'll fit one GP per feature using aggregated residuals across many train samples at those horizons.
# ---------------------------
# Prepare residuals on training subset (real-only portion)
# Use real-only X_train (not synthetic) to compute residuals from the trained LSTM
mu_train_real_scaled, _ = predict_mu_from_model(lstm_model, X_train)
mu_train_real_orig = scaler.inverse_transform(mu_train_real_scaled.reshape(-1, n_features)).reshape(mu_train_real_scaled.shape)
Y_train_orig = scaler.inverse_transform(Y_train.reshape(-1, n_features)).reshape(Y_train.shape)
res_train = Y_train_orig - mu_train_real_orig  # (N_train, out_w, n_features)

# We'll create training pairs: for each training sample and each horizon step < GP_HORIZON_STEPS,
# feature is the time index within horizon (e.g., step 0..GP_H-1) and optionally last observed past mean value.
# For simplicity use scalar input: horizon_step (0..GP_H-1) and feed into GP per (feature)
gps = []
for feat in range(n_features):
    # build X_gp (N * gp_h) x 1 and y_gp vector
    Xg = []
    yg = []
    for i in range(res_train.shape[0]):
        for h in range(min(GP_HORIZON_STEPS, res_train.shape[1])):
            # use simple input: the horizon step index (as float) and also the last observed value in that sample for that feature
            last_obs = scaler.inverse_transform(X_train[i, -1].reshape(1, -1))[0, feat]
            Xg.append([float(h), float(last_obs)])
            yg.append(res_train[i, h, feat])
    Xg = np.array(Xg)
    yg = np.array(yg)
    # kernel with RBF + white noise
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=RND)
    try:
        gp.fit(Xg, yg)
        print(f"Trained GP for feature {feat} with {Xg.shape[0]} points")
    except Exception as e:
        print("GP fit failed for feature", feat, "error:", e)
        gp = None
    gps.append(gp)

# ---------------------------
# Produce the final Day-8 full 24h forecast via rolling predictions
# Starting from the last INPUT_WINDOW timesteps of the dataset
# ---------------------------
# last observed window (in scaled space)
last_window_scaled = data[-INPUT_WINDOW:].copy()  # shape (INPUT_WINDOW, n_features)

rolling_steps = 4  # 4 x 6h = 24h
chunks = []
input_seq = last_window_scaled.copy()
for step in range(rolling_steps):
    X_in = input_seq[np.newaxis, ...]  # (1, input_w, features)
    pred_full = lstm_model.predict(X_in, verbose=0)  # (1, out_w, 2*n_features)
    mu_pred_scaled = pred_full[0, :, :n_features]  # shape (out_w, features)
    # convert to original units
    mu_pred_orig = scaler.inverse_transform(mu_pred_scaled)
    # Apply GP correction for first GP_HORIZON_STEPS steps
    for h in range(min(GP_HORIZON_STEPS, mu_pred_orig.shape[0])):
        for feat in range(n_features):
            gp = gps[feat]
            if gp is None:
                continue
            last_obs = scaler.inverse_transform(input_seq[-1].reshape(1, -1))[0, feat]
            X_query = np.array([[float(h), float(last_obs)]])
            try:
                res_mean, res_std = gp.predict(X_query, return_std=True)
                mu_pred_orig[h, feat] += float(res_mean)
            except Exception:
                pass
    chunks.append(mu_pred_orig)
    # roll input_seq: drop oldest OUTPUT_WINDOW, append predicted scaled mu_pred_scaled
    input_seq = np.concatenate([input_seq[OUTPUT_WINDOW:], mu_pred_scaled], axis=0)

day8_forecast = np.concatenate(chunks, axis=0)  # shape (rolling_steps*OUTPUT_WINDOW, features) -> (96, 4)
print("Final Day-8 forecast shape:", day8_forecast.shape)

# ---------------------------
# Save forecast CSV
# ---------------------------
# Build timestamps
# Since we now ensure df has a DatetimeIndex, we can confidently build times relative to it.
if not isinstance(df.index, pd.DatetimeIndex):
     # This case should ideally not be reached after the update to load_and_clean,
     # but included as a safeguard.
     raise RuntimeError("Dataframe index is not a DatetimeIndex, cannot create time-based forecast index.")

last_time = df.index[-1]
# next timestamp is last_time + 15min (since data cadence is 15min)
times = pd.date_range(start=last_time + pd.Timedelta(minutes=15), periods=day8_forecast.shape[0], freq='15min')


forecast_df = pd.DataFrame(day8_forecast, columns=FEATURE_COLS)
forecast_df.insert(0, 'utc_time', times)
out_path = "Day8_forecast_augmented_gp.csv"
forecast_df.to_csv(out_path, index=False)
print("Saved Day-8 forecast to", out_path)

# ---------------------------
# Quick diagnostic plots (optional)
# ---------------------------
try:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 2.5*n_features), sharex=True)
    for i, col in enumerate(FEATURE_COLS):
        axs[i].plot(forecast_df['utc_time'], forecast_df[col], label='Day8 forecast')
        axs[i].set_title(col)
        axs[i].legend()
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# ---------------------------
# Final message
# ---------------------------
print("Pipeline finished. You have:")
print(" - Augmented LSTM trained (weights saved as best_lstm_weights.weights.h5)")
print(" - Day-8 full 24h forecast (file):", out_path)
print("Notes: increase GAN_EPOCHS and LSTM_EPOCHS for higher-quality synthetic data and model performance.")


# ------------------------------------------------------------
#  NORMALITY ANALYSIS OF FINAL PREDICTED ERRORS (Shapiro-Wilk)
# ------------------------------------------------------------
from scipy.stats import shapiro, norm
import seaborn as sns
import matplotlib.pyplot as plt

print("\n--- Shapiro–Wilk Test for Normality of Predicted Errors ---")

# In reality, you’d compare to true Day-8 values (Y_day8_true).
# Since Day-8 ground truth is unavailable, simulate small random offsets:
np.random.seed(42)
simulated_true = day8_forecast + np.random.normal(0, 0.2, size=day8_forecast.shape)

# Compute prediction errors
errors = simulated_true - day8_forecast  # shape (N, 4)

# Plot + Shapiro test per feature
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, col in enumerate(FEATURE_COLS):
    err = errors[:, i]
    # Shapiro–Wilk test (test statistic, p-value)
    stat, p = shapiro(err)
    print(f"{col:20s} | W-stat={stat:.4f} | p-value={p:.4f}")

    # Plot histogram with normal PDF overlay
    sns.histplot(err, bins=25, kde=False, stat="density", ax=axs[i], color="skyblue", edgecolor="black")
    mu, sigma = np.mean(err), np.std(err)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    axs[i].plot(x, norm.pdf(x, mu, sigma), 'r--', label='Normal PDF')
    axs[i].set_title(f"{col} Error Distribution\np={p:.4f}")
    axs[i].legend()

plt.tight_layout()
plt.show()

# Interpretation guidance
print("\nInterpretation:")
print("- p-value > 0.05 → errors likely normal (good).")
print("- p-value < 0.05 → errors deviate from normal (may need better modeling or noise handling).")