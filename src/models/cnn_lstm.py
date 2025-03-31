import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Attention, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def build_cnn_lstm_han_model(input_shape, learning_rate=0.0005):
    """
    Build the hybrid CNN-LSTM model with Hierarchical Attention Networks.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    inp = Input(shape=input_shape, name='price_input')
    
    # CNN layer for feature extraction
    x = Conv1D(32, kernel_size=3, activation='relu')(inp)
    
    # Bidirectional LSTM with Attention
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    att = Attention()([x, x])
    x = GlobalMaxPooling1D()(att)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    out = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0),
        loss=Huber(),
        metrics=['mae']
    )
    
    return model
