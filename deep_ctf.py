"""Generate CTF team names using a recurrent neural network.

Some useful links:
1. https://stackabuse.com/gpt-style-text-generation-in-python-with-tensorflowkeras/
2. https://towardsdatascience.com/data-to-text-generation-with-t5-building-a-simple-yet-advanced-nlg-model-b5cce5a6df45
3. https://towardsdatascience.com/text-generation-with-bi-lstm-in-pytorch-5fda6e7cc22c
4. https://livecodestream.dev/post/lstm-based-name-generator-first-dive-into-nlp/
5. https://www.thepythoncode.com/article/text-generation-keras-python
6. https://stackabuse.com/python-for-nlp-deep-learning-text-generation-with-keras/

"""

import numpy as np
import logging as log
import argparse

from alive_progress import alive_bar
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow import keras
from typing import Tuple, List, Dict


NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MODEL_VERSION = 'v1'


def load_data(dataset_path: str = 'teams.txt') -> Tuple[np.ndarray, np.ndarray, int, Dict, Dict, int, str, List[str]]:
    """Load dataset from file.

    Args:
        dataset_path (str, optional): Path to dataset. Defaults to 'teams.txt'.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, Dict, Dict, int, str, List[str]]:
            X: Input data.
            y: Output data.
            n_vocab: Number of unique characters.
            int_to_char: Mapping from integer to character.
            char_to_int: Mapping from character to integer.
            max_seq_length: Maximum length of a name.
            concat_names: Concatenated names, mostly for seeding.
            names: List of names.
    """
    with open(dataset_path, 'r') as f:
        names = [name.lower() for name in f.read().split('\n')]

    concat_names = '\n'.join(
        [name for name in names if len(name) >= 4 and len(name) <= 20])

    unique_chars = sorted(list(set(''.join(concat_names))))

    char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
    int_to_char = dict((i, c) for i, c in enumerate(unique_chars))

    n_vocab = len(unique_chars)
    max_seq_length = max([len(name) for name in names])

    log.info(f'Vocab size: {n_vocab}')
    log.info(f'Max length: {max_seq_length}')

    sequences_dataX = []
    next_chars_dataY = []

    for i in range(0, len(concat_names) - max_seq_length, 1):
        sequences_dataX.append(concat_names[i:i + max_seq_length])
        next_chars_dataY.append(concat_names[i + max_seq_length])

    dataset_len = len(sequences_dataX)

    X = np.zeros((dataset_len, max_seq_length, n_vocab), dtype=np.bool8)
    y = np.zeros((dataset_len, n_vocab), dtype=np.bool8)

    for i, sequence in enumerate(sequences_dataX):
        for j, char in enumerate(sequence):
            X[i, j, char_to_int[char]] = 1
            y[i, char_to_int[next_chars_dataY[i]]] = 1

    return X, y, n_vocab, int_to_char, char_to_int, max_seq_length, concat_names, names


def create_model(
        in_max_seq_length: int, in_n_vocab: int, out_dim: int) -> Sequential:
    """Create model.

    Args:
        in_max_seq_length (int): Maximum length of a name.
        in_n_vocab (int): Number of unique characters.
        out_dim (int): Output dimension.

    Returns:
        Sequential: Built model.
    """
    model = Sequential()
    model.add(GRU(128, input_shape=(in_max_seq_length, in_n_vocab),
                  return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(GRU(128, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(96, activation='tanh'))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])

    return model


def sample(preds: np.ndarray, temperature: float = 1.0) -> int:
    """Sample an index from a probability array.

    Args:
        preds (np.ndarray): Array of probabilities.
        temperature (float, optional): Temperature. Defaults to 1.0.

    Returns:
        int: Index of the sampled character.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_names(total_names: int, model: Sequential,
                   int_to_char: Dict[int, str],
                   char_to_int: Dict[str, int],
                   n_vocab: int, seed: str, max_seq_length: int,
                   names: List[str]) -> List[str]:
    """Generate names.

    Args:
        total_names (int): Number of names to generate.
        model (Sequential): Model to use for generating names.
        int_to_char (Dict[int, str]): Mapping from integer to character.
        char_to_int (Dict[str, int]): Mapping from character to integer.
        n_vocab (int): Number of unique characters.
        seed (str): Seed to start generating names.
        max_seq_length (int): Sequence length.
        names (List[str]): List of names to avoid duplicates.

    Returns:
        List[str]: List of generated names.
    """
    new_names: List[str] = []
    sequence = seed
    with alive_bar(total_names) as bar:
        while len(new_names) < total_names:
            x = np.zeros((1, max_seq_length, n_vocab))
            for t, char in enumerate(sequence):
                x[0, t, char_to_int[char]] = 1

            # Sample a character from the model
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, 0.5)
            next_char = int_to_char[next_index]

            sequence = sequence[1:] + next_char

            # If we have a new line, we have a new name
            if next_char == '\n':
                gen_name = [name for name in sequence.split('\n')][1]

                # Discard all names that are too short or too long
                if len(gen_name) >= 4 and len(gen_name) <= 20:
                    # Only allow new and unique names
                    if gen_name not in names + new_names:
                        new_names.append(gen_name)
                        bar.text(f'-> Current name: {gen_name}')
                        bar()

    return new_names


def plot_history(history, max_seq_length: int, latest_accuracy: float):
    """Plot history.

    Args:
        history (History): History object.
        n_vocab (int): Number of unique characters.
        int_to_char (Dict[int, str]): Mapping from integer to character.
        max_seq_length (int): Sequence length.
        latest_accuracy (float): Latest accuracy.
    """
    # Plot loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(
        f'name_generator-loss-{NUM_EPOCHS}-{LEARNING_RATE}-{BATCH_SIZE}-{max_seq_length}-{latest_accuracy}-{MODEL_VERSION}.png',
        dpi=1200)

    # Plot accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(
        f'name_generator-acc-{NUM_EPOCHS}-{LEARNING_RATE}-{BATCH_SIZE}-{max_seq_length}-{latest_accuracy}-{MODEL_VERSION}.png',
        dpi=1200)


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--generate', action='store_true', help='Generate names')
    parser.add_argument('--model_path', type=str, default='./name_generator{}.h5', help='Path to the model')
    parser.add_argument('--num_names', type=int, default=10, help='Number of names to generate')
    args = parser.parse_args()

    return args


def main():
    """Main function."""
    args = parse_args()

    X, y, n_vocab, int_to_char, char_to_int, max_seq_length, concat_names, names = load_data()
    model = create_model(max_seq_length, n_vocab, y.shape[1])

    if args.train:
        # Save model on every epoch
        filepath = 'models/weights-improvement-{epoch:02d}-{loss:.4f}.h5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(
            X, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=callbacks_list)
        model.save(args.model_path.format('-train'))

        # Get latest accuracy and loss
        latest_accuracy = history.history['val_accuracy'][-1]

        # Plot history
        plot_history(history, max_seq_length, latest_accuracy)
    elif args.generate:
        model.load_weights(args.model_path.format('-pretrained'))
        names = generate_names(
            args.num_names,  # Number of names to generate
            model,           # Model to use for name generation
            int_to_char,     # Integer to character mapping
            char_to_int,     # Character to integer mapping
            n_vocab,         # Number of unique characters
            concat_names[-(max_seq_length - 1):] + '\n',  # Seed
            max_seq_length,  # Max length of generated names
            names            # Existing names (to avoid duplicates)
        )

        for name in names:
            print(name)
    else:
        raise ValueError('Please specify --train or --generate')


if __name__ == '__main__':
    main()
