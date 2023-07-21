import tensorflow as tf

BASE_LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 10


def _create_base_model(image_size):
    IMG_SHAPE = image_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet', classes=4)
    base_model.trainable = False
    base_model.summary()
    return base_model


def _split_features_labels(base_model, train_dataset):
    image_batch, label_batch = next(iter(train_dataset))
    return base_model(image_batch)


def _create_averaging_layer(feature_batch):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    return global_average_layer, feature_batch_average


def _create_prediction_layer(feature_batch_average):
    prediction_layer = tf.keras.layers.Dense(4, activation='softmax')
    prediction_batch = prediction_layer(feature_batch_average)
    return prediction_layer, prediction_batch


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def call(self, inputs):
        return _preprocess_input()(inputs)


def crete_layered_model(image_size, train_dataset):
    base_model = _create_base_model(image_size)
    features_batch = _split_features_labels(base_model, train_dataset)
    global_average_layer, feature_batch_average = _create_averaging_layer(features_batch)
    prediction_layer, prediction_batch = _create_prediction_layer(feature_batch_average)
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = PreprocessLayer()(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return base_model, model


def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()


def fit_model(model, train_dataset, validation_dataset):
    model.evaluate(validation_dataset)

    history = model.fit(train_dataset,
                        epochs=INITIAL_EPOCHS,
                        validation_data=validation_dataset)
    return history


def _refine_trainable_layers(base_model):
    base_model.trainable = True

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False


def _compile_for_fine_tune(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=BASE_LEARNING_RATE / 10),
                  metrics=['accuracy'])
    model.summary()


def fine_tune(model, base_model, train_dataset, validation_dataset, history):
    _refine_trainable_layers(base_model)
    _compile_for_fine_tune(model)
    fine_tune_epochs = 10
    total_epochs = INITIAL_EPOCHS + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)
    return history_fine


def _preprocess_input():
    return tf.keras.applications.mobilenet_v2.preprocess_input
