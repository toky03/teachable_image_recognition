import tensorflow as tf

import model
import presentation

IMG_SIZE = (160, 160)
BATCH_SIZE = 10


def load_data(images_folder):
    train_ds = tf.keras.utils.image_dataset_from_directory(images_folder,
                                                           validation_split=0.2,
                                                           subset='training',
                                                           seed=123,
                                                           image_size=IMG_SIZE,
                                                           batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(images_folder,
                                                         validation_split=0.2,
                                                         subset='validation',
                                                         seed=123,
                                                         image_size=IMG_SIZE,
                                                         batch_size=BATCH_SIZE)
    return train_ds, val_ds


def split_batches(validation_dataset):
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 2)
    validation_dataset = validation_dataset.skip(val_batches // 2)
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
    return test_dataset, validation_dataset


def save_tf_lite_model(layered_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(layered_model)
    tflite_model = converter.convert()
    with open('../layered_model.tflite', 'wb') as f:
        f.write(tflite_model)


def add_auto_tune(train_dataset, test_dataset, validation_dataset):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, validation_dataset


def create_predictions(layered_model, test_dataset):
    loss, accuracy = layered_model.evaluate(test_dataset)
    print('Test accuracy:', accuracy)
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = layered_model.predict_on_batch(image_batch).flatten()

    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    return predictions, image_batch, label_batch


def create_model(train_dataset, validation_dataset, retrain):
    if not retrain:
        layered_model = tf.keras.models.load_model('../layered_model')
        save_tf_lite_model(layered_model)
        return layered_model
    base_model, layered_model = model.crete_layered_model(IMG_SIZE, train_dataset)
    model.compile_model(layered_model)
    history = model.fit_model(layered_model, train_dataset, validation_dataset)
    acc, val_acc, loss, val_loss = presentation.print_evaluation_chart(history)
    history_finetuned = model.fine_tune(layered_model, base_model, train_dataset, validation_dataset, history)
    presentation.print_evaluation_chart(history_finetuned, 10, acc, val_acc, loss, val_loss)
    base_model.save('../base_model')
    layered_model.save('../layered_model')
    save_tf_lite_model(layered_model)
    return layered_model


def main(train=False):
    train_ds, val_ds = load_data('../image_set')
    presentation.print_example_images(train_ds)
    test_dataset, validation_dataset = split_batches(val_ds)
    train_dataset, test_dataset, validation_dataset = add_auto_tune(train_ds, test_dataset, validation_dataset)
    layered_model = create_model(train_dataset, validation_dataset, train)
    predictions, image_batch, label_batch = create_predictions(layered_model, test_dataset)
    class_names = train_ds.class_names
    presentation.print_example_predictions(predictions, label_batch, image_batch, class_names)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
