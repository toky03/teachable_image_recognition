import matplotlib.pyplot as plt


def print_example_images(dataset):
    class_names = dataset.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.show()


def print_evaluation_chart(history, initial_epochs=0, acc=[], val_acc=[], loss=[], val_loss=[]):
    acc += history.history['accuracy']
    val_acc += history.history['val_accuracy']

    loss += history.history['loss']
    val_loss += history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    return acc, val_acc, loss, val_loss


def print_example_predictions(predictions, label_batch, image_batch, class_names):
    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")
    plt.show()
