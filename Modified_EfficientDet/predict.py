
import tensorflow as tf

def save_preds(predictions):
    #Function that saves away the predictions as jpg images.


def main():

    #Load test data

    model = tf.keras.model.load_model("efficientdet.h5")
    preds = model.predict(test_data)
    save_preds(preds)


if __name__ == '__main__':
    main()
