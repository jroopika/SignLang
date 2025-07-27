import tensorflow as tf

def load_dataset(data_dir, img_size=(64, 64), batch_size=32):
    original_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    class_names = original_ds.class_names

    # Normalize after capturing class_names
    dataset = original_ds.map(lambda x, y: (x / 255.0, y))
    
    return dataset, class_names
