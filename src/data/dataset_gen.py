try:
    import tensorflow as tf
except:
    pass

def image_parser_generator(input_shape):
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, input_shape)
        return image, image
    
    return parse_image

def paths_to_dataset(image_paths, image_shape):
    image_parser = image_parser_generator(image_shape)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths,)).map(image_parser).batch(64).prefetch(-1)

    return dataset