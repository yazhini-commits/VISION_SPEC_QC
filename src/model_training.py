from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2( 
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

model.summary()
