img_input = Input(shape=(128,128,1))

#Encoder Units
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
conv1_1 = Dropout(0.2)(conv1_1)
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_1)
pool1_1 = MaxPooling2D((2, 2))(conv1_1)

conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1_1)
conv2_1 = Dropout(0.2)(conv2_1)
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_1)
pool2_1 = MaxPooling2D((2, 2))(conv2_1)

#Skip Connection Unit-1
up1_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv2_1)
up1_2 = concatenate([up1_2, conv1_1])
conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1_2)
conv1_2 = Dropout(0.2)(conv1_2)
conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_2)

conv3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2_1)
conv3_1 = Dropout(0.2)(conv3_1)
conv3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3_1)
pool3_1 = MaxPooling2D((2, 2))(conv3_1)

#Skip Connection Unit-2 and Unit-3
up2_2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv3_1)
up2_2 = concatenate([up2_2, conv2_1])
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2_2)
conv2_2 = Dropout(0.2)(conv2_2)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_2)

up1_3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv2_2)
up1_3 = concatenate([up1_3, conv1_1, conv1_2])
conv1_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1_3)
conv1_3 = Dropout(0.2)(conv1_3)
conv1_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_3)

#The bridge between the Encoder and Decoder units
conv4_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
conv4_1 = Dropout(0.2)(conv4)
conv4_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

#Decoder Units
up3_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv4_1)
up3_2 = concatenate([up3_2, conv3_1])
conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3_2)
conv3_2 = Dropout(0.2)(conv3_2)
conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3_2)


up2_3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv3_2)
up2_3 = concatenate([up2_3, conv2_1, conv2_2])
conv2_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2_3)
conv2_3 = Dropout(0.2)(conv2_3)
conv2_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_3)


up1_4 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(conv2_3)
up1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3])
conv1_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1_4)
conv1_4 = Dropout(0.2)(conv1_4)
conv1_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_4)

#Nested Output for deepsupervision
nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)

if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3])
    else:
        model = Model(input=img_input, output=[nestnet_output_3])

model.summary()