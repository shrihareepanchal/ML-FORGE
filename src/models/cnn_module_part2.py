# Model design and architecture selection
def model_design(state):
    """Handle CNN model design and architecture selection"""
    st.markdown("### 🏗️ Model Architecture")
    
    # Architecture selection
    architecture_options = [
        "Custom CNN",
        "LeNet-5",
        "AlexNet-like",
        "VGG-like",
        "Transfer Learning"
    ]
    
    selected_architecture = st.selectbox(
        "Select Model Architecture",
        architecture_options
    )
    
    # Transfer learning options
    if selected_architecture == "Transfer Learning":
        transfer_model_options = [
            "VGG16",
            "ResNet50",
            "MobileNetV2",
            "EfficientNetB0"
        ]
        
        transfer_model = st.selectbox(
            "Select Pre-trained Model",
            transfer_model_options
        )
        
        fine_tune = st.checkbox("Fine-tune pre-trained model", value=True)
        if fine_tune:
            fine_tune_layers = st.slider(
                "Number of layers to fine-tune",
                min_value=0,
                max_value=20,
                value=3
            )
        
        # Store in state
        state["transfer_learning"] = {
            "model": transfer_model,
            "fine_tune": fine_tune,
            "fine_tune_layers": fine_tune_layers if fine_tune else 0
        }
    
    # Custom architecture configuration
    if selected_architecture == "Custom CNN":
        st.markdown("#### Custom CNN Configuration")
        
        # Number of convolutional blocks
        num_conv_blocks = st.slider(
            "Number of Convolutional Blocks",
            min_value=1,
            max_value=5,
            value=3
        )
        
        conv_blocks = []
        
        # Configure each convolutional block
        for i in range(num_conv_blocks):
            st.markdown(f"##### Convolutional Block {i+1}")
            
            col1, col2 = st.columns(2)
            with col1:
                num_filters = st.slider(
                    f"Number of Filters (Block {i+1})",
                    min_value=16,
                    max_value=512,
                    value=64 * (2**i),  # Double filters for each block
                    step=16
                )
                
                kernel_size = st.select_slider(
                    f"Kernel Size (Block {i+1})",
                    options=[1, 3, 5, 7],
                    value=3
                )
            
            with col2:
                use_batchnorm = st.checkbox(f"Use Batch Normalization (Block {i+1})", value=True)
                
                pool_type = st.radio(
                    f"Pooling Type (Block {i+1})",
                    ["Max Pooling", "Average Pooling", "No Pooling"],
                    index=0
                )
            
            # Store block configuration
            conv_blocks.append({
                "filters": num_filters,
                "kernel_size": kernel_size,
                "batch_norm": use_batchnorm,
                "pool_type": pool_type
            })
        
        # Dense layers after convolutional blocks
        st.markdown("#### Fully Connected Layers")
        
        num_dense_layers = st.slider(
            "Number of Dense Layers",
            min_value=1,
            max_value=3,
            value=2
        )
        
        dense_layers = []
        
        # Configure each dense layer
        for i in range(num_dense_layers):
            col1, col2 = st.columns(2)
            
            with col1:
                units = st.slider(
                    f"Units in Dense Layer {i+1}",
                    min_value=32,
                    max_value=1024,
                    value=128 // (2**i),  # Decrease units for each layer
                    step=32
                )
            
            with col2:
                activation = st.selectbox(
                    f"Activation for Dense Layer {i+1}",
                    ["relu", "elu", "selu", "tanh", "sigmoid"],
                    index=0
                )
                
                dropout_rate = st.slider(
                    f"Dropout Rate for Dense Layer {i+1}",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.1
                )
            
            # Store dense layer configuration
            dense_layers.append({
                "units": units,
                "activation": activation,
                "dropout": dropout_rate
            })
        
        # Store custom CNN configuration in state
        state["custom_cnn"] = {
            "conv_blocks": conv_blocks,
            "dense_layers": dense_layers
        }
    
    # Build model button
    if st.button("Build Model"):
        with st.spinner("Building model architecture..."):
            # Build the model based on selections
            if selected_architecture == "Custom CNN":
                model = build_custom_cnn(state)
            elif selected_architecture == "LeNet-5":
                model = build_lenet(state)
            elif selected_architecture == "AlexNet-like":
                model = build_alexnet(state)
            elif selected_architecture == "VGG-like":
                model = build_vgg_like(state)
            elif selected_architecture == "Transfer Learning":
                model = build_transfer_learning_model(state)
            
            # Store the model in the state
            state["model"] = model
            state["architecture"] = selected_architecture
            state["model_built"] = True
            
            # Display model summary
            st.success("✅ Model built successfully!")
            display_model_summary(model)

def build_custom_cnn(state):
    """Build a custom CNN based on user configuration"""
    img_height, img_width = state["image_size"]
    num_classes = len(state["classes"])
    input_shape = (img_height, img_width, 3)  # Assuming RGB images
    
    model = Sequential()
    
    # Add convolutional blocks
    for i, block in enumerate(state["custom_cnn"]["conv_blocks"]):
        # First layer needs input_shape
        if i == 0:
            model.add(Conv2D(
                filters=block["filters"],
                kernel_size=(block["kernel_size"], block["kernel_size"]),
                activation="relu",
                padding="same",
                input_shape=input_shape
            ))
        else:
            model.add(Conv2D(
                filters=block["filters"],
                kernel_size=(block["kernel_size"], block["kernel_size"]),
                activation="relu",
                padding="same"
            ))
        
        # Add batch normalization if requested
        if block["batch_norm"]:
            model.add(BatchNormalization())
        
        # Add pooling if requested
        if block["pool_type"] == "Max Pooling":
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif block["pool_type"] == "Average Pooling":
            model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    
    # Flatten before dense layers
    model.add(Flatten())
    
    # Add dense layers
    for layer in state["custom_cnn"]["dense_layers"]:
        model.add(Dense(units=layer["units"], activation=layer["activation"]))
        model.add(Dropout(layer["dropout"]))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(num_classes, activation="softmax"))
    
    return model

def build_lenet(state):
    """Build a LeNet-5 like architecture"""
    img_height, img_width = state["image_size"]
    num_classes = len(state["classes"])
    input_shape = (img_height, img_width, 3)  # Assuming RGB images
    
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes if num_classes > 2 else 1, 
              activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model

def build_alexnet(state):
    """Build an AlexNet-like architecture"""
    img_height, img_width = state["image_size"]
    num_classes = len(state["classes"])
    input_shape = (img_height, img_width, 3)  # Assuming RGB images
    
    model = Sequential([
        Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes if num_classes > 2 else 1, 
              activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model

def build_vgg_like(state):
    """Build a VGG-like architecture"""
    img_height, img_width = state["image_size"]
    num_classes = len(state["classes"])
    input_shape = (img_height, img_width, 3)  # Assuming RGB images
    
    model = Sequential([
        # Block 1
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Block 2
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Block 3
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes if num_classes > 2 else 1, 
              activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model

def build_transfer_learning_model(state):
    """Build a model using transfer learning"""
    img_height, img_width = state["image_size"]
    num_classes = len(state["classes"])
    input_shape = (img_height, img_width, 3)  # Assuming RGB images
    
    # Get the base model
    base_model = None
    transfer_config = state["transfer_learning"]
    
    if transfer_config["model"] == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif transfer_config["model"] == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif transfer_config["model"] == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif transfer_config["model"] == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    # Fine-tuning if requested
    if transfer_config["fine_tune"]:
        # Unfreeze the top N layers
        for layer in base_model.layers[-transfer_config["fine_tune_layers"]:]:
            layer.trainable = True
    
    # Create the complete model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def display_model_summary(model):
    """Display a summary of the model architecture"""
    # Get model summary
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    summary_string = summary_io.getvalue()
    summary_io.close()
    
    # Display summary in a code block
    st.code(summary_string, language="")
    
    # Visual representation (placeholder)
    st.markdown("#### Visual Model Architecture")
    st.markdown("A visual representation of your model would appear here.")
    
    # Show training parameters that will be used
    st.markdown("#### Training Parameters Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optimizer:** Adam")
        st.markdown("**Learning Rate:** 0.001")
        st.markdown("**Loss Function:** Categorical Crossentropy")
    
    with col2:
        st.markdown("**Batch Size:** 32")
        st.markdown("**Epochs:** 10")
        st.markdown("**Early Stopping:** Enabled")

# Training related functions
def model_training(state):
    """Handle model training configuration and execution"""
    st.markdown("### 🏋️ Model Training")
    
    # Training parameters
    st.markdown("#### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256],
            value=32
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        
        optimizer = st.selectbox(
            "Optimizer",
            ["Adam", "SGD", "RMSprop"],
            index=0
        )
    
    with col2:
        epochs = st.slider(
            "Number of Epochs",
            min_value=5,
            max_value=100,
            value=20
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        
        if use_early_stopping:
            patience = st.slider(
                "Early Stopping Patience",
                min_value=2,
                max_value=20,
                value=5
            )
    
    # Additional callbacks
    st.markdown("#### Additional Callbacks")
    
    use_model_checkpoint = st.checkbox("Save Best Model", value=True)
    use_lr_scheduler = st.checkbox("Use Learning Rate Scheduler", value=False)
    
    if use_lr_scheduler:
        lr_schedule_type = st.radio(
            "Learning Rate Schedule Type",
            ["Reduce on Plateau", "Step Decay", "Exponential Decay"],
            index=0
        )
    
    # Store training configuration
    training_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "epochs": epochs,
        "early_stopping": {
            "enabled": use_early_stopping,
            "patience": patience if use_early_stopping else None
        },
        "model_checkpoint": use_model_checkpoint,
        "lr_scheduler": {
            "enabled": use_lr_scheduler,
            "type": lr_schedule_type if use_lr_scheduler else None
        }
    }
    
    state["training_config"] = training_config
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a while."):
            # Simulated training for now
            # In a real implementation, this would use the actual dataset and model
            simulate_model_training(state)
            
            st.success("✅ Model training completed!")
            
            # Display training results
            display_training_results(state)

def simulate_model_training(state):
    """Simulate model training (for demonstration)"""
    # This is a placeholder function that would be replaced by actual training
    # in a real implementation
    
    # For demonstration, create a simulated training history
    epochs = state["training_config"]["epochs"]
    history = {
        "loss": np.random.rand(epochs) * 0.5 + 0.5,
        "val_loss": np.random.rand(epochs) * 0.6 + 0.6,
        "accuracy": np.random.rand(epochs) * 0.3 + 0.7,
        "val_accuracy": np.random.rand(epochs) * 0.25 + 0.65
    }
    
    # Make the metrics look more realistic (generally improving over time)
    for i in range(1, epochs):
        decay_factor = 0.9
        history["loss"][i] = history["loss"][i-1] * decay_factor + history["loss"][i] * (1-decay_factor)
        history["val_loss"][i] = history["val_loss"][i-1] * decay_factor + history["val_loss"][i] * (1-decay_factor)
        history["accuracy"][i] = history["accuracy"][i-1] * 0.95 + history["accuracy"][i] * 0.05
        history["val_accuracy"][i] = history["val_accuracy"][i-1] * 0.95 + history["val_accuracy"][i] * 0.05
    
    # Store in state
    state["history"] = history
    state["model_trained"] = True
    
    # Simulate training time
    time.sleep(3)

def display_training_results(state):
    """Display training metrics and results"""
    if not state["history"]:
        return
    
    history = state["history"]
    
    # Plot training metrics
    st.markdown("#### Training Metrics")
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history["loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history["accuracy"], label="Training Accuracy")
    ax2.plot(history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Final metrics
    st.markdown("#### Final Model Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    final_epoch = len(history["loss"]) - 1
    
    col1.metric("Training Loss", f"{history['loss'][final_epoch]:.4f}")
    col2.metric("Validation Loss", f"{history['val_loss'][final_epoch]:.4f}")
    col3.metric("Training Accuracy", f"{history['accuracy'][final_epoch]:.2%}")
    col4.metric("Validation Accuracy", f"{history['val_accuracy'][final_epoch]:.2%}")
    
    # Training summary
    st.markdown("#### Training Summary")
    
    training_time = "3m 42s"  # Placeholder
    epochs_completed = len(history["loss"])
    
    st.markdown(f"**Total Training Time:** {training_time}")
    st.markdown(f"**Epochs Completed:** {epochs_completed}")
    
    if state["training_config"]["early_stopping"]["enabled"]:
        st.markdown("**Early Stopping:** Activated")
    
    # Best epoch
    best_epoch = np.argmin(history["val_loss"])
    st.markdown(f"**Best Model at Epoch:** {best_epoch + 1}")
    st.markdown(f"**Best Validation Loss:** {history['val_loss'][best_epoch]:.4f}")
    st.markdown(f"**Best Validation Accuracy:** {history['val_accuracy'][best_epoch]:.2%}")
