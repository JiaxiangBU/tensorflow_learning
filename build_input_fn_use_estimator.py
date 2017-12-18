##Custom Input Pipelines with input_fn
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#        filename=IRIS_TRAINING,
#        target_dtype=np.int,
#        features_dtype=np.float32)

##Convert feature data to tensors
#train_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": np.array(training_set.data)},
#        y=np.array(training_set.target),
#        num_epochs=None,
#        shuffle=True)

#classifier.train(input_fn=train_input_fn, steps=2000)

##Anatomy of an input_fn
#def my_input_fn():
    ##preprocess your data here
    
    
    ##...then return 1) a mapping of feature columns to Tensors with
    ##the corresponding festure data, and 2) a Tensor containing labels
#    return feature_cols, labels
    