[featuresTrain, labelsTrain, featuresTest, labelsTest] = load_data();

%%

model = fitcecoc(featuresTrain, labelsTrain);

predicted_labels = predict(model, featuresTest);
