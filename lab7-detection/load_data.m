%% Parametry działania
% Powtarzalne wyniki
function [featuresTrain, labelsTrain, featuresTest, labelsTest] = load_data()

% Wielkość wektora gradientow
gradient_count = 3780 ;

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
imtrain = imageDatastore("INRIAPerson/trainJPG", "IncludeSubfolders", true, "LabelSource", "foldernames");
imtest = imageDatastore("INRIAPerson/testJPG", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds)

%% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
train_size = length(imtrain.Files);
test_size = length(imtest.Files);
FeaturesTrain = zeros(train_size, gradient_count);
FeaturesTest = zeros(test_size, gradient_count);

for i=1:train_size
    I = readImage(imtrain.Files{i});
    [FeaturesTrain(i,:) trash] = extractHOGFeatures(I, 'CellSize',[8 8]);
end
for i=1:test_size
    I = readImage(imtest.Files{i});
    [FeaturesTest(i,:) trash] = extractHOGFeatures(I, 'CellSize',[8 8]);
end

featuresTrain = FeaturesTrain;
featuresTest = FeaturesTest;
labelsTrain = imtrain.Labels;
labelsTest = imtest.Labels;

end



%%
% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,1) ~= 134
        I = imresize(I, [134 NaN]);
        if size(I, 2) ~= 70
            targetSize = [134 70];
            r = centerCropWindow2d(size(I),targetSize);
            I = imcrop(I,r);
        end
    end
end
