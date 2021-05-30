%% Parametry działania
% Powtarzalne wyniki
function [featuresTrain, labelsTrain, featuresTest, labelsTest] = load_data()

% Wielkość słownika
gradient_count = 30 ;

% Detekcja cech
% �?adowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
imds = imageDatastore("../INRIAPerson/96X160H96/Train/pos", "IncludeSubfolders", true, "LabelSource", "foldernames");
imtest = imageDatastore("../INRIAPerson/70X134H96/Test/pos", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds)

%% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = zeros(files_cnt, 3780);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    [all_points(i,:) trash] = extractHOGFeatures(I, 'CellSize',[8 8]);
    total_features = total_features + length(all_points(i, 2));
end

%% Przygotowanie listy przechowującej indeksy plików i punktów charakterystycznych
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

%% Obliczenie deskryptorów punktów charakterystycznych
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractHOGFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

%%

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, gradient_count, "MaxIter", 10000);
% Wizualizacja wyliczonych gradientów

%% Wyznaczenie cech
file_features_hog = zeros(files_cnt, gradient_count);
for i=1:files_cnt
    file_features_hog(i,:) = getFeatureHOGs(I);
end

%% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), gradient_count);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractHOGFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

featuresTrain = file_features_hog;
featuresTest = test_hist;
labelsTrain = imds.Labels;
labelsTest = imtest.Labels;

end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,1) ~= 134
        I = imresize(I, [134 NaN]);
        if size(I, 2) ~= 70
            targetSize = [134 70];
            r = centerCropWindow2d(size(I),targetSize);
            I = imcrop(I,r);
            %half_of_width = size(I, 2) / 2;
            %x1 = half_of_width - 70/2;
            %y1 = 0;
            %width = 70;
            %height = 134;
            %I = imcrop(I, [x1 y1 width height]);
        end
    end
end

function h = wordHist(feats, words)
    gradient_count = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:gradient_count+1)-0.5, 'Normalization', 'probability');
end
