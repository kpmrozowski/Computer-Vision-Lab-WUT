%% Parametry działania
% Powtarzalne wyniki
function [featuresTrain, labelsTrain, featuresTest, labelsTest] = load_data()

% Wielkość słownika
words_cnt = 30 ;

% Detekcja cech
% �?adowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
imds = imageDatastore(".\INRIAPerson\my_train", "IncludeSubfolders", true, "LabelSource", "foldernames");
imtest = imageDatastore(".\INRIAPerson\my_test", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds)

%% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeatureHOGs(I);
    total_features = total_features + length(all_points{i});
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
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

%% Tworzenie słownika

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Wizualizacja wyliczonych słów

%% Wyznaczenie histogramów słów dla każdego obrazu treningowego
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = getFeatureHOGs(I);
end

%% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

featuresTrain = file_hist;
featuresTest = test_hist;
labelsTrain = imds.Labels;
labelsTest = imtest.Labels;

end

function pts = getFeatureHOGs(I)
    if size(I, 3) > 1
        I2 = rgb2gray(I);
    else
        I2 = I;
    end
    
    pts = extractHOGFeatures(I2);
end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end

function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end
