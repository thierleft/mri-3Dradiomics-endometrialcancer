%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Analyze ICC for each radiomics feature and each preprocessing
%   parameters set.
%   Select most stable set of preprocessing parameters and features to be
%   excluded from further modeling.
%
%   Not for clinical use.
%   SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2021 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2021 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
warning off;

%   INPUT == ICC FOR EACH PREPROCESSING COMBINATION AND SELECTED PREPROCESSING PARAMS selectVoxelSize selectBinWidth
%   OUTPUT == FIGURES (ICC HEATMAPS 5.4 AND BOXPLOT 5.5 IN THESIS) AND FEATURES TO EXCLUDE namersonRealout

modality = 'MRI_SEQUENCE'; % insert name of MRI sequence of interest (e.g. DCE2, ADC, DWI, etc.)

% Load feature names list
load(['MYPROJECTFILEPATH/SAVE/' modality '/featureNames.mat'])

% Reformat names list
namerson=[];
for sizing = 1:size(names,1)
    namerson{sizing} = strtrim(names(sizing,:));
end

indexing = 0;
indexingout = 0;

mypathsave = ['MYPROJECTFILEPATH/SAVE/' modality '/'];

% Isotropic voxel sizes resampling, same as in PreprocessingRadiomicsExtractions1.py
voxelsizeList = [5, 1, 2, 3];
% Bin width sizes, same as in PreprocessingRadiomicsExtractions1.py
binwidthList = [15,20,25,30];

% This is to identify features to exclude when the set of
% preprocessing steps leading to the highest reproducibility is
% identified (e.g. here it was voxelsize = 1 and binWidth = 30)
selectVoxelSize = 1;
selectBinWidth = 30;

% Number of features per radiomics features class
numClasses = [14,18,24,14,16,16,4];
labels = {'Shape','First Order','GLCM','GLDM','GLRLM','GLSZM','NGTDM'};
% Shape [1,  14] - 14
% First [15, 32] - 18
% GLCM  [33, 56] - 24
% GLDM  [57, 70] - 14
% GLRLM [71, 86] - 16
% GLSZM [87,102] - 16
% NGTDM [103,106]- 4

listdir = dir(mypathsave);
listdir(1) = []; % Always two empty lines in file list when using dir (removed)
listdir(1) = [];
[listSize, J] = size(listdir);

% % % % % % %
%
%  ICCscoresSize = featClasses x binwidth x voxelsizes
%
% % % % % % %
scores = zeros(length(voxelsizeList),length(binwidthList),sum(numClasses)); % featClassesSize = 106
scorethresh = 0.8;

for iterVoxel =1:length(voxelsizeList)  
    for iterBinwidth = 1:length(binwidthList)
        disp(['Voxel = ',num2str(voxelsizeList(iterVoxel)),' and Bin Width = ',num2str(binwidthList(iterBinwidth))])
        
        % Load ICC list for a given set of preprocessing parameters
        % Here False is used for when no normalization was applied
        load([mypathsave,'RadFtsTrue',num2str(voxelsizeList(iterVoxel)),num2str(binwidthList(iterBinwidth)),'.mat'])
        
        % Store ICC for each radiomics feature according to the set of
        % preprocessing parameters in this loop
        scores(iterVoxel,iterBinwidth,:)=icc;
        
        % This is to identify features to exclude when the set of
        % preprocessing steps leading to the highest reproducibility is
        % identified (e.g. here it was voxelsize = 1 and binWidth = 30)
        if voxelsizeList(iterVoxel)==selectVoxelSize && binwidthList(iterBinwidth)==selectBinWidth
            
            scoresrank = icc>scorethresh;
            scoresout  = icc<scorethresh;
            for goin =1:length(icc)
                if scoresrank(goin) == 1
                    indexing = indexing+1;
                    namersonout{indexing} = ['original_' namerson{goin}];
                end
                if scoresout(goin) == 1
                    indexingout = indexingout+1;
                    namersonRealout{indexingout} = ['original_' namerson{goin}];
                end
                
            end
        end
        
        
    end
end

% Save excluded features for the selected set of preprocessing parameters!
save(['namersonRealout' modality '.mat'],'namersonRealout')

for iterVoxel =1:length(voxelsizeList)
    for iterBinwidth = 1:length(binwidthList)
        
        reconplot = [];
        xPlot = [];
        ii = 1;
        for iterClass = 1:length(numClasses)
            
            for num=1:numClasses(iterClass)
                
                if contains(labels{iterClass},'Shape')
                    iternum = num;
                else
                    iternum = numClasses(iterClass-1)+num;
                end

                reconplot= [reconplot scores(iterVoxel,iterBinwidth,iternum)];
                xPlot = [xPlot ii];
                
            end
            
            ii = ii+1;
            
        end
        
        if voxelsizeList(iterVoxel)==5
            voxelSize = 0.5;
        else
            voxelSize = voxelsizeList(iterVoxel);
        end
        
        % This is to identify features to exclude when the set of
        % preprocessing steps leading to the highest reproducibility is
        % identified (e.g. here it was voxelsize = 1 and binWidth = 30)
        if voxelSize == selectVoxelSize && binwidthList(iterBinwidth) == selectBinWidth
            figfig = figure;
            yl = yline(scorethresh,'--','LineWidth',1.5); hold on;
            boxplot(reconplot, xPlot,'Labels',labels); hold on;
            title('ICC')
            xlabel('Category');
            ylabel('ICC')
            ylim([0.3,1.05])
            ax = gca;
            ax.YAxis.TickLabelFormat = '%,.2f';         
        end
    end
end


%% Heatmaps of ICCs to identify which combination of features lead to highest reproducibility
% Inspect which combination lead to highest mean ICC with smallest distribution
% of ICC(std ICC)
reconmean = squeeze(mean(scores,3));

yvalues = {0.5, 1, 2, 3};
xvalues = {15,20,25,30};

heat1 = figure;
h = heatmap(xvalues,yvalues,reconmean);
h.XLabel = 'Bin width';
h.YLabel = 'Voxel size';
h.Title = ['Mean ICC for ' modality];
h.ColorMethod = 'median';
h.GridVisible = 'off';
h.FontSize = 14;


reconstd = squeeze(std(scores,0,3));
heat2 = figure;
h1 = heatmap(xvalues,yvalues,reconstd);
h1.XLabel = 'Bin width';
h1.YLabel = 'Voxel size';
h1.Title = ['ICC standard deviation for ' modality];
h1.ColorMethod = 'median';
h1.GridVisible = 'off';
h1.FontSize = 14;

scoresrank = scores>scorethresh;
reconsum = squeeze(sum(scoresrank,3));

heat3 = figure;
h2 = heatmap(xvalues,yvalues,reconsum);
h2.XLabel = 'Bin width';
h2.YLabel = 'Voxel size';
h2.Title = ['Included features for ' modality];
h2.ColorMethod = 'median';
h2.GridVisible = 'off';
h2.FontSize = 14;
