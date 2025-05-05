%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare Conditions (MA vs. PL) Across Multiple Networks %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% ========================================================================
%  1) Load data, intersect subjects, prepare fc_amph_s and fc_PL_s
%  ========================================================================

task = 'rest';  % <-- Change this to 'MID' or 'doors' as needed

switch task
    case 'rest'
        load("../1_data/fc/fc_amph_rest.mat");
        load("../1_data/fc/fc_PL_rest.mat");

        load("../1_data/fc/behav_amph_rest.mat");
        load("../1_data/fc/behav_PL_rest.mat");

        load("../1_data/fc/subj_num_amph_rest.mat");
        load("../1_data/fc/subj_num_PL_rest.mat");

    case 'MID'
        load("../1_data/fc/fc_amph_MID.mat");
        load("../1_data/fc/fc_PL_MID.mat");

        load("../1_data/fc/behav_amph_MID.mat");
        load("../1_data/fc/behav_PL_MID.mat");

        load("../1_data/fc/subj_num_amph_MID.mat");
        load("../1_data/fc/subj_num_PL_MID.mat");

    case 'doors'
        load("../1_data/fc/fc_amph_doors.mat");
        load("../1_data/fc/fc_PL_doors.mat");

        load("../1_data/fc/behav_amph_doors.mat");
        load("../1_data/fc/behav_PL_doors.mat");

        load("../1_data/fc/subj_num_amph_doors.mat");
        load("../1_data/fc/subj_num_PL_doors.mat");

    otherwise
        error('Invalid task specified. Choose "rest", "MID", or "doors".');
end


% Find subjects that have both Amphetamine & Placebo data
[C, ia, ib] = intersect(subj_num_amph, subj_num_PL);
subj_num_amph = subj_num_amph(ia);
subj_num_PL   = subj_num_PL(ib);

fc_amph_s = fc_amph_s(:, :, ia);
fc_PL_s   = fc_PL_s(:, :, ib);

behav_amph = behav_amph(ia);
behav_PL   = behav_PL(ib);

numParticipants = size(fc_amph_s, 3);

%% ========================================================================
%  2) Compare with High and Low Attention Network (moved up)
%     We'll define net_strength_amph and net_strength_PL
%     and then do the difference for each subject & t-tests
%  ========================================================================
% Load the high_attention_mask and low_attention_mask
% (You mention Rosenberg_PNAS2020/saCPM.mat in your original code)
load('Rosenberg_PNAS2020/saCPM.mat', 'high_attention_mask', 'low_attention_mask');

% Preallocate space
net_strength_amph.high = zeros(numParticipants,1);
net_strength_amph.low  = zeros(numParticipants,1);
net_strength_PL.high   = zeros(numParticipants,1);
net_strength_PL.low    = zeros(numParticipants,1);

for subj = 1:numParticipants
    % Amphetamine
    fc_amph = fc_amph_s(:, :, subj);
    net_strength_amph.high(subj) = nanmean(fc_amph(high_attention_mask > 0));
    net_strength_amph.low(subj)  = nanmean(fc_amph(low_attention_mask  > 0));
    
    % Placebo
    fc_PL = fc_PL_s(:, :, subj);
    net_strength_PL.high(subj) = nanmean(fc_PL(high_attention_mask > 0));
    net_strength_PL.low(subj)  = nanmean(fc_PL(low_attention_mask  > 0));
end

% Differences across subjects
difference_high = net_strength_amph.high - net_strength_PL.high;
difference_low  = net_strength_amph.low  - net_strength_PL.low;

% Perform one-sample t-tests
[~, p_high, ~, stats_high] = ttest(difference_high);
t_high = stats_high.tstat;

[~, p_low, ~, stats_low] = ttest(difference_low);
t_low = stats_low.tstat;

% Create small 2x2 matrices for t-scores and p-values
t_scores_special = NaN(2,2);
p_values_special = NaN(2,2);

% Fill them in the diagonal for a simple display
t_scores_special(1,1) = t_high;   
t_scores_special(2,2) = t_low;    

p_values_special(1,1) = p_high;
p_values_special(2,2) = p_low;

% For visualization, define the custom colormap
color1       = [76, 102, 67] / 255;   % Greenish color #4C6643
color_white  = [1, 1, 1];             % White
color2       = [230, 188, 71] / 255;  % Yellowish color
nColors      = 256; 
midPoint     = round(nColors / 2);
r = [linspace(color1(1), color_white(1), midPoint), linspace(color_white(1), color2(1), midPoint)]';
g = [linspace(color1(2), color_white(2), midPoint), linspace(color_white(2), color2(2), midPoint)]';
b = [linspace(color1(3), color_white(3), midPoint), linspace(color_white(3), color2(3), midPoint)]';
customColormap = [r, g, b];

% Plot the t-scores matrix for the Attention networks
figure;
imagesc(t_scores_special);
colormap(customColormap);
colorbar;
caxis([-12, 10]);  % Adjust color scaling as needed
set(gca, 'XTick', 1:2, ...
         'XTickLabel', {'High Attention','Low Attention'}, ...
         'XTickLabelRotation',45);
set(gca, 'YTick', 1:2, ...
         'YTickLabel', {'High Attention','Low Attention'});
title('T-Scores of Connectivity Differences (MA - Placebo) - Attention Networks', 'FontSize', 16);
axis square;

%% ========================================================================
%  3) Change in MA-PL in Arousal & Valence Networks
%     (Compute "pos - neg" masks for each subject in each condition)
%  ========================================================================

%% ----------------- Arousal Network -----------------
load("a_awake_zscore_0.05.mat","pos_feat","neg_feat");

% Average across the first dimension
pos_feat_avg = squeeze(mean(pos_feat, 1));
neg_feat_avg = squeeze(mean(neg_feat, 1));

% Threshold the features
pos_feat_avg(pos_feat_avg < 1) = 0;
neg_feat_avg(neg_feat_avg < 1) = 0;

% Arousal: Amphetamine and Placebo
arousal_highlow_amph = zeros(numParticipants,1);
arousal_highlow_PL   = zeros(numParticipants,1);
for i = 1:numParticipants
    fc_amph = fc_amph_s(:,:,i);
    fc_PL   = fc_PL_s(:,:,i);

    arousal_highlow_amph(i) = nanmean(nanmean(pos_feat_avg .* fc_amph)) ...
                              - nanmean(nanmean(neg_feat_avg .* fc_amph));
    arousal_highlow_PL(i)   = nanmean(nanmean(pos_feat_avg .* fc_PL)) ...
                              - nanmean(nanmean(neg_feat_avg .* fc_PL));
end
awake_highlow = mean(arousal_highlow_amph - arousal_highlow_PL, 1);  % If you want the group mean

%% ----------------- Valence Network -----------------
load("f_valence_zscore_0.05.mat","pos_feat","neg_feat");

% Average across the first dimension
pos_feat_avg = squeeze(mean(pos_feat, 1));
neg_feat_avg = squeeze(mean(neg_feat, 1));

% Threshold the features
pos_feat_avg(pos_feat_avg < 1) = 0;
neg_feat_avg(neg_feat_avg < 1) = 0;

% Valence: Amphetamine and Placebo
valence_highlow_amph = zeros(numParticipants,1);
valence_highlow_PL   = zeros(numParticipants,1);
for i = 1:numParticipants
    fc_amph = fc_amph_s(:,:,i);
    fc_PL   = fc_PL_s(:,:,i);

    valence_highlow_amph(i) = nanmean(nanmean(pos_feat_avg .* fc_amph)) ...
                              - nanmean(nanmean(neg_feat_avg .* fc_amph));
    valence_highlow_PL(i)   = nanmean(nanmean(pos_feat_avg .* fc_PL)) ...
                              - nanmean(nanmean(neg_feat_avg .* fc_PL));
end
valence_highlow = mean(valence_highlow_amph - valence_highlow_PL, 1);

%% ========================================================================
%  4) Save Behavioral + Network Strengths in Tables
%  ========================================================================

% Amphetamine
subj_behav_network_amph = table( ...
    subj_num_amph,         ...
    behav_amph,            ...
    net_strength_amph.high, ...
    net_strength_amph.low,  ...
    (net_strength_amph.high - net_strength_amph.low), ...
    arousal_highlow_amph,  ... % Arousal
    valence_highlow_amph,  ... % Valence
    'VariableNames', { ...
       'subID', 'behav_amph', ...
       'amph_high', 'amph_low', 'amph_highLow', ...
       'amph_arousal_highLow', 'amph_valence_highLow'});

% Placebo
subj_behav_network_PL = table( ...
    subj_num_PL,           ...
    behav_PL,              ...
    net_strength_PL.high,  ...
    net_strength_PL.low,   ...
    (net_strength_PL.high - net_strength_PL.low), ...
    arousal_highlow_PL,    ... % Arousal
    valence_highlow_PL,    ... % Valence
    'VariableNames', { ...
       'subID', 'behav_PL', ...
       'PL_high', 'PL_low', 'PL_highLow', ...
       'PL_arousal_highLow', 'PL_valence_highLow'});

fname_amph = sprintf('../1_data/subj_network_amph_%s_allNetworks.csv', task);
fname_PL   = sprintf('../1_data/subj_network_PL_%s_allNetworks.csv',   task);

writetable(subj_behav_network_amph, fname_amph);
writetable(subj_behav_network_PL,   fname_PL);

%% ========================================================================
%  5) 8 Canonical Networks: Average within each network, (Amphetamine - PL)
%  ========================================================================

load('shen_network_labels.mat');  % variable Shen_network_labels, 268x1
numNetworks = 8;
networkMasks = zeros(268,268,numNetworks);

for n = 1:numNetworks
    for i = 1:268
        for j = 1:268
            if Shen_network_labels(i) == n && Shen_network_labels(j) == n
                networkMasks(i,j,n) = 1;
            end
        end
    end
end

network_highlow = zeros(numNetworks,1);

for i = 1:numNetworks
    amph_highlow_vec = zeros(numParticipants,1);
    PL_highlow_vec   = zeros(numParticipants,1);

    for subj = 1:numParticipants
        amph_fc = fc_amph_s(:,:,subj);
        pl_fc   = fc_PL_s(:,:,subj);

        amph_highlow_vec(subj) = nanmean(amph_fc(networkMasks(:,:,i) > 0));
        PL_highlow_vec(subj)   = nanmean(pl_fc(networkMasks(:,:,i) > 0));
    end

    % The group-average difference for network i
    network_highlow(i) = mean(amph_highlow_vec - PL_highlow_vec);
end


%% ========================================================================
%  7) Matrix: Canonical 8x8 differences (Amph - PL) and T-tests
%  ========================================================================
numNetworks_canonical = 8;
numParticipants       = size(fc_amph_s, 3);

amph_connectivity_canonical = NaN(numNetworks_canonical, numNetworks_canonical, numParticipants);
PL_connectivity_canonical   = NaN(numNetworks_canonical, numNetworks_canonical, numParticipants);

for subj = 1:numParticipants
    fc_amph = fc_amph_s(:, :, subj);
    fc_PL   = fc_PL_s(:, :, subj);

    for i = 1:numNetworks_canonical
        for j = 1:numNetworks_canonical
            mask_i = (Shen_network_labels == i);
            mask_j = (Shen_network_labels == j);
            between_mask = mask_i * mask_j';  % 268x268

            amph_values = fc_amph(between_mask>0);
            PL_values   = fc_PL(between_mask>0);

            amph_connectivity_canonical(i, j, subj) = nanmean(amph_values);
            PL_connectivity_canonical(i, j, subj)   = nanmean(PL_values);
        end
    end
end

% Compute connectivity differences
connectivity_diff_canonical = amph_connectivity_canonical - PL_connectivity_canonical;

% Average across participants for visualization
mean_diff_canonical = nanmean(connectivity_diff_canonical, 3);  % 8x8 matrix

network_labels_canonical = {
    'Medial Frontal', 'Frontoparietal', 'Default mode', ...
    'Subcortical - cerebellum', 'Motor', 'Visual I', ...
    'Visual II', 'Visual Association'
};

% Plot the average difference matrix
figure;
imagesc(mean_diff_canonical);
colormap(customColormap);
colorbar;
caxis([-0.1, 0.1]); 
set(gca, 'XTick', 1:numNetworks_canonical, ...
         'XTickLabel', network_labels_canonical, ...
         'XTickLabelRotation', 45);
set(gca, 'YTick', 1:numNetworks_canonical, ...
         'YTickLabel', network_labels_canonical);
title('Connectivity Differences (Amphetamine - Placebo) - Canonical Networks', 'FontSize', 16);
axis square;

% Run t-tests on each cell of the 8x8 matrix
t_scores_canonical = NaN(numNetworks_canonical, numNetworks_canonical);
p_values_canonical = NaN(numNetworks_canonical, numNetworks_canonical);

for i = 1:numNetworks_canonical
    for j = 1:numNetworks_canonical
        connectivity_diff = squeeze(connectivity_diff_canonical(i, j, :));
        valid_idx = ~isnan(connectivity_diff);
        connectivity_diff_valid = connectivity_diff(valid_idx);

        [~, p, ~, stats] = ttest(connectivity_diff_valid);
        t_scores_canonical(i, j) = stats.tstat;
        p_values_canonical(i, j) = p;
    end
end

% Correct for multiple comparisons (Bonferroni)
m_canonical = numNetworks_canonical^2;  % 64
p_values_vector_canonical = p_values_canonical(:);
adj_p_canonical = p_values_vector_canonical * m_canonical;
adj_p_canonical(adj_p_canonical > 1) = 1;
adj_p_canonical_matrix = reshape(adj_p_canonical, size(p_values_canonical));

% Plot the T-score matrix
figure;
imagesc(t_scores_canonical);
colormap(customColormap);
colorbar;
caxis([-12, 10]);
set(gca, 'XTick', 1:numNetworks_canonical, ...
         'XTickLabel', network_labels_canonical, ...
         'XTickLabelRotation', 45);
set(gca, 'YTick', 1:numNetworks_canonical, ...
         'YTickLabel', network_labels_canonical);
title('T-Scores of Connectivity Differences (MA - Placebo) - Canonical Networks', 'FontSize', 16);
axis square;
