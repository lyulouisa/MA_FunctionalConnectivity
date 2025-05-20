% Clear workspace and add paths as needed
clear all

% -------------------------------------------------------------------------
% Load data
% -------------------------------------------------------------------------
load('Rosenberg_PNAS2020/saCPM.mat')
load('a_awake_zscore_0.05.mat')

% Average across first dimension
pos_feat_avg = squeeze(mean(pos_feat, 1));
neg_feat_avg = squeeze(mean(neg_feat, 1));

% Threshold the features
pos_feat_avg(pos_feat_avg < 1) = 0;
neg_feat_avg(neg_feat_avg < 1) = 0;

% Load network difference data
load('../3_results/stats/network_diff_positive_rest.txt')
load('../3_results/stats/network_diff_negative_rest.txt')
% For consistency, rename or store them as:
network_diff_pos = network_diff_positive_rest;  
network_diff_neg = network_diff_negative_rest;

% -------------------------------------------------------------------------
% Define the lower-triangle indices (exclude diagonal)
% -------------------------------------------------------------------------
nROI = 268;
lowerTriInd = find(tril(true(nROI), -1));  % indices of the lower triangle

% -------------------------------------------------------------------------
% Flatten each mask into a vector of edges in the lower triangle
% -------------------------------------------------------------------------
pos_feat_vec      = pos_feat_avg(lowerTriInd);
neg_feat_vec      = neg_feat_avg(lowerTriInd);
net_diff_pos_vec  = network_diff_pos(lowerTriInd);
net_diff_neg_vec  = network_diff_neg(lowerTriInd);

% -------------------------------------------------------------------------
% Hypergeometric test function
% 
% M = total number of edges (here, the number of elements in lower triangle)
% K = number of 'success' edges in first mask
% N = number of 'drawn' edges in second mask
% x = overlap (how many edges are 1 in both masks)
% 
% p-value = 1 - hygecdf(x-1, M, K, N)
% -------------------------------------------------------------------------
hypgeo_test = @(x,M,K,N) 1 - hygecdf(x-1, M, K, N);

% -------------------------------------------------------------------------
% Define total number of edges: M = 268*267/2 = 35778
% (For good measure, it should match length(lowerTriInd).)
% -------------------------------------------------------------------------
M = length(lowerTriInd);  % should be 35778

%% overlap with arousal network
% -------------------------------------------------------------------------
%  A) Arousal vs. Positive/Negative Edges
% -------------------------------------------------------------------------

% -- 1) High Arousal vs. Positive Edges
K_pos          = sum(pos_feat_vec == 1);        % # edges in "high arousal" mask
N_posDiff      = sum(net_diff_pos_vec == 1);    % # edges in "positive edges"
x_pos_posDiff  = sum(pos_feat_vec == 1 & net_diff_pos_vec == 1);
p_val_pos_posDiff = hypgeo_test(x_pos_posDiff, M, K_pos, N_posDiff);
fprintf('--- High Arousal vs. Positive Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_pos, N_posDiff, x_pos_posDiff, p_val_pos_posDiff);

% -- 2) Low Arousal vs. Negative Edges
K_neg          = sum(neg_feat_vec == 1);        % # edges in "low arousal" mask
N_negDiff      = sum(net_diff_neg_vec == 1);    % # edges in "negative edges"
x_neg_negDiff  = sum(neg_feat_vec == 1 & net_diff_neg_vec == 1);
p_val_neg_negDiff = hypgeo_test(x_neg_negDiff, M, K_neg, N_negDiff);
fprintf('--- Low Arousal vs. Negative Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_neg, N_negDiff, x_neg_negDiff, p_val_neg_negDiff);

% -- 3) High Arousal vs. Negative Edges
x_pos_negDiff  = sum(pos_feat_vec == 1 & net_diff_neg_vec == 1);
p_val_pos_negDiff = hypgeo_test(x_pos_negDiff, M, K_pos, N_negDiff);
fprintf('--- High Arousal vs. Negative Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_pos, N_negDiff, x_pos_negDiff, p_val_pos_negDiff);

% -- 4) Low Arousal vs. Positive Edges
x_neg_posDiff  = sum(neg_feat_vec == 1 & net_diff_pos_vec == 1);
p_val_neg_posDiff = hypgeo_test(x_neg_posDiff, M, K_neg, N_posDiff);
fprintf('--- Low Arousal vs. Positive Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_neg, N_posDiff, x_neg_posDiff, p_val_neg_posDiff);

%% overlap with attentional network
% Flatten the high/low attention masks to the lower triangle
high_att_vec = high_attention_mask(lowerTriInd);
low_att_vec  = low_attention_mask(lowerTriInd);

% -- 5) High Attention vs. Positive Edges
K_high         = sum(high_att_vec == 1);        % # edges in "high attention" mask
N_posDiff      = sum(net_diff_pos_vec == 1);    % # edges in "positive edges" (already computed)
x_high_posDiff = sum(high_att_vec == 1 & net_diff_pos_vec == 1);
p_val_high_posDiff = hypgeo_test(x_high_posDiff, M, K_high, N_posDiff);
fprintf('--- High Attention vs. Positive Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_high, N_posDiff, x_high_posDiff, p_val_high_posDiff);

% -- 6) Low Attention vs. Negative Edges
K_low         = sum(low_att_vec == 1);          % # edges in "low attention" mask
N_negDiff     = sum(net_diff_neg_vec == 1);     % # edges in "negative edges" (already computed)
x_low_negDiff = sum(low_att_vec == 1 & net_diff_neg_vec == 1);
p_val_low_negDiff = hypgeo_test(x_low_negDiff, M, K_low, N_negDiff);
fprintf('--- Low Attention vs. Negative Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_low, N_negDiff, x_low_negDiff, p_val_low_negDiff);

% -- 7) High Attention vs. Negative Edges
x_high_negDiff = sum(high_att_vec == 1 & net_diff_neg_vec == 1);
p_val_high_negDiff = hypgeo_test(x_high_negDiff, M, K_high, N_negDiff);
fprintf('--- High Attention vs. Negative Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_high, N_negDiff, x_high_negDiff, p_val_high_negDiff);

% -- 8) Low Attention vs. Positive Edges
x_low_posDiff  = sum(low_att_vec == 1 & net_diff_pos_vec == 1);
p_val_low_posDiff = hypgeo_test(x_low_posDiff, M, K_low, N_posDiff);
fprintf('--- Low Attention vs. Positive Edges ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n\n', ...
    M, K_low, N_posDiff, x_low_posDiff, p_val_low_posDiff)


%% overlap between arousal and attention networks
% -------------------------------------------------------------------------
% Overlap: arousal vs. attention
%   1) High vs. High
%   2) Low vs. Low
%   3) Combined vs. Combined
% -------------------------------------------------------------------------
posneg_feat_vec = (pos_feat_vec == 1 | neg_feat_vec == 1);
posneg_diff_vec = (net_diff_pos_vec == 1 | net_diff_neg_vec == 1);

combined_att_vec = (high_att_vec == 1 | low_att_vec == 1);

% -- 1) Overlap: pos_feat_vec (high arousal) vs high_att_vec (high attention) --
K_pos_arousal = sum(pos_feat_vec == 1);     % # edges in high-arousal mask
N_high_att    = sum(high_att_vec == 1);     % # edges in high-attention mask
x_pos_high    = sum(pos_feat_vec == 1 & high_att_vec == 1);  % overlap

p_val_pos_high = hypgeo_test(x_pos_high, M, K_pos_arousal, N_high_att);

fprintf('\n--- Overlap: High Arousal vs. High Attention ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n', ...
    M, K_pos_arousal, N_high_att, x_pos_high, p_val_pos_high);

% -- 2) Overlap: neg_feat_vec (low arousal) vs low_att_vec (low attention) --
K_neg_arousal = sum(neg_feat_vec == 1);     % # edges in low-arousal mask
N_low_att     = sum(low_att_vec == 1);      % # edges in low-attention mask
x_neg_low     = sum(neg_feat_vec == 1 & low_att_vec == 1);   % overlap

p_val_neg_low = hypgeo_test(x_neg_low, M, K_neg_arousal, N_low_att);

fprintf('\n--- Overlap: Low Arousal vs. Low Attention ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n', ...
    M, K_neg_arousal, N_low_att, x_neg_low, p_val_neg_low);

% -- 3) Overlap: combined arousal (pos|neg) vs. combined attention (high|low) --
att_combined_vec = (high_att_vec == 1 | low_att_vec == 1);

K_posneg_arousal  = sum(posneg_feat_vec == 1);        
N_posneg_att      = sum(att_combined_vec == 1);       
x_posneg_combined = sum(posneg_feat_vec == 1 & att_combined_vec == 1);

p_val_posneg_combined = hypgeo_test(x_posneg_combined, M, ...
                                    K_posneg_arousal, N_posneg_att);

fprintf('\n--- Overlap: Combined Arousal (pos+neg) vs. Combined Attention (high+low) ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n', ...
    M, K_posneg_arousal, N_posneg_att, x_posneg_combined, p_val_posneg_combined);


%% overlap with valence network
load("f_valence_zscore_0.05.mat")
% average across the first dimension
pos_feat_avg = squeeze(mean(pos_feat, 1));
neg_feat_avg = squeeze(mean(neg_feat, 1));

% threshold the features
pos_feat_avg(pos_feat_avg < 1) = 0;
neg_feat_avg(neg_feat_avg < 1) = 0;

valence_pos_vec = pos_feat_avg(lowerTriInd);  % pos_feat_avg from your valence file
valence_neg_vec = neg_feat_avg(lowerTriInd);  % neg_feat_avg from your valence file

% 2) Combine them into one overall valence mask
valence_combined_vec = (valence_pos_vec == 1 | valence_neg_vec == 1);

% 3) Count how many edges are in the combined valence mask
K_valence_combined = sum(valence_combined_vec);

% 4) Combined attention mask (already defined as high OR low)
N_att = sum(combined_att_vec == 1);

% 5) Overlap count between valence and attention
x_val_att = sum(valence_combined_vec & combined_att_vec);

% 6) Hypergeometric test
%    M = total edges in lower triangle (already set to length(lowerTriInd) = 35778)
p_val_val_att = hypgeo_test(x_val_att, M, K_valence_combined, N_att);

% 7) Print results
fprintf('\n--- Overlap: Combined Valence (pos+neg) vs. Combined Attention (high+low) ---\n');
fprintf('M = %d, K = %d, N = %d, x = %d, p-value = %.6g\n', ...
    M, K_valence_combined, N_att, x_val_att, p_val_val_att);
