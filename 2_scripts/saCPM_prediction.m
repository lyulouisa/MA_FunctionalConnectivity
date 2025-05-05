
clear all;
addpath('Rosenberg_PNAS2020');
plotting = 0;

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
        load("../31_data/fc/behav_PL_doors.mat");

        load("../1_data/fc/subj_num_amph_doors.mat");
        load("../1_data/fc/subj_num_PL_doors.mat");

    otherwise
        error('Invalid task specified. Choose "rest", "MID", or "doors".');
end
% Identify subjects present in both conditions
[C, ia, ib] = intersect(subj_num_amph, subj_num_PL);
subj_num_amph = subj_num_amph(ia);
subj_num_PL = subj_num_PL(ib);

% Subset the FC and behavior data to include only common subjects
fc_amph_s = fc_amph_s(:, :, ia);
fc_PL_s = fc_PL_s(:, :, ib);
behav_amph = behav_amph(ia);
behav_PL = behav_PL(ib);

% Compute the difference in functional connectivity between conditions
fc_amph_PL = fc_amph_s - fc_PL_s;

% Compute the difference in behavior scores
behav_amph_PL = behav_amph - behav_PL;
nan_idx = isnan(behav_amph_PL);

% Remove NaN rows from behav_amph_PL
behav_amph_PL(nan_idx) = [];

% Remove corresponding 3rd level slices from fc_amph_PL
fc_amph_PL(:, :, nan_idx) = [];

%% saCPM for MA and PL
[r_amph, p_amph, predicted_behav_amph] = apply_sacpm(fc_amph_s, behav_amph, 'Rosenberg_PNAS2020/');
[r_PL, p_PL, predicted_behav_PL] = apply_sacpm(fc_PL_s, behav_PL, 'Rosenberg_PNAS2020/');

%% Change in behavior between session related to change in attentional network?
% Apply saCPM using the computed FC and behavior differences
[r, p, predicted_behav] = apply_sacpm(fc_amph_PL, behav_amph_PL, 'Rosenberg_PNAS2020/');

% Plot results if enabled
if plotting
    figure;
    scatter(behav_amph_PL, predicted_behav);
    xlabel('Observed Behavior (amph - PL)');
    ylabel('Predicted Behavior (amph - PL)');
    title('Observed vs Predicted Behavior (amph - PL)');
end

% Load saCPM results
load(['Rosenberg_PNAS2020/saCPM.mat']);
fc_mats = fc_amph_PL;

% Compute network strength for high and low attention networks
for i = 1:size(fc_mats,3)
    net_strength_amph_PL.high(i,1) = nansum(nansum(high_attention_mask .* fc_mats(:,:,i)));
    net_strength_amph_PL.low(i,1)  = nansum(nansum(low_attention_mask .* fc_mats(:,:,i)));
    predicted_behav(i,1)   = robGLM_fit(1) + robGLM_fit(2) * (net_strength_amph_PL.high(i) - net_strength_amph_PL.low(i));
end