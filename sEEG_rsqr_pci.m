%addpath D:\MEMOIRE_PE\fonctions_Matlab\fieldtrip-20230118\fieldtrip-20230118
addpath D:\MEMOIRE_PE\fonctions_Matlab\fieldtrip-20230118\fieldtrip-20230118
ft_defaults;

% % Patient worspace
out_path = 'C:\Users\nadege\Desktop\camille\seeg\agnes\1000perm\San_Ray\laure_reg_onsets_accou_align';
% out_path = 'C:\Users\nadege\Desktop\camille\seeg\agnes\1000perm\San_Ray\laure_reg_onsets_accou';
% % SonsPurs file 
%SonsPurs_path =
%'\\139.124.148.123\seeg\Seeg_1998-2024\Seeg2024\24.01_seeg02_Blu_Ro\17.01\SonsPurs_BaPa.vhdr';

fig_path = [out_path '\figs'];
if exist(fig_path, 'dir') ~= 7
    mkdir(fig_path);
end

threshold = 0.01

%% === Sorcière : acoustic ===
% Load r2
dat_file = '\sorciere_acoustic.mat';
sorciere_acou = load([out_path '\' dat_file]);

channels = cellstr(sorciere_acou.channels);

% Convertir r² en %
sorciere_acou.r2_hr = sorciere_acou.r2 * 100;
sorciere_acou.r2_hr(sorciere_acou.r2_hr < 0) = 0;

% Layout
cfg = [];
cfg.dataset = sorciere_acou.r2;
cfg.label   = channels;
layout_bp = sEEG_layout(cfg);
layout_bp.width  = layout_bp.width * 0.9;
layout_bp.height = layout_bp.height * 0.9;

% Structure TRF
trf = [];
trf.powspctrm(:,1,:) = repmat(sorciere_acou.r2_hr, 2, 1)';
trf.powspctrm(:,2,:) = repmat(sorciere_acou.r2_hr, 2, 1)';
trf.label   = channels;
trf.freq    = linspace(0,1,2);
trf.time    = linspace(0,1,2);
trf.dimord  = 'chan_freq_time';

% Plot
fig = figure;
cfg = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp;
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim        = [0, max(sorciere_acou.r2_hr(:))];  % zlim harmonisé
cfg.title       = 'Acoustic regions (speech)';
cfg.comment     = 'no';
cfg.box         = 'yes';
cfg.fontsize    = 8;

ft_multiplotTFR(cfg, trf);
colorbar;
title(colorbar, '% de variance expliquée');

set(fig, 'Position', get(0, 'ScreenSize'));
saveas(fig, [fig_path '\speech_acoustic_region_spy.png'], 'png');

%% === Sorcière : predictive ===

% Load predictive mat
dat_file = '\sorciere_predictive.mat';
sorciere_pred = load([out_path dat_file]);

% p-valeurs + calcul du gain
pvals = sorciere_pred.pval;
predictive_gain = sorciere_pred.r2_real * 100 - sorciere_acou.r2 * 100;

% Masquage : p >= 0.01 ou valeurs trop faibles
predictive_gain(pvals >= threshold) = 0;
predictive_gain(predictive_gain < 1e-4) = 0;

zvals = predictive_gain(predictive_gain > 0);
if isempty(zvals)
    warning('Aucune valeur significative pour predictive_gain. Forçage de zlim à [0 1e-6].');
    zlim_range = [0 1e-6];  % petite valeur positive pour éviter [0 0]
else
    zlim_range = [0 prctile(zvals, 95)];
end


% TRF structure
trf_p = [];
trf_p.powspctrm(:,1,:) = repmat(predictive_gain,2,1)';
trf_p.powspctrm(:,2,:) = repmat(predictive_gain,2,1)';
trf_p.label   = channels;
trf_p.freq    = linspace(0,1,2);
trf_p.time    = linspace(0,1,2);
trf_p.dimord  = 'chan_freq_time';

% Plot
fig = figure;
cfg = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp;
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim = zlim_range;
cfg.title       = 'Predictive regions (speech)';
cfg.comment     = 'no';
cfg.box         = 'yes';
cfg.fontsize    = 8;

ft_multiplotTFR(cfg, trf_p);

% Colormap modifiée pour griser les valeurs à zéro
my_cmap = parula;
my_cmap(1,:) = [0.5 0.5 0.5];  % gris pour valeurs nulles
colormap(my_cmap);
set(gcf, 'Colormap', my_cmap);

colorbar;
title(colorbar, '% de variance expliquée');

set(fig, 'Position', get(0, 'ScreenSize'));
saveas(fig, [fig_path '\speech_predictive_region.png'], 'png');

% %%
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%% ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % define a structure for plotting kernels w/ fieldtrip
% krnl_acou                  = [];
% krnl_acou.label            = channels;
% krnl_acou.freq             = linspace(-0.3, 0.7, 100)
% krnl_acou.dimord           = 'chan_freq'
% 
% % initialize kernel w/ full of zeros 
% krnl_acou.powspctrm = zeros(numel(channels), 100);
% 
% for i = 1:numel(channels)
% 
%     % Load kernels
%     kernel_acou = squeeze(sorciere_acou.kernels(:,:,i));
% 
%     % Calcul the sum of squared 
%     kernel_acou = zscore(sum(kernel_acou.^2, 2));
% 
%     % if r² > 0 : save kernel, else : keep zeros
%     if sorciere_acou.r2(i) > 0
%         krnl_acou.powspctrm(i,:) = kernel_acou
% 
%     end
% end
% 
% %%%%%% PREDICTIF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % define a structure for plotting kernels w/ fieldtrip
% krnl_pred                  = [];
% krnl_pred.label            = channels;
% krnl_pred.freq             = linspace(-0.3, 0.7, 100)
% krnl_pred.dimord           = 'chan_freq'
% 
% % initialize kernel w/ full of zeros 
% krnl_pred.powspctrm = zeros(numel(channels), 100);
% 
% for i = 1:numel(channels)
% 
%     % Load kernels
%     kernel_pred =squeeze(sorciere_pred.kernels(:, end-1:end, i));
% 
%     % Calcul the sum of squared 
%     kernel_pred = zscore(sum(kernel_pred.^2, 2));
% 
%     % if gain > 0 : save kernel, else : keep zeros
%     gain = sorciere_pred.r2_real(i) - sorciere_acou.r2(i)
% 
%     if gain > 0
%         krnl_pred.powspctrm(i,:) = kernel_pred
% 
%     end
% end
% 
% % plot kernels
% fig = figure();
% cfg              = [];
% cfg.layout       = layout_bp; % 'ordered'
% cfg.channel      = 'all';
% cfg.interactive  = 'yes';
% cfg.showoutline  = 'yes';
% cfg.showlabels   = 'yes';
% cfg.title        = 'Décours temporel acoustic du processing du language : acoustic (bleu) / predicif (rouge)';
% cfg.comment      = 'no';
% cfg.linecolor    = 'br';
% cfg.comment = 'no';
% cfg.fontsize = 8;
% % Ajouter la légende
% 
% ft_multiplotER(cfg, krnl_acou, krnl_pred);
% % Définir la position de la figure pour qu'elle prenne tout l'écran
% screenSize = get(0, 'ScreenSize');
% set(fig, 'Position', screenSize);
% 
% saveas(fig, [fig_path '/DécoursTemporel_speech.fig'], 'fig');
% 
% %%
% 
% % % % Pianos acoustic r2
% % % Load r2
% dat_file = '\pianos_acoustic.mat'
% pianos_acou = load([out_path dat_file]);
% 
% pianos_acou.r2_hr = pianos_acou.r2*100
% pianos_acou.r2_hr(pianos_acou.r2_hr < 0) = 0;
% 
% % % Plot TRF result
% trf                     = [];
% trf.powspctrm(:,1,:)    =  repmat(pianos_acou.r2_hr,2,1)';
% trf.powspctrm(:,2,:)    =  repmat(pianos_acou.r2_hr,2,1)';
% 
% trf.label               = channels;
% trf.freq                = linspace(0,1,2);
% trf.time                = linspace(0,1,2);
% trf.dimord              = 'chan_freq_time'
% 
% fig = figure();
% cfg             = [];
% cfg.channel     = 'all';
% cfg.layout      = layout_bp; % 'ordered'
% cfg.interactive = 'yes';
% cfg.showoutline = 'yes';
% cfg.showlabels  = 'yes';
% cfg.xlim        = [0,1];
% cfg.ylim        = [0,1];
% cfg.zlim        = [prctile(pianos_acou.r2_hr, 5), prctile(pianos_acou.r2_hr,95)];
% cfg.title       = 'Acoustic regions (music)';
% cfg.comment = 'no';
% cfg.box ='yes';
% cfg.fontsize = 8;
% 
% ft_multiplotTFR(cfg, trf);
% colorbar; 
% cbar_title = title(colorbar, '% de variance expliquée');
% 
% % Définir la position de la figure pour qu'elle prenne tout l'écran
% screenSize = get(0, 'ScreenSize');
% set(fig, 'Position', screenSize);
% saveas(fig, [out_path '/figs' '/music_acoustic_region.png'], 'png');
% 
% 
% % Pianos predictive gain 
% % % Load r2
% dat_file = '\pianos_predictive.mat'
% pianos_pred = load([out_path dat_file]);
% 
% % Calcul du gain prédictif et masquage
% pvals = pianos_pred.pval;
% predictive_gain = pianos_pred.r2_real * 100 - pianos_acou.r2 * 100;
% predictive_gain(pvals >= threshold) = 0;
% % predictive_gain(predictive_gain < 1e-4) = 0;
% 
% % TRF structure
% trf_p = [];
% trf_p.powspctrm(:,1,:) = repmat(predictive_gain,2,1)';
% trf_p.powspctrm(:,2,:) = repmat(predictive_gain,2,1)';
% trf_p.label   = channels;
% trf_p.freq    = linspace(0,1,2);
% trf_p.time    = linspace(0,1,2);
% trf_p.dimord  = 'chan_freq_time';
% 
% % Plot
% fig = figure();
% cfg = [];
% cfg.channel     = 'all';
% cfg.layout      = layout_bp;
% cfg.interactive = 'yes';
% cfg.showoutline = 'yes';
% cfg.showlabels  = 'yes';
% cfg.xlim        = [0,1];
% cfg.ylim        = [0,1];
% cfg.zlim        = [0, prctile(predictive_gain(predictive_gain > 0), 95)];
% cfg.title       = 'Predictive regions (music)';
% cfg.comment     = 'no';
% cfg.box         = 'yes';
% cfg.fontsize    = 8;
% 
% ft_multiplotTFR(cfg, trf_p);
% 
% % Colormap grise pour valeurs nulles (électrodes NS)
% my_cmap = parula;
% my_cmap(1,:) = [0.5 0.5 0.5];
% colormap(my_cmap);
% set(gcf, 'Colormap', my_cmap);
% 
% colorbar;
% title(colorbar, '% de variance expliquée');
% 
% set(fig, 'Position', get(0, 'ScreenSize'));
% saveas(fig, [out_path '/figs/music_predictive_region.png'], 'png');
% 
% %%
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%% ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % define a structure for plotting kernels w/ fieldtrip
% krnl_acou                  = [];
% krnl_acou.label            = channels;
% krnl_acou.freq             = linspace(-0.3, 0.7, 100)
% krnl_acou.dimord           = 'chan_freq'
% 
% % initialize kernel w/ full of zeros 
% krnl_acou.powspctrm = zeros(numel(channels), 100);
% 
% for i = 1:numel(channels)
% 
%     % Load kernels
%     kernel_acou = squeeze(zscore(pianos_acou.kernels(:,:,i)));
% 
%     % Calcul the sum of squared 
%     kernel_acou = sum(kernel_acou.^2, 2);
% 
%     % if r² > 0 : save kernel, else : keep zeros
%     if pianos_acou.r2(i) > 0
%         krnl_acou.powspctrm(i,:) = kernel_acou
% 
%     end
% end
% 
% 
% %%%%%% PREDICTIF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % define a structure for plotting kernels w/ fieldtrip
% krnl_pred                  = [];
% krnl_pred.label            = channels;
% krnl_pred.freq             = linspace(-0.3, 0.7, 100)
% krnl_pred.dimord           = 'chan_freq'
% 
% % initialize kernel w/ full of zeros 
% krnl_pred.powspctrm = zeros(numel(channels), 100);
% 
% for i = 1:numel(channels)
% 
%     % Load kernels
%     kernel_pred =squeeze(zscore(pianos_pred.kernels(:, end-1:end, i)));
% 
%     % Calcul the sum of squared 
%     kernel_pred = sum(kernel_pred.^2, 2);
% 
%     % if gain > 0 : save kernel, else : keep zeros
%     gain = pianos_pred.r2_real(i) - pianos_acou.r2(i)
% 
%     if gain > 0
%         krnl_pred.powspctrm(i,:) = kernel_pred
% 
%     end
% end
% 
% % plot kernels
% fig = figure();
% cfg              = [];
% cfg.layout       = layout_bp; % 'ordered'
% cfg.channel      = 'all';
% cfg.interactive  = 'yes';
% cfg.showoutline  = 'yes';
% cfg.showlabels   = 'yes';
% cfg.title        = 'Décours temporel acoustic du processing de la musique : acoustic (bleu) / predicif (rouge)';
% cfg.comment      = 'no';
% cfg.linecolor    = 'br';
% cfg.comment = 'no';
% cfg.fontsize = 8;
% 
% ft_multiplotER(cfg, krnl_acou, krnl_pred);
% 
% % Définir la position de la figure pour qu'elle prenne tout l'écran
% screenSize = get(0, 'ScreenSize');
% set(fig, 'Position', screenSize);
% 
% saveas(fig, [fig_path '/DécoursTemporel_music.fig'], 'fig');

