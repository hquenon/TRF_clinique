%addpath D:\MEMOIRE_PE\fonctions_Matlab\fieldtrip-20230118\fieldtrip-20230118
addpath D:\MEMOIRE_PE\fonctions_Matlab\fieldtrip-20230118\fieldtrip-20230118
ft_defaults;

% % Patient worspace
%out_path = '\\dynaserv\home\trebuchon\Desktop\Python_code\Ma_Gui';
% % SonsPurs file 
%SonsPurs_path =
%'\\139.124.148.123\seeg\Seeg_1998-2024\Seeg2024\24.01_seeg02_Blu_Ro\17.01\SonsPurs_BaPa.vhdr';

fig_path = [out_path '\figs'];
if exist(fig_path, 'dir') ~= 7
    mkdir(fig_path);
end

%%
% % % Sorciere acoustic 

% % Load r2
dat_file = '\sorciere_acoustic.mat';
sorciere_acou = load([out_path '\' dat_file]);

channels = cellstr(sorciere_acou.channels);

sorciere_acou.r2_hr = sorciere_acou.r2*100;
sorciere_acou.r2_hr(sorciere_acou.r2_hr < 0) = 0;

% % import data & def layout
cfg            = [];
cfg.dataset    = sorciere_acou.r2; 
cfg.label    =  channels;

layout_bp = sEEG_layout(cfg);


% % Plot TRF result : 
trf                     = [];
trf.powspctrm(:,1,:)    =  repmat(sorciere_acou.r2_hr,2,1)';
trf.powspctrm(:,2,:)    =  repmat(sorciere_acou.r2_hr,2,1)';

trf.label               = channels;
trf.freq                = linspace(0,1,2);
trf.time                = linspace(0,1,2);
trf.dimord              = 'chan_freq_time';

layout_bp.width = layout_bp.width * 0.9;
layout_bp.height = layout_bp.height * 0.9;

fig = figure;
cfg             = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp; % 'ordered'
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim        = [0, prctile(sorciere_acou.r2_hr,95)];
cfg.title       = 'Acoustic regions (speech)';

cfg.comment = 'no';
cfg.box ='yes';
cfg.fontsize = 8;
% cfg.fontweight = 100;

ft_multiplotTFR(cfg, trf);
colorbar;
cbar_title = title(colorbar, '% de variance expliquée');

% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);

saveas(fig, [fig_path '\speech_acoustic_region_spy.png'], 'png');

%%
% % % Sorciere predictive gain 

% % Load r2
dat_file = '\sorciere_predictive.mat'
sorciere_pred = load([out_path dat_file]);

pvals = sorciere_pred.pval;  
predictive_gain   =  sorciere_pred.r2_real*100 - sorciere_acou.r2*100 ;

% % Plot TRF result
trf_p                     = [];
trf_p.powspctrm(:,1,:) = repmat(predictive_gain,2,1)';
trf_p.powspctrm(:,2,:) = repmat(predictive_gain,2,1)';
trf_p.mask(:,1,:) = repmat(sorciere_pred.pval < 0.05, 2, 1)';
trf_p.mask(:,2,:) = repmat(sorciere_pred.pval < 0.05, 2, 1)';



trf_p.label               = channels;
trf_p.freq                = linspace(0,1,2);
trf_p.time                = linspace(0,1,2);
trf_p.dimord              = 'chan_freq_time'

fig = figure();
cfg             = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp; % 'ordered'
cfg.interactive = 'yes';
cfg.maskparameter = 'mask';
cfg.maskstyle = 'opacity';
cfg.maskalpha = 0.1; 
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim        = [0, prctile(predictive_gain,95)];
cfg.title       = 'Predictive regions (speech)';
cfg.comment = 'no';
cfg.box ='yes';
cfg.fontsize = 8;


if any(predictive_gain > 0)
    cfg.zlim = [0, prctile(predictive_gain(predictive_gain > 0), 95)];
else
    cfg.zlim = [0, 1];
end

set(gcf, 'Color', [1 1 1]);
ft_multiplotTFR(cfg, trf_p);

colorbar; 
cbar_title = title(colorbar, '% de variance expliquée');

% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);

saveas(fig, [fig_path '\speech_predictive_region.png'], 'png');

%%


%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define a structure for plotting kernels w/ fieldtrip
krnl_acou                  = [];
krnl_acou.label            = channels;
krnl_acou.freq             = linspace(-2, 2, 400)
krnl_acou.dimord           = 'chan_freq'

% initialize kernel w/ full of zeros 
krnl_acou.powspctrm = zeros(numel(channels), 400);

for i = 1:numel(channels)
    
    % Load kernels
    kernel_acou = squeeze(sorciere_acou.kernels(:,:,i));

    % Calcul the sum of squared 
    kernel_acou = zscore(sum(kernel_acou.^2, 2));
    
    % if r² > 0 : save kernel, else : keep zeros
    if sorciere_acou.r2(i) > 0
        krnl_acou.powspctrm(i,:) = kernel_acou

    end
end

%%%%%% PREDICTIF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define a structure for plotting kernels w/ fieldtrip
krnl_pred                  = [];
krnl_pred.label            = channels;
krnl_pred.freq             = linspace(-2, 2, 400)
krnl_pred.dimord           = 'chan_freq'

% initialize kernel w/ full of zeros 
krnl_pred.powspctrm = zeros(numel(channels), 400);

for i = 1:numel(channels)
    
    % Load kernels
    kernel_pred =squeeze(sorciere_pred.kernels(:, end-1:end, i));

    % Calcul the sum of squared 
    kernel_pred = zscore(sum(kernel_pred.^2, 2));
    
    % if gain > 0 : save kernel, else : keep zeros
    gain = sorciere_pred.r2_real(i) - sorciere_acou.r2(i);
    
    % Only keep kernel if gain > 0 and p < 0.05
    if gain > 0 && pvals(i) < 0.05
        krnl_pred.powspctrm(i,:) = kernel_pred;
    end

end

% plot kernels
fig = figure();
cfg              = [];
cfg.layout       = layout_bp; % 'ordered'
cfg.channel      = 'all';
cfg.interactive  = 'yes';
cfg.showoutline  = 'yes';
cfg.showlabels   = 'yes';
cfg.title        = 'Décours temporel acoustic du processing du language : acoustic (bleu) / predicif (rouge)';
cfg.comment      = 'no';
cfg.linecolor    = 'br';
cfg.comment = 'no';
cfg.fontsize = 8;
% Ajouter la légende

ft_multiplotER(cfg, krnl_acou, krnl_pred);
% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);

saveas(fig, [fig_path '/DécoursTemporel_speech.fig'], 'fig');

%%

% % % Pianos acoustic r2
% % Load r2
dat_file = '\pianos_acoustic.mat'
pianos_acou = load([out_path dat_file]);

pianos_acou.r2_hr = pianos_acou.r2*100
pianos_acou.r2_hr(pianos_acou.r2_hr < 0) = 0;

% % Plot TRF result
trf                     = [];
trf.powspctrm(:,1,:)    =  repmat(pianos_acou.r2_hr,2,1)';
trf.powspctrm(:,2,:)    =  repmat(pianos_acou.r2_hr,2,1)';

trf.label               = channels;
trf.freq                = linspace(0,1,2);
trf.time                = linspace(0,1,2);
trf.dimord              = 'chan_freq_time'

fig = figure();
cfg             = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp; % 'ordered'
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim        = [prctile(pianos_acou.r2_hr, 5), prctile(pianos_acou.r2_hr,95)];
cfg.title       = 'Acoustic regions (music)';
cfg.comment = 'no';
cfg.box ='yes';
cfg.fontsize = 8;

ft_multiplotTFR(cfg, trf);
colorbar; 
cbar_title = title(colorbar, '% de variance expliquée');

% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);
saveas(fig, [out_path '/figs' '/music_acoustic_region.png'], 'png');


% Pianos predictive gain 
% % Load r2
dat_file = '\pianos_predictive.mat'
pianos_pred = load([out_path dat_file]);

pvals = pianos_pred.pval;  % Charger les p-values

predictive_gain  = pianos_pred.r2_real*100 - pianos_acou.r2*100;

trf_p.powspctrm(:,1,:)    =  repmat(predictive_gain,2,1)';
trf_p.powspctrm(:,2,:)    =  repmat(predictive_gain,2,1)';
trf_p.mask(:,1,:) = repmat(pianos_pred.pval < 0.05, 2, 1)';
trf_p.mask(:,2,:) = repmat(pianos_pred.pval < 0.05, 2, 1)';




trf_p.label               = channels;
trf_p.freq                = linspace(0,1,2);
trf_p.time                = linspace(0,1,2);
trf_p.dimord              = 'chan_freq_time'

fig = figure();
cfg             = [];
cfg.channel     = 'all';
cfg.layout      = layout_bp; % 'ordered'
cfg.maskparameter = 'mask';
cfg.maskstyle = 'opacity';
cfg.maskalpha = 0.1; 
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels  = 'yes';
cfg.xlim        = [0,1];
cfg.ylim        = [0,1];
cfg.zlim        = [prctile(predictive_gain, 5), prctile(predictive_gain,95)];
cfg.title       = 'Predictive regions (music)';
cfg.comment = 'no';
cfg.box ='yes';
cfg.fontsize = 8;


if any(predictive_gain > 0)
    cfg.zlim = [0, prctile(predictive_gain(predictive_gain > 0), 95)];
else
    cfg.zlim = [0, 1];
end

set(gcf, 'Color', [1 1 1]);  
ft_multiplotTFR(cfg, trf_p);

colorbar; 
cbar_title = title(colorbar, '% de variance expliquée');

% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);

saveas(fig, [out_path '/figs' '/music_predictive_region.png'], 'png');

%%


%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Kernels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define a structure for plotting kernels w/ fieldtrip
krnl_acou                  = [];
krnl_acou.label            = channels;
krnl_acou.freq             = linspace(-2, 2, 400)
krnl_acou.dimord           = 'chan_freq'

% initialize kernel w/ full of zeros 
krnl_acou.powspctrm = zeros(numel(channels), 400);

for i = 1:numel(channels)
    
    % Load kernels
    kernel_acou = squeeze(zscore(pianos_acou.kernels(:,:,i)));

    % Calcul the sum of squared 
    kernel_acou = sum(kernel_acou.^2, 2);
    
    % if r² > 0 : save kernel, else : keep zeros
    if pianos_acou.r2(i) > 0
        krnl_acou.powspctrm(i,:) = kernel_acou

    end
end


%%%%%% PREDICTIF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pvals = pianos_pred.pval;

% define a structure for plotting kernels w/ fieldtrip
krnl_pred                  = [];
krnl_pred.label            = channels;
krnl_pred.freq             = linspace(-2, 2, 400)
krnl_pred.dimord           = 'chan_freq'

% initialize kernel w/ full of zeros 
krnl_pred.powspctrm = zeros(numel(channels), 400);

for i = 1:numel(channels)
    
    % Load kernels
    kernel_pred =squeeze(zscore(pianos_pred.kernels(:, end-1:end, i)));

    % Calcul the sum of squared 
    kernel_pred = sum(kernel_pred.^2, 2);
    
    % if gain > 0 : save kernel, else : keep zeros
    gain = pianos_pred.r2_real(i) - pianos_acou.r2(i);
    
    if gain > 0 && pvals(i) < 0.05
        krnl_pred.powspctrm(i,:) = kernel_pred;
    end

end

% plot kernels
fig = figure();
cfg              = [];
cfg.layout       = layout_bp; % 'ordered'
cfg.channel      = 'all';
cfg.interactive  = 'yes';
cfg.showoutline  = 'yes';
cfg.showlabels   = 'yes';
cfg.title        = 'Décours temporel acoustic du processing de la musique : acoustic (bleu) / predicif (rouge)';
cfg.comment      = 'no';
cfg.linecolor    = 'br';
cfg.comment = 'no';
cfg.fontsize = 8;

ft_multiplotER(cfg, krnl_acou, krnl_pred);

% Définir la position de la figure pour qu'elle prenne tout l'écran
screenSize = get(0, 'ScreenSize');
set(fig, 'Position', screenSize);

saveas(fig, [fig_path '/DécoursTemporel_music.fig'], 'fig');
