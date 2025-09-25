function layout = sEEG_layout(dat)

% SEEG_layout prepare electrode layout where electrodes are ordered by shaft
%
% dependency: functions from fieldtrip (ft_prepare_layout, ft_appendlayout)
%
%
%
% Inputs:
%        cfg.dat = data with field label containing electride names
%
%
%
% Outputs:
%         layout structure used in fieldtrip
%
%
%
% Aug.  2021 _ function created by MrM (manuel.mercier@inserm.fr)
% Sept. 2021 _ modifs to take into account the different label format (with/out underscore, bipolar montage with hyphen)
%
%

%% CLEANUP TO DO
tmp = dat;
%% trick to deal with label format

% bipolar
if all(contains(dat.label,'-'))
    bipolar = 'yes';
    e_shaft = regexp(tmp.label,'\D+(\.)?(\D+)?','match');
    e_num = regexp(tmp.label,'\d+(\.)?(\d+)?','match');
    tmp.labelold = tmp.label;
    for e = 1:length(dat.label)
       tmp.label{e} = [e_shaft{e}{1} '_' e_num{e}{1} '-' e_num{e}{2}];
    end
    shaft_n = regexp(tmp.label,'\D+(\.)?(\D+)?','match');
    clear e_shaft e_num;
    for e=1:length(shaft_n); shaft_name(e) = shaft_n{e}(1); end
    shaft_name = unique(shaft_name);
    
% not bipolar
elseif ~ all(contains(dat.label,'_'))
    unscrd = 'no';
    e_shaft = regexp(tmp.label,'\D+(\.)?(\D+)?','match');
    e_num = regexp(tmp.label,'\d+(\.)?(\d+)?','match');
    for i=1:length(tmp.label)
        if ~ isempty(e_num{i})
            tmp.label(i,1) = strcat(e_shaft{i},'_', e_num{i});
        else
            tmp.label(i,1) = strcat(e_shaft{i});
        end
    end
    e_shaft = regexp(tmp.label,'\D+(\.)?(\D+)?','match');

    for i=1:length(e_shaft); shaft_name{i,1} = e_shaft{i}{1}; end
    shaft_name = unique(shaft_name);
elseif all(contains(dat.label,'_'))
    shaft_name = unique(string(regexp(dat.label,'\D+(\.)?(\D+)?','match')));
end

%% create layout
for s = 1:length(shaft_name)
    cfg             = [];
    cfg.layout      = 'vertical';
    cfg.direction   = 'TB';
    if contains(shaft_name{s},'_')
        cfg.channel     = ['' shaft_name{s} '*'];
    else
        cfg.channel     = shaft_name{s};
    end
    cfg.width        = 0.04;
    cfg.height       = 0.03;
    layout_tmp      = ft_prepare_layout(cfg, tmp);
    
    if s ==1
    layout = layout_tmp;
    elseif s > 1
        cfg = [];
        cfg.direction = 'horizontal';
        cfg.align     = 'top';
        cfg.distance = 0.05;
        layout = ft_appendlayout(cfg, layout, layout_tmp); 
    end
    clear layout_tmp; 
end

%% plot the layout
layout_tmp = layout;
for i=1:length(layout_tmp.label)
    idx = strfind(layout_tmp.label{i},'_');
    layout_tmp.label{i}(idx) = [];
end
figure;
ft_plot_layout(layout_tmp);


%% deal with the different formats
if exist('bipolar')
    if bipolar == 'yes'
    layout.label = dat.label;
    end
elseif exist('unscrd')
    if unscrd == 'no'
        layout = layout_tmp;
    end
end

end