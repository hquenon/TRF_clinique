function data_bipolar = iEEG_data2bipolar(data)

% DATA_BIPOLAR computes a bipolar reference montage on s-EEG data
%                       (i.e. two consecutive contacts on the same shaft)
%
% dependency: functions from fieldtrip (ft_channelselection, ft_preprocessing, ft_appenddata)
%
% Input:
%       data: data from fieldtrip
%             
% Output:
%       data_bipolar: data referenced to bipolar montage
%                     bipolar montage is applied to the channel positions
%                     (chanpos field containing the mean locations of electrode pairs corresponding a bipolar channel
%
%
% CC-BY-NC-SA
%
% Nov 2020 _ function created by MrM (manuel.mercier@inserm.fr)
% 
%

%% get shaft names
shaft_name = unique(string(regexp(data.label,'\D+(\.)?(\D+)?','match')));

% trick to deal with names
if ~ all(contains(shaft_name,'_'))
    data.oldlabel= data.label;
    e_shaft = regexp(data.label,'\D+(\.)?(\D+)?','match');
    e_num = regexp(data.label,'\d+(\.)?(\d+)?','match');
        
    for i=1:length(data.label)
        if ~ isempty(e_num{i})
            data.label(i,1) = strcat(e_shaft{i},'_', e_num{i});
        else
            data.label(i,1) = strcat(e_shaft{i});
        end
    end
    shaft_name = unique(string(regexp(data.label,'\D+(\.)?(\D+)?','match')));
end

for e = 1:length(shaft_name)

    cfg            = [];
    cfg.channel    = ft_channelselection([shaft_name{e} '*'], data.label);
    cfg.reref      = 'yes';
    cfg.refchannel = 'all';
    cfg.refmethod  = 'bipolar';
    cfg.updatesens = 'yes';
    data_reref{e}  = ft_preprocessing(cfg, data);
    
end
    
cfg            = [];
cfg.appendsens = 'yes';
data_bipolar = ft_appenddata(cfg, data_reref{:});

if isfield(data,'oldlabel')
   for e = 1:length(data_bipolar.label)
       idx = strfind(data_bipolar.label{e},'_');
       data_bipolar.label{e}(idx)=[]; 
   end
end

end