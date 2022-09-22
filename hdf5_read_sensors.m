% Copyright (c) 2022 Jake Carducci
% Version 1.0

clear; clc; close all;
set(groot,'defaultFigureVisible','on')
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0,'defaultAxesFontSize',18)
set(0,'defaultFigureUnits','normalized')
set(groot, 'defaultFigurePosition',[0.25 0.2 0.5 0.7])

filename = '06172019_s56_c.hdf5';
%filename = '06172019_s56_v.hdf5';

fs = 1000;
f_nyq = fs/2;
[b_band,a_band] = butter(4,[0.5,80]/f_nyq,'bandpass');
[b_notch,a_notch] = butter(4,[59,61]/f_nyq,'stop');
% -----------------------------------------------------------------
showRaw = false;

loadLegs = false;
newTrial = true;
zeroHit = true;
isLoading = true;

shortAvg = true;
cutoff = 10;
% ------------------------------------------------------------------
% NOTICE: FALSE FOR CALIBRATION; TRUE FOR VALIDATION
% PLEASE TOGGLE AS NEEDED
isValidate = false;
validTransform = false;
showError = isValidate; %true in valid
cmatKnown = isValidate; %true in valid
hasZeroing = isValidate; %true in valid
showCycles = ~isValidate; %false in valid
isFilter = false;
%------------------------------------------------------------------

addpath(genpath('Plugins'));
addpath(genpath('Data'));

% just noise, butrue example of extracting data
hinfo = h5info(filename);
num_grps = length(hinfo.Groups);

% Current trial index
tr_curr = 1;
% Current subtrial index w/n curr trial
st_curr = 1;
%
cycle_curr = 1;
leg_curr = 1;
% Initialize trial weight sequence
tr_weights = [];

% Initialize current angles
q_curr = [0,0];
dup = [];
validScaleLimit = 1500;
%%
show_arrows = false;

padding = 0.1;
% Color Tableload
color_load = [0, 114, 189, ...
            237, 177, 32, ...
            217, 83, 25];
        
color_unload = [120, 65, 154, ...
                181, 178, 33, ...
                213, 43, 93];

q = zeros(num_grps,2);
W = zeros(num_grps,1);
A = zeros(num_grps,3);
%tr = struct('st',,'weights',,'angles',,'data',,'averages',,'offsets',)

% Groups(#) = Collection subtrial
% Attribute(#) = 4:first angle, 5:second angle, 6:weight
for i = 1:num_grps
    % Read the angles of the subtrial
    q(i,1) = hinfo.Groups(i).Attributes(4).Value;
    q(i,2) = hinfo.Groups(i).Attributes(5).Value;
    % Read the weighloadt of the subtrial
    W(i,1) = hinfo.Groups(i).Attributes(6).Value;
    
    % When either angle changes, indicating a trial change
    if ((q(i,1) ~= q_curr(1) || q(i,2) ~= q_curr(2)) && i ~= 1)
        % Store 
        tr(tr_curr).st = st_curr - 1;
        tr(tr_curr).weights = tr_weights;
        if validTransform
            tr(tr_curr).angles = q_new(i-1,:);
        else
            tr(tr_curr).angles = q(i-1,:);
        end
        tr(tr_curr).data = tr_data;
        tr(tr_curr).averages = tr_avg;
        tr(tr_curr).offsets = tr_off;
        
        A(i-st_curr+1:i-1,:) = tr_avg;
        
        % Reset
        tr_data = {};
        tr_weights = [];
        tr_avg = [];
        cycle_curr = 1;
        
        % Increment trial number
        tr_curr = tr_curr + 1;
        % Reset subtrial number to 1
        st_curr = 1;
        % First data point becomes an offset
        tr_off = st_data(:,1);
        newTrial = true;
    end

    % Read all data from subtrial 
    st_data = h5read(filename, strcat(hinfo.Groups(i).Name, '/forces'));
    
    %Raw voltage readouts
    raw_voltage = h5read(filename, strcat(hinfo.Groups(i).Name, '/voltages'));
    
    if (isFilter)
        st_data = filter(b_notch,a_notch,filter(b_band,a_band,st_data));
        raw_voltage = filter(b_notch,a_notch,filter(b_band,a_band,raw_voltage));
    end
    
    if (newTrial && ~isValidate)
        % First data point of trial becomes an offset
        tr_off = st_data(:,1);
    elseif (hasZeroing)
        if (mod(i,2) == 1)
            tr_off = st_data(:,1);
        end
    elseif (isValidate && i == 1)
        %tr_off = [0;0.7;1];
    else
    end
    
    % Normalize data based on offset (assuming DC offset isn't removed)
    st_dataoff = st_data - tr_off;
    % Store normalized data into trial-wide cell array
    tr_data{st_curr} = st_dataoff;
    % Find average output for each dimension and store into trial-wide
    % array
    if (shortAvg)
        st_dataoff_cut = st_dataoff(:,1:cutoff);
    else
        st_dataoff_cut = st_dataoff;
    end
    tr_avg_curr = mean(st_dataoff_cut,2);
    tr_avg(st_curr,:) = tr_avg_curr;
    % Store subtrial weight in trial-wide array
    tr_weights(st_curr) = W(i);
    % Store angles and weights into current variables
    q_curr = q(i,:);
    if (i ~= 1 && i ~= num_grps)
        if (W(i) == 0 && ~zeroHit)
            zeroHit = true;
            cycle_ind{tr_curr}(cycle_curr,2) = st_curr - 1;
            cycle_curr = cycle_curr + 1;
            leg_curr = 1;
            isLoading = true;
        elseif (W(i) < W(i-1) && isLoading)
            isLoading = false;
            cycle_ind{tr_curr}(cycle_curr,1) = st_curr - 1;
            leg_curr = 1;
            zeroHit = false;
        elseif (W(i) > W(i-1) && ~isLoading)
            isLoading = true;
            leg_curr = 1;
        elseif (W(i) == W(i-1))
            dup = [dup,[i;W(i);tr_curr;st_curr]];
        else
        end

    else
        
    end
    
        % If very last subtrial of very last trial
    if (i == num_grps)
        % Store
        tr(tr_curr).st = st_curr;
        tr(tr_curr).weights = tr_weights;
        if validTransform
            tr(tr_curr).angles = q_new(i-1,:);
        else
            tr(tr_curr).angles = q(i-1,:);
        end
        tr(tr_curr).data = tr_data;
        tr(tr_curr).averages = tr_avg;
        tr(tr_curr).offsets = tr_off;
        cycle_ind{tr_curr}(cycle_curr,2) = st_curr;
        
        A(i-st_curr+1:i,:) = tr_avg;
    else
        % Increment subtrial by 1
        st_curr = st_curr + 1;
        leg_curr = leg_curr + 1;
    end
    newTrial = false;
end

if (validTransform)
    q = q_new;
end

theta = (q - [270,344])*pi./180;
% n-by-3
Wr = [W.*cos(theta(:,1)).*cos(theta(:,2)),-W.*sin(theta(:,1)),-W.*cos(theta(:,1)).*sin(theta(:,2))];
% Wr = C*A;
if (~cmatKnown)
    C = Wr'*A/(A'*A);
else
    load cal_mat_56_final.mat
end
W_adj = C*A';
e = W_adj - Wr';

wgt_list = unique(W);
m = length(wgt_list);
e_avg = zeros(3,m);
e_max = zeros(3,m);
e_min = zeros(3,m);
e_std = zeros(3,m);
e_sort = zeros(size(e));
ie = [1,1];
cmap = [0, 0.4470, 0.7410;0.8500, 0.3250, 0.0980;0.9290, 0.6940, 0.1250];
dim_str = {'X-dir Force','Y-dir Force','Z-dir Force'};

if (showError)
    figure(1)
    for k = 1:length(wgt_list)
        iw = find(W==wgt_list(k));
        
        % Errors that correspond to a given mass amount
        e_iw = e(:,iw);
        ie(2) = ie(1) + size(e_iw,2) - 1;
        e_iw_c(:,ie(1):ie(2)) = e_iw;
        g(:,ie(1):ie(2)) = wgt_list(k)*ones(size(e_iw));
        ie(1) = ie(2);
        % Statistics
        if (size(e_iw,2) ~= 1)
            e_avg(:,k) = mean(e_iw');
            e_max(:,k) = max(e_iw');
            e_min(:,k) = min(e_iw');
            e_25(:,k) = quantile(e_iw',0.25);
            e_med(:,k) = median(e_iw');
            e_75(:,k) = quantile(e_iw',0.75);
            e_std(:,k) = std(e_iw');
        else
            e_avg(:,k) = e_iw';
            e_max(:,k) = e_iw';
            e_min(:,k) = e_iw';
            e_25(:,k) = e_iw';
            e_med(:,k) = e_iw';
            e_75(:,k) = e_iw';
            e_std(:,k) = [0;0;0];
        end

    end
    for j = 1:3
        subplot(2,3,j)
        b_iqr = [abs(e_med(j,:) - e_25(j,:));abs(e_75(j,:) - e_med(j,:))];
        b_lim = [abs(e_med(j,:) - e_min(j,:));abs(e_max(j,:) - e_med(j,:))];
        hold on
        boundedline(wgt_list,e_med(j,:),b_lim','transparency',0.3,'cmap',cmap(j,:))
        boundedline(wgt_list,e_med(j,:),b_iqr','transparency',0.7,'cmap',cmap(j,:))
        hold off
        title(dim_str{j})
        ylabel('Fitted error (g)')
        ylim([-validScaleLimit validScaleLimit])
        xlabel('Actual mass (g)')
    end
    % Max length of whisker tails is 1.0 * IQR; beyond are outliers
    for j = 1:3
        subplot(2,3,j+3)
        boxplot(e_iw_c(j,:),g(j,:),'LabelOrientation','inline')
        title(dim_str{j})
        ylabel('Fitted error (g)')
        ylim([-validScaleLimit validScaleLimit])
        xlabel('Actual mass (g)')
    end

    figure(2)
    for i = 1:3
        subplot(3,1,i)
        hold on
        % Relative, not absolute !!!
        errorbar(wgt_list,e_avg(i,:),e_avg(i,:)-e_min(i,:),e_max(i,:)-e_avg(i,:),'Color',cmap(i,:))
        errorbar(wgt_list,e_avg(i,:),e_std(i,:),'Color',cmap(i,:),'LineWidth',2)
        title(dim_str{i})
        ylabel('Fitted error (g)')
        ylim([-validScaleLimit validScaleLimit])
        hold off
    end
    xlabel('Actual mass (g)')
end

num_row = 5;
for i = 1:tr_curr
    
    if (showRaw)
        figure(3*i)
        num_st = tr(i).st;
        num_col = ceil(num_st/num_row);
        for j = 1:num_st
            subplot(num_row,num_col,j)
            d = tr(i).data{j};
            d_max = max(d(:));
            d_min = min(d(:));
            plot(d')
            w = tr(i).weights(j);
            ylim([-0.2 0.8])
            xlabel(num2str(j))
            ylabel([num2str(w),' g'])
        end
        ax = axes;
        t1 = title(['Raw Force Data']);
        ax.Visible = 'Off';
        t1.Visible = 'On';
    end
    if (~isValidate)
        hax = figure(3*i + 1);
        a = tr(i).averages;
        a_max = max(a(:));
        a_min = min(a(:));
        w = tr(i).weights;
        th_tr = tr(i).angles;
        range = max(a,[],1) - min(a,[],1);
              
        plot(w,a,'--*')
        ylim([a_min-padding a_max+padding])
        xlabel('Weight (g)')
        ylabel({'Relative force'; 'magnitude (V)'})
        title({'Force Averages (Overall)';['Angles: ',num2str(th_tr(1)),', ',num2str(th_tr(2))]})
        set(gca,'box','off')
        
        for j = 1:3
            switch j
                case 1
                    style = '-c';
                case 2
                    style = '-m';
                case 3
                    style = '-g';
                otherwise
                    style = '-k';
            end
            p{i,j} = polyfit(w,a(:,j)',1);
            a_fit = polyval(p{i,j},w);
            nl_err{i,j} = a(:,j) - a_fit';
            SSr = sum(nl_err{i,j}.^2);
            SSt = (length(a(:,j))-1)*var(a(:,j));
            nl_err2(i,j) = 1 - SSr/SSt;
            hold on
            plot(w,a_fit,style)
            hold off
            stats{i}(:,j) = [p{i,j}';nl_err2(i,j)];
        end
    end
    
    if (show_arrows)
        quiv_scale = [1,1000];
        headWidth = 4;
        headLength = 6;
        LineLength = 0.08;
        LineWidth = 1;
        quiv_ind = (6:10:num_st);
        a_quiv = a(quiv_ind,:);
        w_quiv = [w(quiv_ind)',w(quiv_ind)',w(quiv_ind)'];
        w_quiv_m1 = [w(quiv_ind - 1)',w(quiv_ind - 1)',w(quiv_ind - 1)'];
        a_delta = a_quiv - a(quiv_ind - 1,:);
        w_delta = w_quiv - w_quiv_m1;
        len_delta = hypot(a_delta,w_delta);
        hold on
        hq = quiver(w_quiv,a_quiv,-w_delta,-a_delta,quiv_scale(i),'ShowArrowHead','off','LineWidth',LineWidth,'Color','k');
        hold off
        U = hq.UData;matlab
        V = hq.VData;
        X = hq.XData;
        Y = hq.YData;
    
        for ii = 1:size(X,1)
            for ij = 1:size(X,2)
                headWidth = 5;
                ah = annotation('arrow',...
                    'headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
                set(ah,'parent',gca);
                set(ah,'position',[X(ii,ij) Y(ii,ij) U(ii,ij) V(ii,ij)]);

            end
        end
    end
    
    if (~isValidate)
        legend({'X avg','Y avg','Z avg','X best fit','Y best fit','Z best fit'},'Location','northwest','NumColumns',2)
    end
        
        
    % Broken across trials
    if (showCycles && ~isValidate)
        figure(3*i + 2)
        % number of cycles
        n = size(cycle_ind{i},1);
        for j = 1:n
            if (j == 1)
                d_leg = a(1:cycle_ind{i}(j,2)+1,:);
                w_leg = w(1:cycle_ind{i}(j,2)+1);              
            else
                d_leg = a(cycle_ind{i}(j-1,2):cycle_ind{i}(j,2),:);
                w_leg = w(cycle_ind{i}(j-1,2):cycle_ind{i}(j,2));
            end
            d_leg_max = max(d_leg(:));
            d_leg_min = min(d_leg(:));
            % Need a subset of w
            subplot(n,1,j)
            plot(w_leg,d_leg','--*')
            ylim([d_leg_min-padding d_leg_max+padding])
            ylabel({['Cycle ',num2str(j)];' Magnitude (V)'})
            set(gca,'box','off')

            if (show_arrows)
                quiv_ind_leg = [quiv_ind(2*j-1),quiv_ind(2*j)];
                a_quiv_leg = a(quiv_ind_leg,:);
                w_quiv_leg = [w(quiv_ind_leg)',w(quiv_ind_leg)',w(quiv_ind_leg)'];
                w_quiv_m1_leg = [w(quiv_ind_leg - 1)',w(quiv_ind_leg - 1)',w(quiv_ind_leg - 1)'];
                a_delta_leg = a_quiv_leg - a(quiv_ind_leg - 1,:);
                w_delta_leg = w_quiv_leg - w_quiv_m1_leg;
                len_delta_leg = hypot(a_delta_leg,w_delta_leg);
                hold on
                hq = quiver(w_quiv_leg,a_quiv_leg,-w_delta_leg./len_delta_leg,-a_delta_leg./len_delta_leg,quiv_scale(i),'ShowArrowHead','off','LineWidth',LineWidth,'Color','k');
                hold off
                U = hq.UData;
                V = hq.VData;
                X = hq.XData;
                Y = hq.YData;
                for ii = 1:size(X,1)
                    for ij = 1:size(X,2)

                        headWidth = 5;
                        ah = annotation('arrow',...
                            'headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
                        set(ah,'parent',gca);
                        set(ah,'position',[X(ii,ij) Y(ii,ij) -LineLength*U(ii,ij) -LineLength*V(ii,ij)]);

                    end
                end
            end

        end
        xlabel('Weight (g)')
        ax = axes;
        t1 = title({'Force Averages (By Cycle)';['Angles: ',num2str(th_tr(1)),', ',num2str(th_tr(2))]});
        ax.Visible = 'Off';
        t1.Visible = 'On';
        
    end
end

if (isValidate)
    % Component figure
    legendStr = {'X fit','X','Y fit','Y','Z fit','Z'};
    W_adj = W_adj';
    for j = 1:3
        switch j
            case 1
                style_scatter = '.b';
                style_fit = '-c';
            case 2
                style_scatter = '.r';
                style_fit = '-m';
            case 3
                style_scatter = '.g';
                style_fit = '-g';
            otherwise
                style_scatter = '.k';
                style_fit = '-k';
        end
        figure(30)
        subplot(3,1,j);
        p{j} = polyfit(Wr(:,j),W_adj(:,j),1);
        a_fit = polyval(p{j},Wr(:,j));
        hold on
        plot(sort(Wr(:,j)),sort(a_fit),style_fit)
        plot(Wr(:,j),W_adj(:,j),style_scatter)
        plot(-1500:100:1500,-1500:100:1500,'LineStyle','--','Color','[0.6 0.6 0.6]')
        legend(legendStr{2*j-1},legendStr{2*j},'Location','northwest')
        xlabel('Actual mass (g)')
        ylabel('Calib mass (g)')
        set(gca,'box','off')
        hold off
        
        mean_W_adj(j) = mean(W_adj(:,j));
        SST(j) = sum((W_adj(:,j) - mean_W_adj(j)).^2);
        SSR(j) = sum((Wr(:,j) - mean_W_adj(j)).^2);
        RSS(j) = sum((W_adj(:,j) - Wr(:,j)).^2);
        R2(j) = 1 - RSS(j)/SST(j);
        
    end
    
    % Overall figure
    figure(31)
    Wr_all = sqrt(sum(Wr.^2,2));
    W_adj_all = sqrt(sum(W_adj.^2,2));
    
    plot(Wr_all,W_adj_all,'.r')
    p = polyfit(Wr_all,W_adj_all,1);
    a_fit = polyval(p,Wr_all);
    hold on
    plot(Wr_all,a_fit,'-r')
    plot(0:100:1100,0:100:1100,'LineStyle','--','Color','[0.6 0.6 0.6]')
    hold off
    xlabel('Actual mass (g)')
    ylabel('Calib mass (g)')
    legend('Raw data','Best-fit line','Ideal sensor line','Location','northwest')
    
    mean_W_adj_all = mean(W_adj_all);
    SST_overall = sum((W_adj_all - mean_W_adj_all).^2);
    SSR_overall = sum((Wr_all - mean_W_adj_all).^2);
    RSS_overall = sum((W_adj_all - Wr_all).^2);
    R2_overall = 1 - RSS_overall/SST_overall;
    
    % Residual plot
    figure(32)
    r = W_adj_all - Wr_all;
    hold on
    plot(Wr_all,r,'.r')
    plot(0:100:1500,zeros(1,16),'-k')
    hold off
    xlabel('Actual mass (g)')
    ylabel('Residual error (g)')
    
    mean_W_adj(j) = mean(W_adj(:,j));
    SST(j) = sum((W_adj(:,j) - mean_W_adj(j)).^2);
    SSR(j) = sum((Wr(:,j) - mean_W_adj(j)).^2);
    RSS(j) = sum((W_adj(:,j) - Wr(:,j)).^2);
    R2(j) = 1 - RSS(j)/SST(j);
    
    r_scale = 100 * r ./ Wr_all;
    
    RMSE_1200 = sqrt(sum(r.^2)/length(r));
    RMSEp_1200 = 100*RMSE_1200 / 1200;
    
    Wr_1000 = Wr_all <= 1050;
    r_1000 = r .* Wr_1000;
    RMSE_1000 = sqrt(sum(r_1000.^2)/length(r_1000));
    RMSEp_1000 = 100*RMSE_1000 / 1000;   
    
    figure(33)
    histogram(r,15,'FaceAlpha',1)
    xlabel('Residual error (g)')
    ylabel('Frequency')
    
    figure(34)
    boxplot(A(1:2:end,:),'Labels',{'X','Y','Z'})
    ylabel('Calculated Reading')
    
    for j = 1:num_grps/2
        Z(j,:) = (C*A(2*j-1,:)')';
    end
    
    figure(35)
    boxplot(Z,'Labels',{'X','Y','Z'})
    ylabel('Calculated Mass (g)')
    
    
    %C-Space
    calib_angles = [315,344;45,344;135,344;225,344;270,74;270,254;90,74;90,254;315,164;45,164;135,164;225,164];
    err_max = max(max(r),max(max(e)));
    err_min = min(min(r),min(min(e)));
    err_rng = err_max - err_min;
    r_norm = (r - err_min)./err_rng;
    e_norm = (e - err_min)./err_rng;
    err_map = linspace(err_min,err_max,length(r));
    e_rel = 100.*e(:,2:2:end)'./Wr(2:2:end,:);
    r_rel = 100.*r(2:2:end)./W(2:2:end);
    q2 = q(2:2:end,:);
    q2_1 = q2(:,1);
    q2_2 = q2(:,2);
    
    figure(36)
    colormap(turbo);
    dim_str2 = {'X error','Y error','Z error'};
    
    subplot(2,2,1)
    scatter(q2_1,q2_2,[],r(2:2:end));
    hold on
    plot(calib_angles(:,1),calib_angles(:,2),'xb')
    title('Total residual')
    xlabel('Angle 1 (degrees)')
    ylabel('Angle 2 (degrees)')
    cb = colorbar;
    cb.Label.String = 'Residual (g)';
    xlim([0 360])
    xticks([0 90 180 270 360])
    ylim([0 360])
    yticks([0 90 180 270 360])
    caxis([-350 250])
    hold off
    for i=1:3
        subplot(2,2,1+i)
        scatter(q2_1,q2_2,[],e(i,2:2:end)');
        hold on
        plot(calib_angles(:,1),calib_angles(:,2),'xb')
        cb = colorbar;
        cb.Label.String = 'Error (g)';
        xlim([0 360])
        xticks([0 90 180 270 360])
        ylim([0 360])
        yticks([0 90 180 270 360])
        caxis([-350 250])
        title(dim_str2(i))
        xlabel('Angle 1 (degrees)')
        ylabel('Angle 2 (degrees)')
        hold off
    end
    
    figure(37)
    colormap(turbo);
    dim_str2 = {'X relative error','Y relative error','Z relative error'};

    subplot(2,2,1)
    scatter(q2_1,q2_2,[],r_rel);
    hold on
    plot(calib_angles(:,1),calib_angles(:,2),'xm')
    title('Total relative residual')
    xlabel('Angle 1 (degrees)')
    ylabel('Angle 2 (degrees)')
    cb = colorbar;
    cb.Label.String = 'Residual wrt load (%)';
    xlim([0 360])
    xticks([0 90 180 270 360])
    ylim([0 360])
    yticks([0 90 180 270 360])
    caxis([-100 100])
    hold off
    for i=1:3
        subplot(2,2,1+i)
        scatter(q2_1,q2_2,[],e_rel(:,i));
        hold on
        plot(calib_angles(:,1),calib_angles(:,2),'xm')
        cb = colorbar;
        cb.Label.String = 'Error wrt load component (%)';
        xlim([0 360])
        xticks([0 90 180 270 360])
        ylim([0 360])
        yticks([0 90 180 270 360])
        caxis([-150 150])
        title(dim_str2(i))
        xlabel('Angle 1 (degrees)')
        ylabel('Angle 2 (degrees)')
        hold off
    end   
    
    figure(38)
    dim_str2 = {'X surface fit','Y surface fit','Z surface fit'};
    subplot(2,2,1)
    f_r = fit([q2_1,q2_2],r_rel,'poly22');
    f_r_mdl = fitlm([q2_1,q2_2],r_rel,'poly22');
    plot(f_r,[q2_1,q2_2],r_rel);   
    hold on
    
    title('Total residual surface fit')
    xlabel('Angle 1 (degrees)')
    ylabel('Angle 2 (degrees)')

    xlim([0 360])
    xticks([0 90 180 270 360])
    ylim([0 360])
    yticks([0 90 180 270 360])

    hold off
    for i=1:3
        subplot(2,2,1+i)
        f_e{i} = fit([q2_1,q2_2],e_rel(:,i),'poly22');
        f_e_mdl{i} = fitlm([q2_1,q2_2],e_rel(:,i),'poly22');
        plot(f_e{i},[q2_1,q2_2],e_rel(:,i));
        hold on

        xlim([0 360])
        xticks([0 90 180 270 360])
        ylim([0 360])
        yticks([0 90 180 270 360])

        title(dim_str2(i))
        xlabel('Angle 1 (degrees)')
        ylabel('Angle 2 (degrees)')
        hold off
    end   
end

if (~isValidate)
    rate = [0;sign(diff(W))];
    delta_q = [1;any(~(diff(q) == 0),2)];
    W0 = W == 0;

    %% Mark current pose number across session
    poses = cumsum(delta_q);

    %% Mark cycle number (overall and per pose) across session
    cyc = double(xor(circshift(delta_q,-1),W0));
    cyc_all = cumsum(cyc);
    reset = -diff([0;cyc_all(logical(delta_q))-1]);
    cyc(logical(delta_q)) = cyc(logical(delta_q)) + reset;
    cyc_pose = cumsum(cyc);
    pose_list = unique(poses);
    cyc_list = unique(cyc_pose);
    
    FSO = max(A);
    noise_max = zeros(1,3);
    for i = 1:tr_curr
        for j = 1:st_curr
            d = tr(i).data{j};
            noise = (max(d,[],2) - min(d,[],2))';
            noise_max(noise > noise_max) = noise(noise > noise_max);
        end
    end

    rpt = zeros(1,3);
    hys = zeros(1,3);
    nl = zeros(1,3);
    % for each angular configuration
    for i = 1:length(pose_list)
        idx2 = (poses == pose_list(i));
        A_pose = A(idx2,:);
        W_pose = W(idx2,:);
        
        px = polyfit(W_pose,A_pose(:,1),1);
        py = polyfit(W_pose,A_pose(:,2),1);
        pz = polyfit(W_pose,A_pose(:,3),1);
        
        px_inv = [1/px(1),-px(2)/px(1)];
        py_inv = [1/py(1),-py(2)/py(1)];
        pz_inv = [1/pz(1),-pz(2)/pz(1)];
        
        noise_load = [polyval(px_inv,noise_max(1)),polyval(py_inv,noise_max(2)),polyval(pz_inv,noise_max(3))];
        
        % for each weight index
        for j = 1:length(wgt_list)
            % able to exclude first cycle to mitigate offset effects
            for k2 = 2:length(cyc_list)
            % for each loading direction
                for k = -1:1
                    idx = (poses == pose_list(i)) & (W == wgt_list(j)) & (rate == k) & (cyc_pose == cyc_list(k2));
                    A_repeat = A(idx,:);
                    A_diff = max(A_repeat,[],1) - min(A_repeat,[],1);
                    rpt(A_diff > rpt) = A_diff(A_diff > rpt);
                end
                % for each cycle index
                for k = 1:length(cyc_list)
                    % should only be two idx
                    idx = (poses == pose_list(i)) & (W == wgt_list(j)) & (cyc_pose == cyc_list(k)) & (cyc_pose == cyc_list(k2));
                    for l = 1:3
                        A_hys = A(idx,:);
                        A_diff = max(A_hys,[],1) - min(A_hys,[],1);
                        hys(A_diff > hys) = A_diff(A_diff > hys);
                    end
                end
            end
            idx = (cyc_pose == cyc_list(k2)) & (poses == pose_list(i));
            A_nl = A(idx,:);
            W_nl = W(idx,:);
                
            A_fit = [polyval(px,W_nl),polyval(py,W_nl),polyval(pz,W_nl)];
            A_diff = max((A_fit' - A_nl'),[],2)';
            nl(A_diff > nl) = A_diff(A_diff > nl);
        end
    end
    
    % Hysteresis
    hys_p = 100*hys./FSO;
    % Repeatability
    rpt_p = 100*rpt./FSO;
    % Nonlinearity
    nl_p = 100*nl./FSO;
    % Accuracy
    acc_p = sqrt((max(hys_p)^2 + max(rpt_p)^2 + max(nl_p)^2)/3);
    
    % Voltage response
    figure(40)
    % 315,344; 500g
    raw_bits = h5read(filename, strcat(hinfo.Groups(2).Name, '/voltages'));
    raw_mV = double(raw_bits)'.*3300./(2^16);
    % Second strain gauge
    raw_mV_2 = raw_mV(:,2);
    raw_T = (1:length(raw_mV_2))./fs;
    plot(raw_T,raw_mV_2,'Color','#D95319')
    hold on
    plot(raw_T,max(raw_mV_2)*ones(1,length(raw_mV_2)),'Color','#77AC30','LineStyle','--')
    plot(raw_T,min(raw_mV_2)*ones(1,length(raw_mV_2)),'Color','#4DBEEE','LineStyle','--')
    xlabel('Time (seconds)')
    ylabel('Gauge signal voltage (mV)')
    hold off
    
    %noise_raw = (max(raw_mV_2) - min(raw_mV_2)) / 1000;
    noise_levels = noise_max*(2^16)/3.300;
    noise_volt = (FSO./noise_max);
    noise_bits = log2(noise_levels);
    SNR = 10*log10(noise_volt.^2);
end
