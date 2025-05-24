clc; close all; clear;

% Load data
T = readtable('./data/route1.csv');
% Get receiver coordinate
rx_coords = table2array(T(:,1:3));
% Set transmitter coordinates
tx_coord = [81.36, -318, 13];
% Compute distance between tx and rx:s
d = sqrt((tx_coord(:,1)-rx_coords(:,1)).^2 + ...
         (tx_coord(:,2)-rx_coords(:,2)).^2 + ...
         (tx_coord(:,3)-rx_coords(:,3)).^2 );
% Get pathlosses
y = table2array(T(:,4));

%%

figure(1)
% Plot transmitter and receiver coordinates on a map
plot(rx_coords(:,1), rx_coords(:,2), 'g', 'LineWidth', 2)
hold on
plot(tx_coord(:,1), tx_coord(:,2),'o','LineWidth',2)
title('Map (x,y) over measurement area')
legend({'Receiver locations', 'Transmitter location'}, 'Location', 'southeast')
xlabel('x [m]')
ylabel('y [m]')

% ----- Task 3a: Perform linear regression with all data ---------------- %

d0 = 1;

logd = 10*log10(d./d0);
A = [ones(length(logd),1), logd];
x = A \ y; % Linreg

% Plot linlog
figure(2)
hold on

scatter(d,y)
plot(d, A * x, 'g', 'LineWidth', 2)

title('Linear Pathloss')
legend({'Datapoint', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss')
set(gca,'yscale','log')

saveas(gcf, 'linpath.png', 'png')

% Plot loglog
figure(3)
hold on

scatter(d,y)
plot(d, A * x, 'g', 'LineWidth', 2)

title('Log Pathloss')
legend({'Datapoint', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss')
set(gca,'yscale','log', 'xscale', 'log')

saveas(gcf, 'logpath.png', 'png')

% ----------------------------------------------------------------------- %

%%

% ----- Task 3b: Perform linear regression excluding outliers ----------- %
% -- Identify outliers -- %
%{
Remove outlier based on the residual of data points from the regression
line
%}

% Set threshold for residual
resid_th = 10; % Set an appropriate value dB
% Get residual 
r = y - A*x; 
% Get indices that correspond to outliers
I = get_outlier_ind(r, resid_th);
% Get y and d without outliers 
y_no = y; d_no = d; 
y_no(I) = [];
d_no(I) = [];

% -- Re-do Task 3a -- % 

d_nolog = 10*log10(d_no./d0);
A_no = [ones(length(d_nolog),1), d_nolog];
x_no = A_no \ y_no; % Linreg

y_yes = y(I);
d_yes = d(I);

% Plot linlog
figure(4)
hold on

scatter(d_no, y_no)
scatter(d_yes, y_yes, 'r*')
plot(d_no, A_no * x_no, 'g', 'LineWidth', 2)

title('Linear Pathloss, outliers marked')
legend({'Datapoint', 'Outlier', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss')
set(gca,'yscale','log') 

saveas(gcf, 'linpathwoout.png', 'png')

% Plot loglog
figure(5)
hold on

scatter(d_no, y_no)
scatter(d_yes, y_yes, 'r*')
plot(d_no, A_no * x_no, 'g', 'LineWidth', 2)

title('Log Pathloss, outliers marked')
legend({'Datapoint', 'Outlier', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss')
set(gca,'yscale','log', 'xscale', 'log')

saveas(gcf, 'logpathwoout.png', 'png')

% ----------------------------------------------------------------------- %

% ----- Task 3c: Optional. Identify outliers on map and explain why ----- %
figure(1)
plot(rx_coords(I,1), rx_coords(I,2), 'r*')
legend({'Receiver locations', 'Transmitter location', ...
        'Outliers'}, 'Location', 'southeast')
saveas(gcf, 'map.png', 'png')
% ----------------------------------------------------------------------- %

% ---- Task 4: Create data and determine if the estimator appears ------- %
% ---- consistent and/or unbiased                                 ------- %

%%

%{
Function to generate measurements y, of the pathloss at distances d, given
- vector d, of distances
- scalar d0, reference distance at which we have known pathloss
- scalar l_d0, known pathloss at distance d0
- scalar a, pathloss exponent
- scalar sigma2, variance of noise in s-term
%}

d_gen = (50:1800)';
l_d0 = x_no(1);
a = x_no(2);
r = y - A * x;
sigma2 = (r'*r)/(length(r)-2);

[A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2);

x_gen = A_gen \ y_gen; % Linreg

len = 1000;

x_tot = zeros(len,2);
sigma2_tot = zeros(len,1);
cumulative_avg = zeros(len,1);

for i = 1:len
    [A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2);
    x_tot(i,:) = (A_gen \ y_gen)'; % Linreg
    r_gen = y_gen - A_gen * x_tot(i,:)';
    sigma2_tot(i) = (r_gen'*r_gen)/(length(r_gen)-2);
    cumulative_avg(i) = sum(sigma2_tot(1:i)/i);
end

figure(6)
hold on

scatter(1:len, cumulative_avg)
plot(1:len, ones(len,1) * sigma2, 'r', 'LineWidth', 2)
set(gca,'yscale','log', 'xscale', 'log')
title('Cumulative mean of variance')
xlabel('Samples')
ylabel('Cumulative mean variance')
legend({'Cumulative mean variance', 'Original variance'}, 'Location', 'southeast')

ylim([sigma2-3, sigma2+3])

saveas(gcf, 'meanvar.png', 'png')

% ----------------------------------------------------------------------- %

%%

% ---- Task 5: Create data and 95% confidence intervals for each -------- %
% ---- of the parameters l0 and a                                -------- %

len = 10000;

x_tot = zeros(len,2);
sigma2_tot = zeros(len,1);

for i = 1:len
    [A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2);
    x_tot(i,:) = (A_gen \ y_gen)'; % Linreg
    r_gen = y_gen - A_gen * x_tot(i,:)';
    sigma2_tot(i) = (r_gen'*r_gen)/(length(r_gen)-2);
end

l_d0_tot = x_tot(:,1);
a_tot = x_tot(:,2);

[l_d0_low, l_d0__high, ul, il] = conf_interval(l_d0_tot)
[a_low, a_high, ua, ia] = conf_interval(a_tot)

% oiia oiia

bins=40;

figure(7)
histogram(l_d0_tot,bins)

title('Histogram l0')
xlabel('Value')
ylabel('Count')

saveas(gcf, 'hista.png', 'png')

figure(8)
histogram(a_tot,bins)

title('Histogram a')
xlabel('Value')
ylabel('Count')

saveas(gcf, 'histl0.png', 'png')
% ----------------------------------------------------------------------- %

%%

%{
Help function that identifies outliers based on the residual r = y - A*x
exceeding a set threshold resid_th.
%}
function I = get_outlier_ind(r, resid_th)
    % Get all indices where |r|>resid_th
    I = find(abs(r) > resid_th);
end

function [C1, C2, u, i] = conf_interval(data)
    n = length(data);
    u = mean(data);
    res = data - u;
    s = sqrt(sum(res.^2)/(n-1));
    i = s * 1.9600 / sqrt(n);
    C1 = u - i;
    C2 = u + i;
end
