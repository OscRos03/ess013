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
A = [ones(length(logd),1), logd]; % Create A matrix as seen in report
x = A \ y; % Do linear regression on data

% Plot linlog
figure(2)
hold on

scatter(d,y) % Scatter plot of real distance vs path loss
plot(d, A * x, 'g', 'LineWidth', 2) % Plot linear regression of said values

title('Linear Pathloss')
legend({'Datapoint', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss [dB]')
set(gca,'yscale','log')

saveas(gcf, 'linpath.png', 'png')

% Plot loglog
figure(3)
hold on

scatter(d,y) % Scatter plot of real distance vs path loss
plot(d, A * x, 'g', 'LineWidth', 2) % Plot linear regression of said values

title('Log Pathloss')
legend({'Datapoint', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss [dB]')
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

% Same as 3a but with no outliers in linear regression
% Outliers plotted separately
scatter(d_no, y_no)
scatter(d_yes, y_yes, 'r*')
plot(d_no, A_no * x_no, 'g', 'LineWidth', 2)

title('Linear Pathloss, outliers marked')
legend({'Datapoint', 'Outlier', 'Model'}, 'Location', 'southeast')
xlabel('d [m]')
ylabel('Loss [dB]')
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
ylabel('Loss [dB]')
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

l_d0 = x_no(1); % l_d0 of model with outliers excluded
a = x_no(2);    % a of model with outliers excluded
r = y - A * x;
sigma2 = (r'*r)/(length(r)-2); % Estimated variance of original data

% --------- Check unbiasedness -----------

len = 1000; % Amount of estimator samples
d_gen = (50:1800)'; % Data point spacing

x_tot = zeros(len,2);
sigma2_tot = zeros(len,1);
cumulative_avg = zeros(len,1);

% 
for i = 1:len
    % Genetate data points according to model
    [A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2);
    x_tot(i,:) = (A_gen \ y_gen)'; % Linreg
    r_gen = y_gen - A_gen * x_tot(i,:)'; % Calculate residual
    sigma2_tot(i) = (r_gen'*r_gen)/(length(r_gen)-2); % Estimate variance
    cumulative_avg(i) = sum(sigma2_tot(1:i)/i); % Calculate cumulative mean
end

figure(6)
hold on

% Plot cumulative mean of variance with logarithmic x-axis
% With a horizontal line of the variance of the original data set
% as a reference
scatter(1:len, cumulative_avg, '.')
yline(sigma2, 'r', 'LineWidth', 2)
set(gca, 'xscale', 'log')
title('Cumulative mean of variance')
xlabel('Samples')
ylabel('Cumulative mean variance [dB^2/m^2]')
legend({'Cumulative mean variance', 'Original variance'}, 'Location', 'southeast')

ylim([sigma2-3, sigma2+3])

saveas(gcf, 'meanvar.png', 'png')

% --------- Check consistency -----------

n = 50:10:5000; % Vector of population sizes
len = length(n);

sigma2_tot = zeros(len,1);

for i = 1:len
    % Genetate n(i) evenly spaced data points
    d_gen = linspace(50,1800,n(i))';
    [A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2);
    
    x_gen = (A_gen \ y_gen)'; % Linreg
    r_gen = y_gen - A_gen * x_gen'; % Calculate residual
    sigma2_tot(i) = (r_gen'*r_gen)/(length(r_gen)-2); % Estimate variance
end

figure(7)
hold on

% Plot change in variance as population size of generated data is increased
% With a horizontal line of the variance of the original data set
% as a reference
scatter(n, sigma2_tot, '.')
yline(sigma2, 'r', 'LineWidth', 2)
title('Variance per given population size')
xlabel('Population size')
ylabel('Variance [dB^2/m^2]')
legend({'Variance', 'Original variance'}, 'Location', 'southeast')

saveas(gcf, 'var_per_population.png', 'png')


% ----------------------------------------------------------------------- %

%%

% ---- Task 5: Create data and 95% confidence intervals for each -------- %
% ---- of the parameters l0 and a                                -------- %

len = 10000;

x_tot = zeros(len,2); % Init variables
sigma2_tot = zeros(len,1);

for i = 1:len % Do this 10000 times
    [A_gen, y_gen] = get_pathloss(d_gen,d0,l_d0,a,sigma2); % Generate data
    x_tot(i,:) = (A_gen \ y_gen)'; % Calculate and store current dataset a and ld0
    r_gen = y_gen - A_gen * x_tot(i,:)';
    sigma2_tot(i) = (r_gen'*r_gen)/(length(r_gen)-2); % Calculate and store current dataset variance
end

l_d0_tot = x_tot(:,1);
a_tot = x_tot(:,2);

[l_d0_low, l_d0__high, ul, il] = conf_interval(l_d0_tot) % Print confidence interval for ld0
[a_low, a_high, ua, ia] = conf_interval(a_tot) % Same for a

bins=40;

figure(8)
histogram(l_d0_tot,bins) % Plot histogram of ld0

title('Histogram l0')
xlabel('Value [dB]')
ylabel('Count')

saveas(gcf, 'hista.png', 'png')

figure(9)
histogram(a_tot,bins) % Plot histogram for a

title('Histogram a')
xlabel('Value [dB/m]')
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

%{
Another help function which calculates highest and lowest value in
confidence interval as well as mean and +/- value
%}
function [C1, C2, u, i] = conf_interval(data)
    n = length(data);
    u = mean(data);
    res = data - u;
    s = sqrt(sum(res.^2)/(n-1)); % Get standard deviation
    i = s * 1.9600 / sqrt(n); % Get +/- value
    C1 = u - i;
    C2 = u + i;
end
